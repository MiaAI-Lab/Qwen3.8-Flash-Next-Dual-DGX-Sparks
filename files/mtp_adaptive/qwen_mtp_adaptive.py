# SPDX-License-Identifier: AGPL-3.0-or-later
"""Censor-aware MTP depth selection. CPU-only; no text or GPU synchronization.

The reward model uses conditional acceptance at each draft position, not
throughput from different text segments. Cost and reward are estimated separately.
Counterfactual rewards are estimates: changing K can change future token boundaries.
"""
from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time

ENABLED = os.environ.get('QWEN_MTP_ADAPTIVE') == '1'
CONTROL = Path(os.environ.get('QWEN_MTP_CONTROL',
                           str(Path.home() / '.cache/vllm/mtp-adaptive/control.json')))
LOG = logging.getLogger(__name__)
VERSION = 'v3-survival-cost'
DEFAULTS = dict(window=64, interval=16, margin=0.04, probe_steps=8,
                probe_interval=96)
# Bootstrapping only: measured on this TP2 kit before v3 (40/46/51/56 ms).
# Every request rescales these and replaces each observed K with its own median.
COST_PRIOR = {1: .040, 2: .046, 3: .051, 4: .056}


def runtime_depth(requested, maximum):
    depth = requested or maximum  # Synthetic startup warmups leave this at zero.
    assert 1 <= depth <= maximum
    return depth


class DepthController:
    def __init__(self, maximum=4, control=CONTROL, clock=time.monotonic):
        assert maximum == 4
        self.clock, self.control = clock, Path(control)
        self.mode, self.fixed = 'adaptive', 3
        self.read_after = 0.0
        self.signature = ()
        self.params = DEFAULTS.copy()
        self.reset()

    def reset(self):
        self.k = 3 if self.mode == 'adaptive' else self.fixed
        self.previous_time = None
        self.settle = 2
        self.steps = self.since_decision = 0
        self.records = deque(maxlen=self.params['window'])
        self.costs = {k: deque(maxlen=32) for k in COST_PRIOR}
        self.scales = deque(maxlen=32)
        self.q = [.5] * 4
        self.seen = [False] * 4
        self.last_probe = 0
        self.probe_left = 0
        self.anchor = self.k
        self.last_scores = {}

    def refresh(self):
        now = self.clock()
        if now < self.read_after:
            return
        self.read_after = now + 1.0
        try:
            obj = json.loads(self.control.read_text())
            mode, fixed = obj['mode'], int(obj.get('k', 3))
            if mode not in ('adaptive', 'fixed') or not 1 <= fixed <= 4:
                raise ValueError('Invalid mode/depth')
            params = DEFAULTS | obj.get('policy', {})
            if set(params) != set(DEFAULTS):
                raise ValueError('Unknown policy parameter')
            for key, low, high in [('window', 32, 256), ('interval', 8, 64),
                                   ('probe_steps', 4, 32), ('probe_interval', 32, 256)]:
                if type(params[key]) is not int or not low <= params[key] <= high:
                    raise ValueError('Invalid integer parameter')
            if not isinstance(params['margin'], (float, int)) or not 0 <= params['margin'] <= .25:
                raise ValueError('Invalid margin')
        except (OSError, ValueError, KeyError, TypeError):
            return
        if (mode, fixed, params) != (self.mode, self.fixed, self.params):
            self.mode, self.fixed, self.params = mode, fixed, params
            self.reset()
            LOG.warning('QWEN_MTP_V3 mode=%s initial_k=%d params=%s', mode, self.k, params)

    def choose(self, request_ids):
        self.refresh()
        signature = tuple(sorted(request_ids))
        if signature != self.signature:
            self.signature = signature
            self.reset()
        return self.k

    def switch(self, depth, reason):
        if depth != self.k:
            LOG.warning('QWEN_MTP_V3 k=%d -> %d reason=%s', self.k, depth, reason)
            self.k = depth
            # Ignore pipeline transition: target verifies old K while drafting new K.
            self.settle = 2
        self.since_decision = 0

    def estimates(self):
        # A K=1 step does NOT report failures at positions 2..4 (right censoring).
        # A rejected first draft does inform P(A>=1), but not conditional P2..P4.
        exposures, hits = [0] * 4, [0] * 4
        for depth, accepted in self.records:
            for a in accepted:
                for j in range(depth):
                    if a >= j:
                        exposures[j] += 1
                        hits[j] += int(a > j)
        for j in range(4):
            if exposures[j]:
                self.q[j] = (hits[j] + .5) / (exposures[j] + 1)
                self.seen[j] = True
        scale = statistics.median(self.scales) if self.scales else 1.0
        cost = {k: statistics.median(v) if len(v) >= 4 else COST_PRIOR[k] * scale
                for k, v in self.costs.items()}
        # TP2 regularizer: suppress non-monotonic cost estimates from timing noise.
        # This is a policy heuristic, not a physical law about kernel shapes.
        for k in (2, 3, 4):
            cost[k] = max(cost[k], cost[k - 1] * 1.01)
        p, reward, scores, survival = 1.0, 1.0, {}, []
        for j in range(4):
            p *= self.q[j]
            survival.append(p)
            reward += p
            scores[j + 1] = reward / cost[j + 1]
        return scores, cost, survival, exposures

    def decide(self):
        scores, cost, survival, exposures = self.estimates()
        self.last_scores = scores
        incumbent = self.anchor if self.probe_left == -1 else self.k
        best = max(scores, key=scores.get)
        if scores[best] <= scores[incumbent] * (1 + self.params['margin']):
            best = incumbent
        finished_probe = self.probe_left == -1
        self.probe_left = 0
        LOG.warning('QWEN_MTP_V3 decision step=%d k=%d best=%d survival=%s ms=%s tps=%s',
                    self.steps, self.k, best, [round(x, 3) for x in survival],
                    [round(cost[k] * 1000, 2) for k in cost],
                    [round(scores[k], 2) for k in scores])

        # Probe an unobserved outer position only if its maximum possible reward
        # could pay for the cost. Low-acceptance Chinese need not sweep all K.
        probe = None
        if not finished_probe and best == self.k and self.k < 4:
            nxt = self.k + 1
            upper_reward = 1 + sum(survival[:self.k]) + survival[self.k - 1]
            if not self.seen[self.k] and upper_reward / cost[nxt] > scores[self.k] * 1.04:
                probe = nxt
            elif self.steps - self.last_probe >= self.params['probe_interval']:
                # Prevent permanent lock-in when only an unobserved tail improves.
                probe = nxt
        if probe is not None:
            self.anchor = self.k
            self.probe_left = self.params['probe_steps']
            self.last_probe = self.steps
            self.switch(probe, 'tail-probe')
        else:
            self.switch(best, 'survival/cost')

    def observe(self, scheduled, output):
        now = self.clock()
        previous, self.previous_time = self.previous_time, now
        if self.mode != 'adaptive' or previous is None:
            return
        ids = tuple(sorted(scheduled.num_scheduled_tokens))
        if ids != self.signature or not ids:
            self.settle = 2
            return
        depths = {len(scheduled.scheduled_spec_decode_tokens.get(r, ())) for r in ids}
        if depths != {self.k} or any(scheduled.num_scheduled_tokens[r] != self.k + 1 for r in ids):
            self.settle = 2
            return  # Prefill, preemption, batch churn or K transition.
        if self.settle:
            self.settle -= 1
            return
        elapsed = now - previous
        if not math.isfinite(elapsed) or not 0 < elapsed < 2:
            return
        emitted = [len(output.sampled_token_ids[output.req_id_to_index[r]]) for r in ids]
        if any(not 1 <= n <= self.k + 1 for n in emitted):
            return
        self.records.append((self.k, tuple(n - 1 for n in emitted)))
        self.costs[self.k].append(elapsed)
        self.scales.append(elapsed / COST_PRIOR[self.k])
        self.steps += 1
        self.since_decision += 1
        if self.probe_left > 0:
            self.probe_left -= 1
            if self.probe_left == 0:
                self.probe_left = -1
                self.decide()
        elif self.since_decision >= self.params['interval']:
            self.decide()


def create_controller(config):
    if not ENABLED:
        return None
    assert config.speculative_config.method == 'mtp'
    assert config.num_speculative_tokens == 4
    assert config.use_v2_model_runner
    if getattr(config.speculative_config, 'num_speculative_tokens_per_batch_size', None):
        raise ValueError('Adaptive MTP cannot be combined with a batch-size Dynamic SD policy')
    LOG.warning('QWEN_MTP_V3 %s max_k=4; default=adaptive', VERSION)
    return DepthController()
