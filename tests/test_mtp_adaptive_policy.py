# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'files/mtp_adaptive'))
from qwen_mtp_adaptive import DepthController, runtime_depth, COST_PRIOR
import qwen_mtp_adaptive as policy

logging.disable(logging.CRITICAL)


class Tests(unittest.TestCase):
    def test_disabled_factory_is_inert(self):
        with patch.object(policy, 'ENABLED', False):
            self.assertIsNone(policy.create_controller(None))

    def test_conflicting_dynamic_sd_refused(self):
        config = NS(speculative_config=NS(method='mtp', num_speculative_tokens_per_batch_size=[[1, 8, 3]]),
                    num_speculative_tokens=4, use_v2_model_runner=True)
        with patch.object(policy, 'ENABLED', True):
            with self.assertRaisesRegex(ValueError, 'Dynamic SD'):
                policy.create_controller(config)

    def test_enabled_missing_control_starts_adaptive(self):
        with tempfile.TemporaryDirectory() as d:
            controller = DepthController(control=Path(d) / 'absent.json')
            self.assertEqual(controller.mode, 'adaptive')
            self.assertEqual(controller.choose(['r']), 3)

    def make(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / 'control.json'
        p.write_text('{"mode":"adaptive"}')
        now = [0.0]
        c = DepthController(control=p, clock=lambda: now[0])
        return c, now, p

    def simulate(self, q, n=800, seed=17):
        c, now, _ = self.make()
        rng = random.Random(seed)
        history = []
        for i in range(n):
            k = c.choose(['r'])
            qs = q(i) if callable(q) else q
            accepted = 0
            for prob in qs[:k]:
                if rng.random() >= prob:
                    break
                accepted += 1
            now[0] += COST_PRIOR[k]
            c.observe(NS(num_scheduled_tokens={'r': k + 1},
                         scheduled_spec_decode_tokens={'r': [0] * k}),
                      NS(sampled_token_ids=[[0] * (1 + accepted)], req_id_to_index={'r': 0}))
            history.append(k)
        return c, history

    def test_warmup(self):
        self.assertEqual(runtime_depth(0, 4), 4)
        for k in range(1, 5):
            self.assertEqual(runtime_depth(k, 4), k)
        with self.assertRaises(AssertionError):
            runtime_depth(5, 4)

    def test_low_acceptance_prefers_one(self):
        c, h = self.simulate([.25, .3, .4, .4])
        self.assertGreater(h[100:].count(1) / len(h[100:]), .80)

    def test_high_acceptance_does_not_shorten_english(self):
        c, h = self.simulate([.85, .85, .72, .65])
        self.assertGreater(sum(k >= 3 for k in h[100:]) / len(h[100:]), .9)
        self.assertIn(4, h)

    def test_phase_change_both_directions(self):
        c, h = self.simulate(lambda i: [.2, .25, .3, .3] if i < 250 or i > 550 else [.98] * 4)
        self.assertGreater(h[150:250].count(1), 70)
        self.assertGreater(h[400:550].count(4), 100)
        self.assertGreater(h[700:].count(1), 70)

    def test_censoring_does_not_mark_unseen_tails_rejected(self):
        c, _, _ = self.make()
        c.records.extend([(4, (4,))] * 40)
        c.estimates()
        tail = c.q[1:]
        c.records.clear()
        c.records.extend([(1, (0,))] * 40)
        _, _, survival, _ = c.estimates()
        self.assertEqual(c.q[1:], tail)
        self.assertLess(survival[0], .02)

    def test_same_samples_determine_all_rewards(self):
        c, _, _ = self.make()
        c.records.extend([(3, (a,)) for a in [0, 1, 2, 3] * 16])
        _, _, p, _ = c.estimates()
        self.assertAlmostEqual(p[0], .75, delta=.015)
        self.assertAlmostEqual(p[1], .50, delta=.025)
        self.assertAlmostEqual(p[2], .25, delta=.03)

    def test_bad_json_fixed_and_parameter_refresh(self):
        c, now, p = self.make()
        p.write_text('{"mode":"fixed","k":2}')
        self.assertEqual(c.choose(['r']), 2)
        p.write_text('{"mode":"fixed","k":99}')
        now[0] = 2
        self.assertEqual(c.choose(['r']), 2)
        p.write_text('{"mode":"adaptive","policy":{"interval":24}}')
        now[0] = 4
        self.assertEqual(c.choose(['r']), 3)
        self.assertEqual(c.params['interval'], 24)

    def test_churn_stale_prefill_and_timing_outlier(self):
        c, now, _ = self.make()
        c.choose(['r'])
        c.settle = 0
        c.previous_time = 0
        now[0] = .05
        c.observe(NS(num_scheduled_tokens={'other': 4}), NS())
        self.assertEqual(len(c.records), 0)
        self.assertEqual(c.choose(['new']), 3)
        self.assertEqual(c.steps, 0)
        c.previous_time = 0
        c.settle = 0
        now[0] = 100
        c.observe(NS(num_scheduled_tokens={'new': 4}, scheduled_spec_decode_tokens={'new': [0] * 3}), NS())
        self.assertEqual(c.steps, 0)

    def test_batch_accounting(self):
        c, now, _ = self.make()
        c.choose(['a', 'b'])
        c.settle, c.previous_time = 0, 0
        now[0] = .06
        c.observe(NS(num_scheduled_tokens={'a': 4, 'b': 4},
                     scheduled_spec_decode_tokens={'a': [0] * 3, 'b': [0] * 3}),
                  NS(sampled_token_ids=[[0] * 4, [0]], req_id_to_index={'a': 0, 'b': 1}))
        self.assertEqual(c.records[0], (3, (3, 0)))


if __name__ == '__main__':
    unittest.main()
