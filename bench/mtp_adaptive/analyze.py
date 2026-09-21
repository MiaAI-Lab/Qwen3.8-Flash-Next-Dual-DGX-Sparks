# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read paired benchmark artifacts; no inference or configuration changes."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def interval(values, seed=721, iterations=10000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return [(math.exp(samples[i]) - 1) * 100 for i in (int(.025 * iterations), int(.975 * iterations))]


def analyze(root):
    p = root / 'waves.jsonl'
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
    compact = root / 'waves.csv'
    if not p.exists() and compact.exists():
        with compact.open(encoding='utf-8', newline='') as handle:
            for raw in csv.DictReader(handle):
                row = dict(raw)
                for k in ('rep', 'concurrency'):
                    row[k] = int(row[k])
                for k in ('aggregate_decode_tps', 'e2e_tps', 'mean_stream_tps', 'mean_ttft_s'):
                    row[k] = float(row[k])
                row['isolated'] = row['isolated'].lower() == 'true'
                row['delta'] = {
                    'vllm:spec_decode_num_drafts_total': float(raw['drafts']),
                    'vllm:spec_decode_num_draft_tokens_total': float(raw['drafted']),
                    'vllm:spec_decode_num_accepted_tokens_total': float(raw['accepted']),
                    'vllm:num_preemptions_total': float(raw['preemptions']),
                }
                rows.append(row)
    groups = defaultdict(dict)
    for r in rows:
        key = (r['case'], r['concurrency'])
        assert (r['rep'], r['arm']) not in groups[key], 'Duplicate run; do not silently cherry-pick'
        groups[key][(r['rep'], r['arm'])] = r
    table = []
    for (case, c), group in sorted(groups.items()):
        reps = sorted({rep for rep, _ in group if (rep, 'fixed3') in group and (rep, 'adaptive') in group})
        if not reps:
            continue
        metric = 'e2e_tps' if case == 'mixed_rolling' else 'aggregate_decode_tps'
        logs = [math.log(group[(rep, 'adaptive')][metric] / group[(rep, 'fixed3')][metric]) for rep in reps]
        row = dict(case=case, concurrency=c, metric=metric, pairs=len(reps),
                   paired_geomean_change_pct=100 * (math.exp(statistics.mean(logs)) - 1),
                   paired_bootstrap_ci95_pct=interval(logs),
                   paired_changes_pct=[100 * (math.exp(v) - 1) for v in logs])
        for arm in ('fixed3', 'adaptive'):
            ar = [group[(rep, arm)] for rep in reps]
            vals = sorted(r[metric] for r in ar)
            drafts = sum(r['delta']['vllm:spec_decode_num_drafts_total'] for r in ar)
            drafted = sum(r['delta']['vllm:spec_decode_num_draft_tokens_total'] for r in ar)
            accepted = sum(r['delta']['vllm:spec_decode_num_accepted_tokens_total'] for r in ar)
            row[arm] = dict(mean=statistics.mean(vals), median=statistics.median(vals),
                            full_range=[min(vals), max(vals)],
                            trimmed_range=[vals[1], vals[-2]] if len(vals) >= 5 else None,
                            mean_stream_tps=statistics.mean(r['mean_stream_tps'] for r in ar),
                            mean_ttft_s=statistics.mean(r['mean_ttft_s'] for r in ar),
                            mean_k=drafted / drafts if drafts else None,
                            accepted_per_step=accepted / drafts if drafts else None,
                            valid=all(r['isolated'] and r['delta'].get('vllm:num_preemptions_total', 0) == 0 for r in ar))
        table.append(row)
    state = json.loads((root / 'state.json').read_text())
    supported = bool(state['complete']) and len(table) == 21 and all(
        r['fixed3']['valid'] and r['adaptive']['valid'] and r['paired_bootstrap_ci95_pct'] and
        r['paired_bootstrap_ci95_pct'][0] > 0 for r in table)
    return dict(state=state, complete_waves=len(rows), table=table,
                all_cells_positive_evidence=supported,
                limitations=['Bootstrap estimates from six paired repetitions (rolling three), not a broad quality evaluation.',
                             'Both arms have IndexShare and maxK4 patched runner; this is not stock-upstream comparison.',
                             'CI is per-cell, not multiplicity-adjusted. No inference from partial results.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    result = analyze(args.root)
    if args.compact:
        print('waves', result['complete_waves'], 'phase', result['state']['phase'])
        for r in result['table']:
            ci = r['paired_bootstrap_ci95_pct']
            print(f"{r['case']:15} C{r['concurrency']} n{r['pairs']} "
                  f"{r['fixed3']['mean']:.2f} -> {r['adaptive']['mean']:.2f} "
                  f"{r['paired_geomean_change_pct']:+.2f}% CI={ci}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
