"""Summarize the preselected paired episodes without selecting by success."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def timing_summary(rows):
    valid = [float(row['elapsed_seconds']) for row in rows if row['status'] == 'ok']
    return {'successful_calls': len(valid), 'failed_calls': sum(row['status'] != 'ok' for row in rows),
            'mean_seconds': statistics.mean(valid) if valid else None,
            'median_seconds': statistics.median(valid) if valid else None,
            'min_seconds': min(valid) if valid else None, 'max_seconds': max(valid) if valid else None}


def collect(directory, count):
    directory = Path(directory)
    results = {}
    for path in sorted(directory.glob('*_task*_episode*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        index = row['episode_id']
        if index in results:
            raise ValueError(f'Duplicate episode {index} in {directory}')
        if index not in range(count):
            raise ValueError(f'Unexpected episode {index}')
        results[index] = row
    missing = sorted(set(range(count)) - results.keys())
    return results, {
        'directory': str(directory), 'expected_episodes': list(range(count)),
        'completed_episodes': len(results), 'missing_episode_ids': missing,
        'successes': sum(row['success'] for row in results.values()),
        'success_rate': sum(row['success'] for row in results.values()) / count if not missing else None,
        'client_roundtrip': timing_summary(read_jsonl(directory / 'policy-request-timing.jsonl')),
        'server_policy_act': timing_summary(read_jsonl(directory / 'server-policy-timing.jsonl')),
        'episodes': [results[index] for index in sorted(results)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--changed', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    protocols = []
    for directory in (args.baseline, args.changed):
        paths = list(directory.glob('*_protocol.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one protocol in {directory}')
        protocols.append(json.loads(paths[0].read_text(encoding='utf-8')))
    for key in ('suite', 'server', 'seed', 'max_steps', 'settle_steps', 'control', 'cameras', 'resolution'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f'Conditions have different {key}')
    base, base_summary = collect(args.baseline, args.episodes)
    changed, changed_summary = collect(args.changed, args.episodes)
    paired = []
    for index in sorted(base.keys() & changed.keys()):
        for key in ('suite', 'task_id', 'seed'):
            if key in base[index] and key in changed[index] and base[index][key] != changed[index][key]:
                raise ValueError(f'Unpaired {key} in episode {index}')
        paired.append({'episode_id': index, 'baseline': base[index]['success'],
                       'changed': changed[index]['success']})
    losses = sum(row['baseline'] and not row['changed'] for row in paired)
    gains = sum(not row['baseline'] and row['changed'] for row in paired)
    discordant = losses + gains
    p_value = min(1., 2*sum(math.comb(discordant, k) for k in range(min(losses, gains)+1))
                  / 2**discordant) if discordant else 1.
    complete = not base_summary['missing_episode_ids'] and not changed_summary['missing_episode_ids']
    summary = {'complete': complete, 'baseline': base_summary, 'changed': changed_summary,
               'condition_protocols': protocols,
               'paired': paired, 'losses': losses, 'gains': gains,
               'exact_paired_mcnemar_p': p_value if complete else None,
               'scope': 'One task, fixed initial-state list. Timings are descriptive; calls are correlated within episodes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'complete': complete, 'baseline_sr': base_summary['success_rate'],
                      'changed_sr': changed_summary['success_rate'], 'paired_episodes': len(paired)}))


if __name__ == '__main__':
    main()
