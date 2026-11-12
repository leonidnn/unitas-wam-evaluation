"""Compare fixed scratch/finetuned LIBERO runs and their initial states."""

import argparse
import copy
import json
from pathlib import Path
import statistics

import numpy as np

from summarize_runs import collect


def protocol(directory):
    paths = list(Path(directory).glob('*_protocol.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one protocol in {directory}')
    return json.loads(paths[0].read_text(encoding='utf-8'))


def comparable(value, native_temporal_config=False):
    value = copy.deepcopy(value)
    value.pop('scene_condition')
    for key in ('checkpoint', 'checkpoint_sha256'):
        value['server'].pop(key)
    if native_temporal_config:
        value['server'].pop('temporal_spec')
    return value


def compare(finetuned, scratch, count=10, native_temporal_config=False):
    root_paths = {'finetuned': Path(finetuned), 'scratch': Path(scratch)}
    groups, protocols, rows = {}, {}, {}
    reference = None
    for model, root in root_paths.items():
        for condition, scale in (('baseline', 1.0), ('dim_light', 0.5)):
            label = f'{model}/{condition}'
            directory = root / condition
            p = protocol(directory)
            if p['scene_condition'] != {'light_scale': scale}:
                raise ValueError(f'Unexpected scene condition: {label}')
            if p['server']['checkpoint'] != f'unitas_libero_{model}.pt':
                raise ValueError(f'Unexpected checkpoint: {label}')
            if native_temporal_config:
                horizon = 20 if model == 'finetuned' else 30
                expected = dict(prediction_horizon=horizon, history_horizon=horizon,
                                gripper_flow_fps=20.0, scene_flow_horizon=horizon,
                                scene_flow_fps=20.0)
                if p['server']['temporal_spec'] != expected:
                    raise ValueError(f'Unexpected native temporal spec: {label}')
            if reference is None:
                reference = comparable(p, native_temporal_config)
            if comparable(p, native_temporal_config) != reference:
                raise ValueError(f'Incompatible protocol: {label}')
            protocols[label] = p
            rows[label], groups[label] = collect(directory, count)
            steps = [r['steps'] for r in rows[label].values()]
            groups[label]['mean_steps_to_termination'] = statistics.mean(steps) if steps else None
            groups[label]['median_steps_to_termination'] = statistics.median(steps) if steps else None
    for model in root_paths:
        if protocols[f'{model}/baseline']['server'] != protocols[f'{model}/dim_light']['server']:
            raise ValueError(f'Checkpoint/config changed between conditions: {model}')
    paired = {}
    for condition in ('baseline', 'dim_light'):
        pairs = []
        for index in sorted(rows[f'finetuned/{condition}'].keys() & rows[f'scratch/{condition}'].keys()):
            left, right = (rows[f'{model}/{condition}'][index] for model in root_paths)
            for key in ('suite', 'task_id', 'episode_id', 'max_steps', 'settle_steps'):
                if left[key] != right[key]:
                    raise ValueError(f'Unpaired {key}: {condition}/{index}')
            initial = []
            for model in root_paths:
                stem = f"{left['suite']}_task{left['task_id']:02d}_episode{index:03d}_replay"
                with np.load(root_paths[model] / condition / stem / 'states.npz', allow_pickle=False) as data:
                    mask = data['phases'] <= 1
                    if int(mask.sum()) != left['settle_steps'] + 1:
                        raise ValueError('Missing initial/settle frames')
                    initial.append((data['states'][mask].copy(), data['times'][mask].copy()))
            identical = all(np.array_equal(a, b) for a, b in zip(*initial))
            if not identical:
                raise ValueError(f'Initial/settle mismatch: {condition}/{index}')
            pairs.append({'episode_id': index, 'finetuned_success': left['success'],
                          'scratch_success': right['success'], 'finetuned_steps': left['steps'],
                          'scratch_steps': right['steps'], 'initial_and_settle_identical': identical})
        paired[condition] = pairs
    return {'complete': all(not g['missing_episode_ids'] for g in groups.values()),
            'groups': groups, 'protocols': protocols, 'paired': paired,
            'native_temporal_config': native_temporal_config,
            'interpretation': 'Published configurations differ in history/prediction duration; differences cannot be attributed solely to pretraining.',
            'timing_scope': 'Synchronized Policy.act including preprocessing/postprocessing; different sessions, descriptive comparison.',
            'scope': 'One task and fixed initial states; no claim of general superiority.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finetuned', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-temporal-config', action='store_true')
    args = parser.parse_args()
    result = compare(args.finetuned, args.scratch, native_temporal_config=args.native_temporal_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'complete': result['complete'], 'successes': {
        k: v['successes'] for k, v in result['groups'].items()}}))


if __name__ == '__main__':
    main()
