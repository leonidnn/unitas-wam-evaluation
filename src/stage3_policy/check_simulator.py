"""Check the selected LIBERO task, RGB-D cameras and illumination intervention."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from record_libero import set_lighting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from libero.libero import benchmark
    from evaluation.libero.client import create_environment
    from evaluation.libero.environment import model_observation

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()['libero_90']()
    task_id = next(i for i in range(suite.n_tasks)
                   if suite.get_task(i).language == 'put the white mug on the plate')
    states = suite.get_task_init_states(task_id)
    if len(states) < 10:
        raise ValueError('Need ten distinct initial states')
    env, observation, instruction, task_name = create_environment('libero_90', task_id, 0, 42)
    try:
        before = env.sim.get_state().flatten().copy()
        base = model_observation(env, observation)
        lights = set_lighting(env, .5)
        changed = model_observation(env, observation)
        after = env.sim.get_state().flatten().copy()
        assert np.array_equal(before, after), 'Lighting modified the physical state'
        stats = {}
        for name, camera in base['cameras'].items():
            dim = changed['cameras'][name]
            assert np.array_equal(camera['depth'], dim['depth']), 'Lighting changed depth'
            assert np.any(camera['rgb'] != dim['rgb']), 'Lighting did not change RGB'
            assert np.isfinite(camera['depth']).all() and (camera['depth'] > 0).any()
            imageio.imwrite(out/f'{name}-baseline.png', camera['rgb'])
            imageio.imwrite(out/f'{name}-dim.png', dim['rgb'])
            stats[name] = {'baseline_mean_rgb': float(camera['rgb'].mean()),
                           'dim_mean_rgb': float(dim['rgb'].mean()),
                           'changed_rgb_fraction': float(np.any(camera['rgb'] != dim['rgb'], axis=-1).mean())}
        result = {'suite': 'libero_90', 'task_id': task_id, 'task_name': task_name,
                  'instruction': instruction, 'initial_states': len(states),
                  'physical_state_unchanged': True, 'depth_unchanged': True,
                  'modified_lights': lights, 'cameras': stats}
        (out/'passed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        env.close()


if __name__ == '__main__':
    main()
