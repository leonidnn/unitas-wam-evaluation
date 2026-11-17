"""Test a benchmark success after explicit release and one second of waiting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from replay_data import StateRecording, load_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['MPLBACKEND'] = 'Agg'
    from evaluation.libero.client import create_environment
    from record_libero import set_lighting

    saved_model, _, metadata, arrays = load_recording(args.input)
    if not metadata.get('result', {}).get('success'):
        raise ValueError('Terminal-stability check requires a successful source episode')
    env, _, _, _ = create_environment(metadata['suite'], metadata['task_id'],
                                       metadata['episode_id'], metadata['seed'])
    try:
        set_lighting(env, metadata.get('condition', {}).get('light_scale', 1.0))
        model, data = env.sim.model._model, env.sim.data._data
        if (model.nq, model.nv, model.nbody, model.ngeom) != (
                saved_model.nq, saved_model.nv, saved_model.nbody, saved_model.ngeom):
            raise ValueError('Reconstructed scene layout differs from recording')
        for i in range(model.nbody):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) != mujoco.mj_id2name(
                    saved_model, mujoco.mjtObj.mjOBJ_BODY, i):
                raise ValueError('Reconstructed body ordering differs')
        mujoco.mj_setState(model, data, arrays['states'][-1], metadata['state_spec'])
        mujoco.mj_forward(model, data)
        controller = env.env.robots[0].controller
        controller.use_delta = False
        controller.update(force=True)
        action = np.concatenate([controller.ee_pos.copy(),
                                 Rotation.from_matrix(controller.ee_ori_mat).as_rotvec(), [-1]])
        source_success = bool(env.check_success())
        if not source_success:
            raise ValueError('Restored terminal state does not reproduce source success')
        recorder = StateRecording(args.output, model, data, {
            'source': str(args.input), 'suite': metadata['suite'], 'task_id': metadata['task_id'],
            'episode_id': metadata['episode_id'], 'seed': metadata['seed'],
            'intervention': 'open gripper, hold measured terminal TCP for 20 control steps'})
        flags = []
        try:
            for _ in range(20):
                env.step(action)
                recorder.capture(action, 2)
                flags.append(bool(env.check_success()))
            result = {'source_success': source_success, 'final_success': flags[-1],
                      'success_after_each_step': flags, 'steps': 20, 'duration_s': 1.0,
                      'interpretation': 'Post-success intervention; does not alter original benchmark SR'}
            recorder.finish(result)
            (args.output/'stability.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result, indent=2))
        except BaseException as error:
            recorder.finish(error=error)
            raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
