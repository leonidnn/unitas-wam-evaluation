"""Construct equal-observation states with opposite unobserved object velocities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest_observation(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        else:
            array = np.asarray(item)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed-m-s', type=float, default=0.2)
    parser.add_argument('--height-m', type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.speed_m_s <= 1 or not 0 < args.height_m <= 0.3:
        parser.error('Use speed in (0,1] m/s and height in (0,0.3] m')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['MPLBACKEND'] = 'Agg'
    from evaluation.libero.client import create_environment
    from evaluation.libero.environment import model_observation, settle_then_enable_absolute_control
    from scipy.spatial.transform import Rotation
    import mujoco

    args.output.mkdir(parents=True, exist_ok=False)
    env, observation, instruction, _ = create_environment('libero_90', 72, 0, 42)
    try:
        observation, _ = settle_then_enable_absolute_control(env, observation, 10)
        model, data = env.sim.model._model, env.sim.data._data
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
                 for i in range(model.nbody)]
        body_id = next(i for i, name in enumerate(names) if name == 'porcelain_mug_1_main')
        joint = int(model.body_jntadr[body_id])
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected a free-joint mug')
        qpos_adr, dof_adr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        data.qpos[qpos_adr + 2] += args.height_m
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, state, state_spec)
        controller = env.env.robots[0].controller
        controller.update(force=True)
        hold = np.r_[controller.ee_pos, Rotation.from_matrix(controller.ee_ori_mat).as_rotvec(), -1.]
        trajectories, hashes, payloads = [], [], []
        for sign in (-1, 1):
            mujoco.mj_setState(model, data, state, state_spec)
            data.qvel[dof_adr] = sign * args.speed_m_s
            mujoco.mj_forward(model, data)
            observation = env.env._get_observations(force_update=True)
            controller.update(force=True)
            payload = model_observation(env, observation)
            payloads.append(payload)
            hashes.append(digest_observation(payload))
            # Удерживаю робот на месте; известное скрытое различие только в скорости кружки.
            trajectory = [data.xpos[body_id].copy()]
            for _ in range(20):
                env.step(hold)
                trajectory.append(data.xpos[body_id].copy())
            trajectories.append(trajectory)
        if hashes[0] != hashes[1]:
            raise ValueError('Input observations differ: cannot claim indistinguishability')
        trajectories = np.asarray(trajectories)
        separation = np.linalg.norm(trajectories[0, 1:] - trajectories[1, 1:], axis=-1)
        result = {'instruction': instruction, 'task_id': 72, 'episode_id': 0,
                  'height_increment_m': args.height_m, 'object_velocity_x_m_s': [-args.speed_m_s, args.speed_m_s],
                  'input_hashes': hashes, 'equal_observation': True,
                  'history': 'same fixed robot history for both states',
                  'horizon': 20, 'fps': 20,
                  'mean_future_center_separation_m': float(separation.mean()),
                  'final_center_separation_m': float(separation[-1]),
                  'lower_bound_pair_average_ade_m': float(separation.mean()/2),
                  'lower_bound_pair_average_fde_m': float(separation[-1]/2),
                  'interpretation': 'Any shared point prediction for these identical inputs has pair-average error at least half their separation. This is an observability limitation, not an empirical UNITAS score.'}
        np.savez_compressed(args.output/'reference-centers.npz', centers=trajectories, fps=20.)
        (args.output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        # Сохраняю одинаковое наблюдение как данные эксперимента, без будущих состояний во входе.
        import json_numpy
        (args.output/'observation.json').write_text(json.dumps({
            'domain': 'libero', 'language_instruction': instruction,
            'observation': json_numpy.dumps(payloads[0]), 'history': '[]'}), encoding='utf-8')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
