"""Capture a causal RGB-D window and execute its predicted open-loop EEF chunk."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage3_policy'))
from replay_data import StateRecording, load_recording, sha256  # noqa: E402
from record_libero import set_lighting  # noqa: E402


def restore_environment(recording, index):
    from evaluation.libero.client import create_environment
    saved_model, _, metadata, arrays = load_recording(recording)
    if not 0 <= index < len(arrays['states']):
        raise ValueError('Anchor frame is outside the recording')
    random.seed(metadata['seed'])
    np.random.seed(metadata['seed'])
    env, _, instruction, _ = create_environment(metadata['suite'], metadata['task_id'],
                                                metadata['episode_id'], metadata['seed'])
    try:
        set_lighting(env, metadata.get('condition', {}).get('light_scale', 1.0))
        model = env.sim.model._model
        for field in ('nq', 'nv', 'nbody', 'ngeom'):
            if getattr(model, field) != getattr(saved_model, field):
                raise ValueError('Reconstructed scene layout differs')
        for i in range(model.nbody):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) != mujoco.mj_id2name(
                    saved_model, mujoco.mjtObj.mjOBJ_BODY, i):
                raise ValueError('Reconstructed body ordering differs')
        for field in ('body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'cam_pos', 'cam_quat',
                      'body_mass', 'body_inertia', 'geom_friction'):
            if not np.array_equal(getattr(model, field), getattr(saved_model, field)):
                raise ValueError(f'Reconstructed scene differs in {field}')
        restore_state(env, arrays['states'][index], metadata['state_spec'])
        env.env.robots[0].controller.use_delta = False
        return env, instruction, metadata, arrays
    except BaseException:
        env.close()
        raise


def restore_state(env, state, state_spec):
    mujoco.mj_setState(env.sim.model._model, env.sim.data._data, state, state_spec)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.env.robots[0].controller.update(force=True)
    return env.env._get_observations(force_update=True)


def capture(args):
    from evaluation.libero.environment import model_observation, state_observation
    from unitas.inference.observations import save_observation
    env, instruction, metadata, arrays = restore_environment(args.recording, args.frame)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        model, data = env.sim.model._model, env.sim.data._data
        history = []
        for index in range(max(0, args.frame - 20), args.frame + 1):
            observation = restore_state(env, arrays['states'][index], metadata['state_spec'])
            history.append(state_observation(env, observation))
        payload = model_observation(env, observation)
        save_observation(args.output/'observation.json', domain='libero', observation=payload,
                         instruction=instruction, history=history)
        candidates, colors, bodies = [], [], []
        for camera_name, camera in payload['cameras'].items():
            height, width = camera['depth'].shape
            segmentation = env.sim.render(camera_name=camera_name, height=height, width=width,
                                          segmentation=True)[::-1].copy()
            geometry = segmentation[..., 1]
            valid = ((segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
                     & (geometry >= 0) & np.isfinite(camera['depth']) & (camera['depth'] > 0))
            v, u = np.nonzero(valid)
            z = camera['depth'][v, u]
            k = camera['intrinsic']
            xyz_camera = np.stack([(u+.5-k[0, 2])*z/k[0, 0],
                                   (v+.5-k[1, 2])*z/k[1, 1], z], axis=-1)
            world_from_camera = camera['camera_to_world']
            candidates.append(xyz_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3])
            colors.append(camera['rgb'][v, u])
            bodies.append(model.geom_bodyid[geometry[v, u]])
        xyz, rgb, body_ids = np.concatenate(candidates), np.concatenate(colors), np.concatenate(bodies)
        rng = np.random.default_rng(42)
        selected = []
        for body_id in np.unique(body_ids):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or 'world'
            if name.startswith(('robot0', 'gripper')):
                continue
            ids = np.flatnonzero(body_ids == body_id)
            selected.extend(rng.choice(ids, min(128, len(ids)), replace=False).tolist())
        if not selected:
            raise ValueError('No non-robot rigid scene queries')
        selected = np.asarray(selected)
        xyz, rgb, body_ids = xyz[selected], rgb[selected], body_ids[selected]
        rotation = data.xmat[body_ids].reshape(-1, 3, 3)
        local = np.einsum('ni,nij->nj', xyz-data.xpos[body_ids], rotation)
        world_from_base = payload['world_from_robot_base']
        base_from_world = np.linalg.inv(world_from_base)
        xyz_model = xyz @ base_from_world[:3, :3].T + base_from_world[:3, 3]
        np.savez_compressed(args.output/'queries.npz', xyz0=xyz.astype(np.float32), colors=rgb)
        np.savez_compressed(args.output/'tracking.npz', local_xyz=local, body_ids=body_ids,
                            world_from_base=world_from_base, xyz0_model=xyz_model.astype(np.float32))
        info = {'recording': str(args.recording), 'frame': args.frame, 'instruction': instruction,
                'source_manifest_sha256': sha256(args.recording/'manifest.json'),
                'source_states_sha256': sha256(args.recording/'states.npz'),
                'code_sha256': sha256(__file__),
                'state_spec': metadata['state_spec'], 'points': len(xyz),
                'selection': 'up to 128 visible pixel queries per non-robot body across two cameras, seed42',
                'body_points': {str(b): int((body_ids == b).sum()) for b in np.unique(body_ids)},
                'history_last_frame': args.frame, 'future_used_for_input': False}
        (args.output/'capture.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print(json.dumps(info, indent=2))
    finally:
        env.close()


def visibility(points_world, cameras, tolerance=0.01):
    result = np.zeros(len(points_world), bool)
    for camera in cameras.values():
        inverse = np.linalg.inv(camera['camera_to_world'])
        xyz = points_world @ inverse[:3, :3].T + inverse[:3, 3]
        z = xyz[:, 2]
        k = camera['intrinsic']
        uv = xyz[:, :2] @ k[:2, :2].T / np.maximum(z[:, None], 1e-8) + k[:2, 2]
        u, v = np.floor(uv).astype(int).T
        h, w = camera['depth'].shape
        valid = (z > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
        ids = np.flatnonzero(valid)
        result[ids] |= np.abs(camera['depth'][v[ids], u[ids]] - z[ids]) <= tolerance
    return result


def execute(args):
    from evaluation.libero.environment import model_observation
    info = json.loads((args.window/'capture.json').read_text())
    env, _, metadata, arrays = restore_environment(Path(info['recording']), info['frame'])
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        observation = restore_state(env, arrays['states'][info['frame']], metadata['state_spec'])
        payload = model_observation(env, observation)
        with np.load(args.window/'tracking.npz', allow_pickle=False) as t:
            local, body_ids, world_from_base = t['local_xyz'], t['body_ids'], t['world_from_base']
        with np.load(args.prediction, allow_pickle=False) as p:
            poses, predicted_scene, fps = p['eef_poses'], p['scene'], float(p['fps'])
        if fps != 20 or len(predicted_scene) != len(poses)+1:
            raise ValueError('Expected matching EEF/scene horizon at 20 Hz')
        base_from_world = np.linalg.inv(world_from_base)
        with np.load(args.window/'actions.npz', allow_pickle=False) as a:
            if str(a['prediction_sha256']) != sha256(args.prediction):
                raise ValueError('Actions do not correspond to this prediction')
            actions = a['actions'].copy()
        if actions.shape != (len(poses), 7) or not np.isfinite(actions).all():
            raise ValueError('Invalid action chunk')
        if not np.allclose(payload['world_from_robot_base'], world_from_base, atol=1e-6):
            raise ValueError('Robot base changed since capture')
        model, data = env.sim.model._model, env.sim.data._data
        recorder = StateRecording(args.output/'replay', model, data, {
            'source_window': str(args.window), 'suite': metadata['suite'], 'task_id': metadata['task_id'],
            'episode_id': metadata['episode_id'], 'seed': metadata['seed'],
            'execution': 'all predicted EEF actions, no replanning'})
        scene, visible, measured_tcp = [], [], []

        def capture_frame():
            pose = data.xmat[body_ids].reshape(-1, 3, 3)
            world = np.einsum('nij,nj->ni', pose, local) + data.xpos[body_ids]
            current = env.env._get_observations(force_update=True)
            measured = model_observation(env, current)
            scene.append(world @ base_from_world[:3, :3].T + base_from_world[:3, 3])
            visible.append(visibility(world, measured['cameras']))
            measured_tcp.append(measured['observed_tcp_world'])

        try:
            capture_frame()
            for action in actions:
                env.step(action)
                recorder.capture(action, 2)
                capture_frame()
            scene = np.asarray(scene, np.float32)
            if not np.allclose(scene[0], predicted_scene[0], atol=1e-5, rtol=0):
                raise ValueError('Reference initial points differ from prediction')
            np.savez_compressed(args.output/'reference.npz', scene=scene, visibility=np.asarray(visible),
                                valid=np.ones(scene.shape[:2], bool), fps=fps, body_ids=body_ids,
                                measured_tcp_world=np.asarray(measured_tcp), actions=actions)
            recorder.finish({'steps': len(actions), 'replanning': False})
            print({'reference': str(args.output/'reference.npz'), 'points': scene.shape[1]})
        except BaseException as error:
            recorder.finish(error=error)
            raise
    finally:
        env.close()


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['MPLBACKEND'] = 'Agg'
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--recording', type=Path, required=True)
    c.add_argument('--frame', type=int, required=True)
    c.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('execute')
    e.add_argument('--window', type=Path, required=True)
    e.add_argument('--prediction', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (capture if args.command == 'capture' else execute)(args)
