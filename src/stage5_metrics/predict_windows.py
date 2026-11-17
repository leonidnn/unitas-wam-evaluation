"""Prepare fixed windows and predict them using one loaded SceneFlow checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def convert_actions(window):
    from unitas.inference.adapters.libero import LiberoEEFActionConverter
    from unitas.inference.observations import decode_arrays
    payload = json.loads((window/'observation.json').read_text())
    observation = decode_arrays(payload['observation'])
    with np.load(window/'scene_flow.npz', allow_pickle=False) as p:
        converter = LiberoEEFActionConverter(max_translation_step_m=None, max_rotation_step_deg=None)
        chunk = converter.trajectory_to_actions(
            p['eef_poses'], p['eef_open'], world_from_robot_base=observation['world_from_robot_base'],
            current_tcp_world=observation['controller_tcp_world'],
            current_open_probability=float(np.clip(np.abs(observation['robot0_gripper_qpos']).mean()/0.04, 0, 1)),
            chunk_size=len(p['eef_poses']))
    np.savez_compressed(window/'actions.npz', actions=chunk.actions,
                        target_tcp_world=chunk.controller_tcp_world,
                        prediction_sha256=hashlib.sha256((window/'scene_flow.npz').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--florence-dir', type=Path, required=True)
    parser.add_argument('--convert-only', action='store_true')
    args = parser.parse_args()
    os.environ['MPLBACKEND'] = 'Agg'
    if args.convert_only:
        for window in args.windows:
            convert_actions(window)
        return
    for window in args.windows:
        if (window/'scene_flow.npz').exists():
            raise FileExistsError(window/'scene_flow.npz')
        if not (window/'input.npz').exists():
            subprocess.run([sys.executable, '-m', 'unitas.scene_flow.prepare',
                            '--checkpoint', str(args.checkpoint),
                            '--observation', str(window/'observation.json'),
                            '--queries', str(window/'queries.npz'), '--query-frame', 'world',
                            '--output', str(window/'input.npz')], check=True)
    from unitas.inference.runtime import load_model
    from unitas.scene_flow.io import load_input
    from unitas.scene_flow.inference import predict_scene
    import torch

    function_source = inspect.getsource(predict_scene)
    if torch.cuda.get_device_capability()[0] < 8 and 'enabled=False' not in function_source:
        raise ValueError('Apply the documented FP32 SceneFlow patch for T4 before prediction')
    start = time.perf_counter()
    model, _, device = load_model(args.checkpoint, args.florence_dir, 'cuda')
    load_duration = time.perf_counter()-start
    for window in args.windows:
        batch, xyz0, colors, domain, instruction = load_input(window/'input.npz', device)
        start = time.perf_counter()
        prediction = predict_scene(model, batch, xyz0, steps=10, chunk_size=256,
                                   policy_seed=42, scene_seed=210921)
        np.savez_compressed(window/'scene_flow.npz', **prediction, colors=colors,
                            domain=domain, instruction=instruction,
                            fps=float(batch['scene_flow_fps'].item()),
                            gripper_fps=float(batch['gripper_flow_fps'].item()),
                            policy_seed=42, scene_seed=210921, steps=10)
        summary = {'duration_s': time.perf_counter()-start, 'model_load_duration_s': load_duration,
                   'window': str(window), 'points': len(xyz0), 'chunk_size': 256,
                   'steps': 10, 'policy_seed': 42, 'scene_seed': 210921, 'precision': 'fp32',
                   'input_sha256': hashlib.sha256((window/'input.npz').read_bytes()).hexdigest(),
                   'prediction_sha256': hashlib.sha256((window/'scene_flow.npz').read_bytes()).hexdigest(),
                   'predict_scene_source_sha256': hashlib.sha256(function_source.encode()).hexdigest()}
        (window/'prediction-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        convert_actions(window)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
