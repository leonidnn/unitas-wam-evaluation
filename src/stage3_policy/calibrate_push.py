"""Measure disturbance size before choosing policy-test amplitudes."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from replay_data import load_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forces', type=float, nargs='+', default=[0.0, 0.2, 0.3])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    model, data, metadata, arrays = load_recording(args.recording)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'porcelain_mug_1_main')
    anchor = int(np.flatnonzero(arrays['phases'] == 2)[0] - 1 + 60)
    rows = []
    for force in args.forces:
        mujoco.mj_setState(model, data, arrays['states'][anchor], metadata['state_spec'])
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        initial = data.xpos[body].copy()
        positions = [initial]
        for step in range(round(.35/model.opt.timestep)):
            data.xfrc_applied[body] = 0
            if step*model.opt.timestep < .15:
                data.xfrc_applied[body, 0] = force
            mujoco.mj_step(model, data)
            positions.append(data.xpos[body].copy())
        positions = np.asarray(positions)
        rows.append({'force_x_n': force, 'duration_s': .15,
                     'maximum_horizontal_displacement_m': float(np.linalg.norm(positions[:, :2]-initial[:2], axis=1).max()),
                     'final_displacement_m': (positions[-1]-initial).tolist(),
                     'minimum_body_height_m': float(positions[:, 2].min())})
    result = {'source': str(args.recording), 'anchor_frame': anchor,
              'mug_mass_kg': float(model.body_mass[body]), 'rows': rows,
              'scope': 'Numerical disturbance calibration; saved actuator controls held fixed, no policy inference. Not a robustness outcome.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
