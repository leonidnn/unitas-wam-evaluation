"""Measure occluder coverage at recorded policy requests with LIBERO's RGB groups."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from occlusion import CAMERAS, NAME
from replay_data import load_recording, restore_frame


def measure(recording):
    model, data, metadata, arrays = load_recording(recording)
    geom = model.geom(NAME).id
    targets = np.flatnonzero(model.geom_bodyid == model.body('porcelain_mug_1_main').id)
    rows = [json.loads(line) for line in (recording / 'occlusion-observations.jsonl')
            .read_text(encoding='utf-8').splitlines()]
    policy_start = int(np.flatnonzero(arrays['phases'] == 2)[0] - 1)
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.geomgroup[1] = 1
    output = []
    with mujoco.Renderer(model, height=256, width=256) as renderer:
        renderer.enable_segmentation_rendering()
        for row in rows:
            frame = policy_start + row['policy_step']
            restore_frame(model, data, metadata, arrays, frame)
            visible_position = data.geom_xpos[geom].copy()
            result = {'frame': frame, 'time_s': float(arrays['times'][frame]),
                      'policy_step': row['policy_step'], 'visibility': {}}
            for camera in CAMERAS:
                renderer.update_scene(data, camera=camera, scene_option=options)
                occluder = renderer.render()[:, :, 0] == geom
                data.geom_xpos[geom] = [0, 0, -4]
                renderer.update_scene(data, camera=camera, scene_option=options)
                target = np.isin(renderer.render()[:, :, 0], targets)
                data.geom_xpos[geom] = visible_position
                count = int(target.sum())
                covered = int(np.count_nonzero(target & occluder))
                result['visibility'][camera] = {
                    'target_pixels_without_occluder': count,
                    'occluded_target_pixels': covered,
                    'occluded_fraction': covered / count if count else None,
                }
            output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = measure(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
