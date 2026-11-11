"""Export fixed-camera comparisons from saved states without stepping physics."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from replay_data import load_recording, restore_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import mujoco

    model, data, metadata, arrays = load_recording(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policy_start = int(np.flatnonzero(arrays['phases'] == 2)[0]) - 1
    indices = [policy_start, (policy_start + len(arrays['states']) - 1)//2,
               len(arrays['states']) - 1]
    restore_frame(model, data, metadata, arrays, indices[0])
    interest = [i for i in range(model.nbody)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                in ('porcelain_mug_1_main', 'plate_1_main')]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.xpos[interest].mean(axis=0) if interest else model.stat.center
    camera.distance, camera.azimuth, camera.elevation = 1.1, 90, -20
    options = mujoco.MjvOption()
    # Скрываю коллизионную группу так же, как robosuite/environments/base.py.
    options.geomgroup[0], options.geomgroup[1] = 0, 1
    width, height = 640, 480
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    sheet = Image.new('RGB', (width*3, (height+32)*2), 'white')
    draw = ImageDraw.Draw(sheet)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for row, (name, view) in enumerate([('agentview', 'agentview'), ('side', camera)]):
            for column, index in enumerate(indices):
                restore_frame(model, data, metadata, arrays, index)
                renderer.update_scene(data, camera=view, scene_option=options)
                image = Image.fromarray(renderer.render().copy())
                image.save(args.output_dir / f'{name}-frame{index:04d}.png')
                x, y = column*width, row*(height+32)
                sheet.paste(image, (x, y+32))
                draw.text((x+10, y+10),
                          f'{name} | state {index} | t={arrays["times"][index]:.2f} s',
                          fill='black')
    sheet.save(args.output_dir / 'comparison.png')
    print(f'Exported fixed views of states {indices}; no physics steps performed')


if __name__ == '__main__':
    main()
