"""Project a frozen SceneFlow prediction onto the held-out RGB frames."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def project(xyz, intrinsic, extrinsic):
    camera = xyz @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    pixel = camera @ intrinsic.T
    uv = pixel[:, :2] / pixel[:, 2, None] - 0.5
    return uv, camera[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        xyz0 = source['xyz0']
        colors = source['colors'].astype(np.float32)
        intrinsic = source['batch.cam0_intrinsic'][0].astype(np.float64)
        extrinsic = source['batch.cam0_extrinsic'][0].astype(np.float64)
    with np.load(args.prediction, allow_pickle=False) as result:
        scene = result['scene']
        gripper = result['gripper']
        fps = float(result['fps'])
    if scene.shape != (21, len(xyz0), 3) or gripper.shape != (21, 20, 3):
        raise ValueError('Unexpected prediction shape')
    if not np.array_equal(scene[0], xyz0):
        raise ValueError('Prediction does not belong to the supplied input')
    # Беру зелёные точки как диагностический срез, не как эталонную сегментацию.
    green = (colors[:, 1] > 1.25 * colors[:, 0]) & (colors[:, 1] > 1.1 * colors[:, 2]) & (colors[:, 1] > 50)
    width, height = 1920, 1080
    intrinsic[0] *= width / 224
    intrinsic[1] *= height / 224
    samples = [(0, 't0.png', 0.0), (10, 't05.png', 0.5), (20, 't1.png', 0.95)]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    metrics = {'prediction_shape': list(scene.shape), 'fps': fps, 'green_query_count': int(green.sum()),
               'green_rule': 'G>1.25R, G>1.1B, G>50; diagnostic only', 'samples': []}
    for row, (step, filename, requested_offset) in enumerate(samples):
        frame = np.asarray(Image.open(args.frames / filename).convert('RGB'))
        if frame.shape[:2] != (height, width):
            raise ValueError(f'Unexpected frame size: {filename}')
        uv, z = project(scene[step, green], intrinsic, extrinsic)
        grip_uv, grip_z = project(gripper[step], intrinsic, extrinsic)
        cup_visible = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        grip_visible = (grip_z > 0) & (grip_uv[:, 0] >= 0) & (grip_uv[:, 0] < width) & (grip_uv[:, 1] >= 0) & (grip_uv[:, 1] < height)
        axes[row, 0].imshow(frame)
        axes[row, 1].imshow(frame)
        axes[row, 1].scatter(uv[cup_visible, 0], uv[cup_visible, 1], s=2, c='#00e8ff', alpha=.65, label='predicted green queries')
        axes[row, 1].scatter(grip_uv[grip_visible, 0], grip_uv[grip_visible, 1], s=14, c='#e600a8', label='predicted gripper')
        for col in range(2):
            axes[row, col].set_xlim(400, 1250)
            axes[row, col].set_ylim(900, 180)
            axes[row, col].set_axis_off()
            time_label = (f'held-out RGB (requested +{requested_offset:.2f} s)' if col == 0
                          else f'prediction +{step/fps:.2f} s on held-out RGB')
            axes[row, col].set_title(time_label)
        green_motion = np.linalg.norm(scene[step, green] - scene[0, green], axis=1)
        metrics['samples'].append({'step': step, 'prediction_time_s': step/fps,
                                   'heldout_requested_offset_s': requested_offset,
                                   'green_visible_projected': int(cup_visible.sum()),
                                   'gripper_visible_projected': int(grip_visible.sum()),
                                   'green_displacement_median_m': float(np.median(green_motion)),
                                   'green_displacement_p95_m': float(np.percentile(green_motion, 95))})
    axes[0, 1].legend(loc='upper right', fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'prediction-vs-heldout.png', dpi=130)
    plt.close(fig)
    displacement = np.linalg.norm(scene[-1] - scene[0], axis=1)
    initial_uv, initial_z = project(scene[0], intrinsic, extrinsic)
    inside = (initial_z > 0) & (initial_uv[:, 0] >= 0) & (initial_uv[:, 0] < width) & (initial_uv[:, 1] >= 0) & (initial_uv[:, 1] < height)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(np.asarray(Image.open(args.frames / 't0.png').convert('RGB')))
    plot = ax.scatter(initial_uv[inside, 0], initial_uv[inside, 1], c=1000 * displacement[inside],
                      s=2, vmin=0, vmax=5, cmap='magma', alpha=.8)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    fig.colorbar(plot, ax=ax, label='Predicted 3D displacement at +1 s (mm); clipped at 5 mm')
    fig.savefig(args.output / 'predicted-displacement.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    metrics['displacement_all_m'] = {name: float(np.quantile(displacement, quantile)) for name, quantile in
                                     [('median', .5), ('p95', .95), ('p99', .99), ('max', 1.0)]}
    metrics['displacement_green_m'] = {name: float(np.quantile(displacement[green], quantile)) for name, quantile in
                                       [('median', .5), ('p95', .95), ('max', 1.0)]}
    (args.output / 'projection-summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
