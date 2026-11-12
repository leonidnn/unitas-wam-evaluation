"""Project a saved SceneFlow prediction onto its initial calibrated RGB-D views.

Uses the inverse convention of unitas/scene_flow/io.py::backproject_queries.
Only NumPy and Matplotlib are needed; model inference is not performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points, intrinsic, extrinsic):
    camera = np.asarray(points, dtype=np.float64) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    homogeneous = camera @ intrinsic.T
    uv = np.full(camera.shape[:-1] + (2,), np.nan)
    valid = np.isfinite(camera).all(axis=-1) & (camera[..., 2] > 0)
    # Инвертирую pixel-center convention (u+0.5, v+0.5) из backproject_queries.
    uv[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None] - .5
    return uv, camera[..., 2], valid


def in_frame(uv, positive, height, width):
    return (positive & (uv[..., 0] >= -.5) & (uv[..., 0] < width-.5)
            & (uv[..., 1] >= -.5) & (uv[..., 1] < height-.5))


def reference_masks(uv, z, positive, depth, tolerance):
    height, width = depth.shape
    inside = in_frame(uv, positive, height, width)
    measured = np.full(len(z), np.nan)
    ids = np.flatnonzero(inside)
    pixel = np.floor(uv[ids] + .5).astype(int)
    measured[ids] = depth[pixel[:, 1], pixel[:, 0]]
    valid = inside & np.isfinite(measured) & (measured > 0)
    residual = z - measured
    return {'in_frame': inside, 'depth_available': valid,
            'depth_compatible': valid & (np.abs(residual) <= tolerance),
            'behind_initial_depth': valid & (residual > tolerance),
            'in_front_of_initial_depth': valid & (residual < -tolerance)}, residual


def quantiles_mm(values):
    if not len(values):
        return None
    return dict(zip(['p05', 'p50', 'p95'], (1000*np.quantile(values, [.05, .5, .95])).tolist()))


def roundtrip(intrinsic, extrinsic, depth):
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    pixel = np.stack([u+.5, v+.5, np.ones(len(u))], axis=1)
    camera = (pixel @ np.linalg.inv(intrinsic).T) * depth[v, u, None]
    inverse = np.linalg.inv(extrinsic)
    points = camera @ inverse[:3, :3].T + inverse[:3, 3]
    uv, z, _ = project(points, intrinsic, extrinsic)
    pixel_error = float(np.max(np.abs(uv - np.stack([u, v], axis=1))))
    depth_error = float(np.max(np.abs(z-depth[v, u])))
    if pixel_error > 1e-6 or depth_error > 1e-6:
        raise ValueError('Projection does not invert the stored calibration convention')
    return {'pixels': len(u), 'max_pixel_error': pixel_error, 'max_depth_error_m': depth_error}


def cloud_raster(uv, z, positive, colors, depth):
    height, width = depth.shape
    ids = np.flatnonzero(in_frame(uv, positive, height, width))
    pixel = np.floor(uv[ids]+.5).astype(int)
    flat = pixel[:, 1]*width + pixel[:, 0]
    order = np.lexsort((z[ids], flat))
    _, first = np.unique(flat[order], return_index=True)
    chosen = ids[order[first]]
    coordinates = np.floor(uv[chosen]+.5).astype(int)
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    raster_z = np.full(depth.shape, np.nan)
    rgb[coordinates[:, 1], coordinates[:, 0]] = colors[chosen]
    raster_z[coordinates[:, 1], coordinates[:, 0]] = z[chosen]
    valid = np.isfinite(raster_z) & np.isfinite(depth) & (depth > 0)
    residual = np.where(valid, (raster_z-depth)*1000, np.nan)
    return rgb, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--depth-tolerance-m', type=float, default=.01)
    parser.add_argument('--scene-trajectories', type=int, default=500)
    args = parser.parse_args()
    if not 0 < args.depth_tolerance_m or args.scene_trajectories < 1:
        parser.error('Tolerance and trajectory count must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    with np.load(args.input, allow_pickle=False) as source:
        batch = {key: source[key].copy() for key in source.files}
    with np.load(args.prediction, allow_pickle=False) as prediction:
        result = {key: prediction[key].copy() for key in prediction.files}
    scene, gripper = result['scene'], result['gripper']
    if not np.array_equal(scene[0], batch['xyz0']):
        raise ValueError('Prediction belongs to another initial point cloud')
    if not np.array_equal(gripper[0], batch['batch.history_robot_flows'][0, -1]):
        raise ValueError('Prediction does not start at the stored gripper history')
    if not np.isfinite(scene).all() or not np.isfinite(gripper).all():
        raise ValueError('Non-finite prediction coordinates')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames = [0, (len(scene)-1)//2, len(scene)-1]
    fps = float(result['fps'])
    report = {'input': str(args.input), 'input_sha256': sha256(args.input),
              'prediction': str(args.prediction), 'prediction_sha256': sha256(args.prediction),
              'script_sha256': sha256(__file__), 'command': sys.argv,
              'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
              'frames': frames, 'fps': fps, 'depth_tolerance_m': args.depth_tolerance_m,
              'sampling': 'Evenly spaced indices among initially depth-compatible scene points; no motion selection',
              'future_visibility': 'Unknown; initial scene cohort is fixed, future points clipped only by initial camera frustum',
              'camera_motion': 'All projections use the initial extrinsic, including the wrist camera',
              'ground_truth_future': False, 'cameras': {}}
    mask_arrays = {}
    for index in range(3):
        name = f'cam{index}'
        if not bool(batch[f'batch.{name}_exists'].item()):
            continue
        intrinsic = batch[f'batch.{name}_intrinsic'][0].astype(np.float64)
        extrinsic = batch[f'batch.{name}_extrinsic'][0].astype(np.float64)
        rgb = batch[f'batch.{name}_initial_rgb'][0]
        depth = batch[f'batch.{name}_initial_depth'][0].squeeze()
        height, width = depth.shape
        uv, z, positive = project(scene, intrinsic, extrinsic)
        grip_uv, grip_z, grip_positive = project(gripper, intrinsic, extrinsic)
        masks, residual = reference_masks(uv[0], z[0], positive[0], depth, args.depth_tolerance_m)
        candidates = np.flatnonzero(masks['depth_compatible'])
        if not len(candidates):
            raise ValueError(f'{name}: no depth-compatible initial points')
        ids = candidates[np.linspace(0, len(candidates)-1,
                                     min(args.scene_trajectories, len(candidates)), dtype=int)]
        for key, mask in masks.items():
            mask_arrays[f'{name}.{key}'] = mask
        mask_arrays[f'{name}.selected_scene_indices'] = ids
        per_camera = {'intrinsic': intrinsic.tolist(), 'extrinsic': extrinsic.tolist(),
                      'roundtrip': roundtrip(intrinsic, extrinsic, depth),
                      'initial_counts': {key: int(value.sum()) for key, value in masks.items()},
                      'scene_points_total': len(scene[0]), 'selected_scene_trajectories': len(ids),
                      'initial_abs_depth_residual_mm': quantiles_mm(np.abs(residual[masks['depth_available']])),
                      'initial_signed_depth_residual_mm': quantiles_mm(residual[masks['depth_available']]),
                      'depth_tolerance_sensitivity_counts': {
                          str(tol): int((masks['depth_available'] & (np.abs(residual) <= tol)).sum())
                          for tol in [.002, .005, .01, .02]},
                      'frames': {}}
        for frame in frames:
            inside = in_frame(uv[frame], positive[frame], height, width)
            grip_inside = in_frame(grip_uv[frame], grip_positive[frame], height, width)
            per_camera['frames'][str(frame)] = {
                'all_scene_in_frame': int(inside.sum()),
                'initial_compatible_cohort_in_frame': int((inside & masks['depth_compatible']).sum()),
                'gripper_in_frame': int(grip_inside.sum())}
        report['cameras'][name] = per_camera

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        projected_rgb, raster_residual = cloud_raster(uv[0], z[0], positive[0], result['colors'], depth)
        axes[0].imshow(rgb)
        axes[0].set_title('Initial RGB')
        axes[1].imshow(projected_rgb)
        axes[1].set_title('Initial cloud, nearest point per pixel')
        axes[2].set_facecolor('#dddddd')
        im = axes[2].imshow(np.ma.masked_invalid(raster_residual), cmap='coolwarm', vmin=-20, vmax=20)
        axes[2].set_title('Initial cloud Z - RGB-D depth')
        fig.colorbar(im, ax=axes[2], label='mm (display clipped at +/-20)')
        for ax in axes:
            ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), xlabel='u, pixels', ylabel='v, pixels')
        fig.suptitle(f'{name}: initial geometry check, not a future prediction error')
        fig.savefig(args.output_dir/f'{name}-alignment.png', dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), constrained_layout=True)
        for ax, frame in zip(axes, frames):
            ax.imshow(rgb, alpha=.65)
            for points, valid_z, selected, color in [
                    (uv, positive, ids, '#0072B2'),
                    (grip_uv, grip_positive, np.arange(gripper.shape[1]), '#CC007A')]:
                if frame:
                    starts, ends = points[:frame, selected], points[1:frame+1, selected]
                    valid = valid_z[:frame, selected] & valid_z[1:frame+1, selected]
                    segments = np.stack([starts, ends], axis=-2)[valid]
                    ax.add_collection(LineCollection(segments, colors=color, linewidths=.7, alpha=.7))
                keep = in_frame(points[frame, selected], valid_z[frame, selected], height, width)
                chosen_uv = points[frame, selected][keep]
                ax.scatter(chosen_uv[:, 0], chosen_uv[:, 1], s=5 if color=='#CC007A' else 2,
                           c=color, linewidths=0, alpha=.9)
            ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), xlabel='u, pixels', ylabel='v, pixels',
                   title=f'Frame {frame}: t={frame/fps:.2f} s')
        fig.suptitle(f'{name}: projected prediction on INITIAL RGB; camera held at its initial pose')
        fig.legend(handles=[Line2D([], [], color='#0072B2', label=f'Scene: {len(ids)} uniform initial-compatible indices'),
                            Line2D([], [], color='#CC007A', label='Gripper: all initial point identities')],
                   loc='outside lower center', ncol=2)
        fig.savefig(args.output_dir/f'{name}-trajectories.png', dpi=180)
        plt.close(fig)
    np.savez_compressed(args.output_dir/'initial-masks.npz', **mask_arrays)
    (args.output_dir/'inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'cameras': {
        name: data['initial_counts'] for name, data in report['cameras'].items()}}, indent=2))


if __name__ == '__main__':
    main()
