"""Score corresponding 3D trajectories and a stationary reference predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(errors, mask):
    final = mask[-1]
    return {'ade_m': float(errors[mask].mean()) if mask.any() else None,
            'fde_m': float(errors[-1, final].mean()) if final.any() else None,
            'valid_point_times': int(mask.sum()), 'final_points': int(final.sum()),
            'points_with_any_valid_future': int(mask.any(axis=0).sum())}


def rigidity(trajectory, body_ids):
    """Distance from each body cloud to its best rigid transform of the initial cloud."""
    body_ids = np.asarray(body_ids)
    if body_ids.shape != (trajectory.shape[1],):
        raise ValueError('body_ids must have one entry per point')
    result = {}
    for body_id in np.unique(body_ids):
        selected = body_ids == body_id
        if selected.sum() < 3:
            continue
        initial = trajectory[0, selected]
        source = initial - initial.mean(axis=0)
        residuals = []
        for cloud in trajectory[1:, selected]:
            target = cloud - cloud.mean(axis=0)
            u, _, vt = np.linalg.svd(source.T @ target)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            rotation = u @ correction @ vt
            residuals.append(float(np.sqrt(np.mean(np.sum((source @ rotation-target)**2, axis=-1)))))
        result[str(body_id)] = {'points': int(selected.sum()),
                               'mean_rigid_fit_rms_m': float(np.mean(residuals)),
                               'max_rigid_fit_rms_m': float(np.max(residuals))}
    return result


def score(prediction, reference, visibility, valid=None, motion_threshold_m=0.01):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError('Prediction and reference must share [T+1,N,3] shape')
    if reference.shape[0] < 2 or reference.shape[1] == 0:
        raise ValueError('Need initial state, future and at least one point')
    if not np.isfinite(motion_threshold_m) or motion_threshold_m <= 0:
        raise ValueError('Motion threshold must be positive')
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    finite = np.isfinite(reference).all(axis=-1)
    valid = finite if valid is None else np.asarray(valid, bool) & finite
    visibility = np.asarray(visibility, bool)
    if valid.shape != reference.shape[:2] or visibility.shape != valid.shape:
        raise ValueError('Masks must share [T+1,N] shape')
    if not valid[0].all() or not np.allclose(prediction[0], reference[0], atol=1e-5, rtol=0):
        raise ValueError('Initial corresponding points must match within 10 micrometres')
    # Использую будущий эталон только для определения группы движения и оценки.
    displacement = np.linalg.norm(reference[1:] - reference[:1], axis=-1)
    moving = np.where(valid[1:], displacement, 0).max(axis=0) > motion_threshold_m
    errors = np.linalg.norm(prediction[1:] - reference[1:], axis=-1)
    stationary_errors = displacement
    groups = {'all': np.ones(len(moving), bool), 'moving': moving, 'static': ~moving}
    result = {'motion_threshold_m': motion_threshold_m,
              'motion_rule': 'maximum valid displacement from t0 over the fixed horizon > threshold',
              'ade_weighting': 'equal weight per valid point-time, excluding t0',
              'fde_rule': 'valid corresponding points at final time only', 'groups': {}}
    for visibility_name, mask in (('all_correspondences', valid[1:]),
                                  ('visible', valid[1:] & visibility[1:])):
        result['groups'][visibility_name] = {}
        for name, selected in groups.items():
            group_mask = mask & selected[None]
            result['groups'][visibility_name][name] = {
                'selected_points': int(selected.sum()),
                'unitas': summarize(errors, group_mask),
                'stationary': summarize(stationary_errors, group_mask)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--motion-threshold-m', type=float, default=0.01)
    args = parser.parse_args()
    with np.load(args.prediction, allow_pickle=False) as p, np.load(args.reference, allow_pickle=False) as r:
        if not np.isclose(float(p['fps']), float(r['fps']), atol=1e-6, rtol=0):
            raise ValueError('Prediction and reference time grids differ')
        result = score(p['scene'], r['scene'], r['visibility'],
                       r['valid'] if 'valid' in r else None, args.motion_threshold_m)
        if 'body_ids' in r:
            result['rigidity'] = {'unitas': rigidity(p['scene'], r['body_ids']),
                                  'reference': rigidity(r['scene'], r['body_ids'])}
    result['sources'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.prediction, args.reference)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
