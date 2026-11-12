"""File mechanics for direct visual annotations; no learned vision or tracking."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def ffmpeg_path(explicit=None):
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError('The supplied FFmpeg executable does not exist')
        return str(Path(explicit).resolve())
    found = shutil.which('ffmpeg')
    if found:
        return found
    project = Path(__file__).resolve().parents[3]
    local = sorted((project/'docs/local/media-tools/imageio_ffmpeg/binaries').glob('ffmpeg*.exe'))
    if local:
        return str(local[0])
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError('Pass --ffmpeg or install a local FFmpeg executable') from error


def run_ffmpeg(executable, arguments, probe=False):
    result = subprocess.run([executable, '-hide_banner', '-nostdin', '-threads', '2', *arguments],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    if result.returncode and not probe:
        # Не вывожу сырой stderr: он может содержать GPS исходного файла.
        raise RuntimeError(f'FFmpeg operation failed with exit code {result.returncode}')
    return result.stderr


def inspect_video(video, executable):
    log = run_ffmpeg(executable, ['-i', str(video)], probe=True)
    duration = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', log)
    video_line = next((line for line in log.splitlines() if 'Video:' in line), '')
    size = re.search(r'\b(\d{2,5})x(\d{2,5})\b', video_line)
    fps = re.search(r'([\d.]+) fps', video_line)
    if not duration or not size:
        raise ValueError('Could not read video dimensions and duration')
    h, m, s = duration.groups()
    return {'name': Path(video).name, 'sha256': digest(video),
            'duration_s': int(h)*3600+int(m)*60+float(s),
            'width': int(size[1]), 'height': int(size[2]),
            'reported_fps': float(fps[1]) if fps else None}


def timeline(video, cutoff, executable):
    expression = f'select=lte(t\\,{cutoff:.9f}),showinfo'
    log = run_ffmpeg(executable, ['-copyts', '-start_at_zero', '-i', str(video),
                                '-an', '-sn', '-dn', '-vf', expression,
                                '-t', str(cutoff+.05), '-f', 'null', '-'])
    times = [float(value) for value in re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([\d.eE+\-]+)', log)]
    if not times or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('No usable monotonic video timestamps in the requested prefix')
    if max(times) > cutoff+1e-6:
        raise ValueError('Decoder returned a frame beyond the permitted cutoff')
    return times


def extract_images(video, indices, folder, executable):
    folder.mkdir(parents=True, exist_ok=False)
    indices = sorted(set(indices))
    selection = '+'.join(f'eq(n\\,{index})' for index in indices)
    run_ffmpeg(executable, ['-i', str(video), '-an', '-sn', '-dn', '-vf', f'select={selection}',
                           '-fps_mode', 'passthrough', '-frames:v', str(len(indices)),
                           '-map_metadata', '-1', '-map_chapters', '-1', str(folder/'frame-%03d.png')])
    paths = sorted(folder.glob('frame-*.png'))
    if len(paths) != len(indices):
        raise ValueError('Extracted frame count differs from the selected source indices')
    return dict(zip(indices, paths))


def clip(video, start, end, destination, executable, include_start=True):
    lower = 'gte' if include_start else 'gt'
    expression = f'select={lower}(t\\,{start:.9f})*lte(t\\,{end:.9f}),setpts=PTS-STARTPTS'
    run_ffmpeg(executable, ['-copyts', '-start_at_zero', '-i', str(video), '-an', '-sn', '-dn',
                           '-vf', expression, '-t', str(end-start+.05), '-fps_mode', 'vfr',
                           '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                           '-map_metadata', '-1', '-map_chapters', '-1', str(destination)])


def preview(args):
    video, work = args.video.resolve(), args.work_dir.resolve()
    executable = ffmpeg_path(args.ffmpeg)
    metadata = inspect_video(video, executable)
    if not math.isfinite(args.until) or not 0 <= args.until <= metadata['duration_s']-1:
        raise ValueError('Preview time must leave at least one second unseen')
    work.mkdir(parents=True, exist_ok=True)
    ledger_path = work/'exposure.json'
    ledger = read_json(ledger_path) if ledger_path.exists() else {'source_sha256': metadata['sha256'], 'previews': []}
    if ledger['source_sha256'] != metadata['sha256']:
        raise ValueError('Workspace belongs to another source video')
    if args.until < ledger.get('exposed_until_s', -1):
        raise ValueError('Cannot move the preview boundary backwards')
    if (work/'manifest.json').exists():
        raise ValueError('Input is already frozen; do not expose more source frames in this workspace')
    times = timeline(video, args.until, executable)
    folder = work/f'preview-{len(ledger["previews"]):03d}'
    path = extract_images(video, [len(times)-1], folder, executable)[len(times)-1]
    ledger['exposed_until_s'] = args.until
    ledger['previews'].append({'requested_s': args.until, 'actual_s': times[-1],
                               'path': path.relative_to(work).as_posix(), 'sha256': digest(path)})
    write_json(ledger_path, ledger)
    print(json.dumps({'frame': str(path), 'actual_s': times[-1], 'exposed_until_s': args.until}))


def extract(args):
    work, video = args.work_dir.resolve(), args.video.resolve()
    executable = ffmpeg_path(args.ffmpeg)
    metadata = inspect_video(video, executable)
    if not math.isfinite(args.cutoff) or not 1 <= args.cutoff <= metadata['duration_s']-1:
        raise ValueError('Cutoff must allow one second of history and one second of held-out video')
    if not math.isfinite(args.cup_height_m) or args.cup_height_m <= 0:
        raise ValueError('Cup height must be a positive measurement')
    ledger = read_json(work/'exposure.json')
    if ledger['source_sha256'] != metadata['sha256'] or ledger['exposed_until_s'] > args.cutoff+1e-9:
        raise ValueError('Source differs or the model has already been exposed beyond this cutoff')
    if any((work/name).exists() for name in ['manifest.json', 'frames', 'heldout', 'input.mp4']):
        raise FileExistsError('Use a new workspace; an extraction or partial extraction already exists')
    times = timeline(video, args.cutoff, executable)
    targets = [args.cutoff-1+i/20 for i in range(21)]
    indices = [bisect.bisect_right(times, target+1e-9)-1 for target in targets]
    if min(indices) < 0:
        raise ValueError('The requested history starts before the first available frame')
    paths = extract_images(video, indices, work/'frames', executable)
    history = []
    for sample, (target, index) in enumerate(zip(targets, indices)):
        path = paths[index]
        with Image.open(path) as image:
            if image.size != (metadata['width'], metadata['height']):
                raise ValueError('Decoded dimensions differ; inspect rotation metadata before proceeding')
        history.append({'sample': sample, 'target_s': target, 'actual_s': times[index],
                        'source_frame_index': index, 'path': path.relative_to(work).as_posix(),
                        'sha256': digest(path), 'repeated': sample > 0 and index == indices[sample-1]})
    clip(video, args.cutoff-1, args.cutoff, work/'input.mp4', executable)
    (work/'heldout').mkdir()
    clip(video, args.cutoff, args.cutoff+1, work/'heldout/future.mp4', executable, include_start=False)
    manifest = {'version': 'llm-media-v1', 'source': metadata, 'cutoff_s': args.cutoff,
                'max_preview_s': ledger['exposed_until_s'], 'cup_height_m': args.cup_height_m,
                'height_source': 'user_measured', 'history': history, 'input_video': 'input.mp4',
                'heldout': {'path': 'heldout/future.mp4', 'start_exclusive_s': args.cutoff,
                            'end_s': args.cutoff+1, 'do_not_open_during_preparation': True},
                'source_audio_used': False, 'source_metadata_copied': False}
    write_json(work/'manifest.json', manifest)
    print(json.dumps({'manifest': str(work/'manifest.json'), 'history_samples': 21,
                      'unique_frames': len(paths), 'cutoff_s': args.cutoff}))


def inside(work, relative):
    path = (work/relative).resolve()
    if not path.is_relative_to(work.resolve()):
        raise ValueError('Frame path escapes the workspace')
    return path


def finite_vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        return False
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (ValueError, TypeError):
        return False


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def uv_valid(value):
    return finite_vector(value, 2) and all(0 <= v <= 1 for v in value)


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)/np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion(matrix):
    # Перевожу уже заданную ориентацию в xyzw, не оцениваю её по изображению.
    trace = np.trace(matrix)
    if trace > 0:
        s = math.sqrt(trace+1)*2
        q = [(matrix[2, 1]-matrix[1, 2])/s, (matrix[0, 2]-matrix[2, 0])/s,
             (matrix[1, 0]-matrix[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(max(0, 1+matrix[i, i]-matrix[j, j]-matrix[k, k]))*2
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = s/4, (matrix[j, i]+matrix[i, j])/s, (matrix[k, i]+matrix[i, k])/s, (matrix[k, j]-matrix[j, k])/s
    q = np.asarray(q)
    return q/np.linalg.norm(q)


def model_to_camera(annotation):
    up = np.asarray(annotation['table']['up_normal_camera'], dtype=float)
    up /= np.linalg.norm(up)
    right = np.array([1., 0., 0.])
    x = right-up*np.dot(right, up)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('Table normal is parallel to camera-right; frame is degenerate')
    x /= np.linalg.norm(x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, np.cross(up, x), up])
    transform[:3, 3] = annotation['table']['origin_camera_m']
    return transform


def load_and_validate(work):
    manifest = read_json(work/'manifest.json')
    annotation = read_json(work/'annotations.json')
    errors, warnings = [], []
    try:
        if manifest.get('version') != 'llm-media-v1':
            errors.append('Unsupported manifest version')
        if annotation.get('version') != 'llm-scene-v1':
            errors.append('Unsupported annotation version')
        if annotation.get('manifest_sha256') != digest(work/'manifest.json'):
            errors.append('Annotations do not reference the exact frozen manifest')
        if annotation.get('analysis_method') != 'direct_visual_reasoning' or annotation.get('external_vision_models') != []:
            errors.append('Direct-model-only provenance is missing or external vision models were declared')
        if not str(annotation.get('instruction', '')).strip():
            errors.append('Instruction is missing')
        scale = annotation.get('scale', {})
        if scale.get('source') != 'user_measured' or scale.get('height_m') != manifest.get('cup_height_m'):
            errors.append('Measured cup-height provenance differs from the manifest')
        if annotation.get('camera_stationary') != 'yes':
            errors.append('Stationary camera has not been established by the annotator')
        history = manifest.get('history', [])
        states = annotation.get('history', [])
        if len(history) != 21 or len(states) != 21:
            errors.append('Exactly 21 history samples are required')
        last_actual = -math.inf
        for i, entry in enumerate(history):
            target = manifest['cutoff_s']-1+i/20
            if entry['sample'] != i or abs(entry['target_s']-target) > 1e-7:
                errors.append(f'Sample {i}: target grid differs from 20 Hz history')
            if entry['actual_s'] > entry['target_s']+1e-7 or entry['actual_s'] < last_actual:
                errors.append(f'Sample {i}: future or non-monotonic timestamp')
            if entry['actual_s'] > manifest['cutoff_s']+1e-7:
                errors.append(f'Sample {i}: frame beyond cutoff')
            last_actual = entry['actual_s']
            path = inside(work, entry['path'])
            if not path.is_relative_to((work/'frames').resolve()) or path.suffix.lower() != '.png':
                errors.append(f'Sample {i}: not a permitted history PNG')
            elif digest(path) != entry['sha256']:
                errors.append(f'Sample {i}: frame checksum mismatch')
            repeated = i > 0 and entry['source_frame_index'] == history[i-1]['source_frame_index']
            if entry['repeated'] != repeated:
                errors.append(f'Sample {i}: repeat flag differs from source identity')
        if manifest['max_preview_s'] > manifest['cutoff_s']+1e-7:
            errors.append('Preview exposure extends beyond the input cutoff')
        camera = annotation.get('camera', {})
        intrinsic = np.asarray(camera.get('intrinsic'), dtype=float)
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or abs(intrinsic[0, 1])+abs(intrinsic[1, 0]) > 1e-8):
            errors.append('Estimated camera intrinsic is missing or invalid')
        if camera.get('status') != 'estimated' or camera.get('distortion_assumption') != 'zero':
            errors.append('Camera estimation/distortion assumptions are not declared')
        table = annotation.get('table', {})
        if (table.get('status') != 'estimated' or not finite_vector(table.get('origin_camera_m'), 3)
                or not finite_vector(table.get('up_normal_camera'), 3)
                or np.linalg.norm(table['up_normal_camera']) < 1e-8):
            errors.append('Estimated table origin/up normal is missing or invalid')
        else:
            try:
                model_to_camera(annotation)
            except ValueError as error:
                errors.append(str(error))
        depth = annotation.get('depth', {})
        if (depth.get('width'), depth.get('height'), depth.get('kind'), depth.get('units'), depth.get('status')) != (64, 36, 'camera_z', 'm', 'estimated'):
            errors.append('Depth must be a declared estimated 36x64 camera-z grid in metres')
        grid = depth.get('values_m')
        if not isinstance(grid, list) or len(grid) != 36 or any(not isinstance(row, list) or len(row) != 64 for row in grid):
            errors.append('Depth grid shape must be 36 rows of 64 cells')
        else:
            values = [value for row in grid for value in row if value is not None]
            if not values or not all(number(value) and value > 0 for value in values):
                errors.append('Depth contains invalid values or no known positive cells')
        objects = annotation.get('objects', [])
        ids = [obj.get('id') for obj in objects]
        if not objects or len(ids) != len(set(ids)) or scale.get('object_id') not in ids:
            errors.append('Unique current objects and the measured cup polygon are required')
        for obj in objects:
            polygon = obj.get('polygon_uv', [])
            if len(polygon) < 3 or not all(uv_valid(point) for point in polygon):
                errors.append(f'Object {obj.get("id")}: polygon is missing or invalid')
            if obj.get('status') not in ['visible', 'inferred']:
                errors.append(f'Object {obj.get("id")}: visibility status is missing')
        for i, state in enumerate(states):
            points = state.get('landmarks', [])
            if state.get('sample') != i or len(points) != 21 or [p.get('id') for p in points] != list(range(21)):
                errors.append(f'History {i}: anatomical identities/order must be exactly 0..20')
            for point in points:
                status, uv, z = point.get('status'), point.get('uv'), point.get('z_m')
                if status == 'unknown':
                    if uv is not None or z is not None:
                        errors.append(f'History {i}, point {point.get("id")}: unknown values must be null')
                elif status not in ['visible', 'inferred'] or not uv_valid(uv) or not number(z) or z <= 0:
                    errors.append(f'History {i}, point {point.get("id")}: invalid estimated landmark')
            eef = state.get('eef', {})
            if eef.get('status') == 'estimated':
                q = eef.get('quaternion_camera_xyzw')
                if (not finite_vector(eef.get('position_camera_m'), 3) or not finite_vector(q, 4)
                        or np.linalg.norm(q) < 1e-8 or not number(eef.get('open_fraction'))
                        or not 0 <= eef['open_fraction'] <= 1):
                    errors.append(f'History {i}: invalid conditional EEF estimate')
            elif eef.get('status') != 'unknown' or any(eef.get(k) is not None for k in ['position_camera_m', 'quaternion_camera_xyzw', 'open_fraction']):
                errors.append(f'History {i}: unknown EEF must use null values')
            if i < len(history) and history[i]['repeated'] and i > 0:
                for key in ['landmarks', 'eef']:
                    if state.get(key) != states[i-1].get(key):
                        errors.append(f'History {i}: repeated image has different {key}; no invented interpolation')
        if len(states) == 21:
            if states[-1].get('eef', {}).get('status') != 'estimated':
                errors.append('Current conditional EEF is unknown')
            if not any(p.get('status') in ['visible', 'inferred'] for p in states[-1].get('landmarks', [])[1:]):
                errors.append('Current finger points are all unknown')
        for key in ['assumptions', 'limitations']:
            if not isinstance(annotation.get(key), list) or not annotation[key] or not all(isinstance(v, str) and v.strip() for v in annotation[key]):
                errors.append(f'Explicit nonempty {key} are required')
        if len({item['source_frame_index'] for item in history}) < 21:
            warnings.append('History contains repeated source frames; no extra observations were created')
    except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        errors.append(f'Incomplete or invalid annotation structure: {type(error).__name__}: {error}')
    warnings.extend(['Depth and calibration are model estimates, not measured RGB-D',
                     'Human joints differ from the trained Panda surface points',
                     'Conditional EEF/opening is an interface approximation',
                     'No independent future visibility, 3D ADE/FDE or physical calibration accuracy'])
    validation = {'status': 'ready' if not errors else 'not_ready', 'errors': errors, 'warnings': warnings,
                  'annotation_sha256': digest(work/'annotations.json'),
                  'manifest_sha256': digest(work/'manifest.json'),
                  'future_semantics_inspected_by_helper': False,
                  'readiness_means': 'container and internal consistency only, not validated physical truth'}
    write_json(work/'validation.json', validation)
    return manifest, annotation, validation


def uvz_to_camera(uv, z, intrinsic, width, height):
    return np.linalg.solve(intrinsic, np.array([uv[0]*width, uv[1]*height, 1.]))*z


def geometry(manifest, annotation):
    width, height = manifest['source']['width'], manifest['source']['height']
    intrinsic = np.asarray(annotation['camera']['intrinsic'], dtype=float)
    transform = model_to_camera(annotation)
    inverse = np.linalg.inv(transform)
    points = np.zeros((21, 20, 3), dtype=np.float32)
    exists = np.zeros((21, 20), dtype=bool)
    observed = np.zeros((21, 21), dtype=bool)
    inferred = np.zeros_like(observed)
    poses = np.tile(np.array([0., 0., 0., 0., 0., 0., 1.]), (21, 2, 1)).astype(np.float32)
    opening = np.zeros((21, 2, 1), dtype=np.float32)
    pose_exists = np.zeros((21, 2), dtype=bool)
    reprojection_errors = []
    for i, state in enumerate(annotation['history']):
        for p in state['landmarks']:
            observed[i, p['id']] = p['status'] == 'visible'
            inferred[i, p['id']] = p['status'] == 'inferred'
            if p['status'] == 'unknown':
                continue
            camera = uvz_to_camera(p['uv'], p['z_m'], intrinsic, width, height)
            model = inverse[:3, :3]@camera+inverse[:3, 3]
            restored = transform[:3, :3]@model+transform[:3, 3]
            pixel = intrinsic@restored
            reprojection_errors.append(np.max(np.abs(pixel[:2]/pixel[2]-np.asarray(p['uv'])*[width, height])))
            if p['id']:
                points[i, p['id']-1] = model
                exists[i, p['id']-1] = True
        eef = state['eef']
        if eef['status'] == 'estimated':
            poses[i, 1, :3] = inverse[:3, :3]@eef['position_camera_m']+inverse[:3, 3]
            poses[i, 1, 3:] = quaternion(inverse[:3, :3]@rotation(eef['quaternion_camera_xyzw']))
            opening[i, 1, 0] = eef['open_fraction']
            pose_exists[i, 1] = True
    valid = pose_exists[:, 1] & exists.any(axis=1)
    exists &= valid[:, None]
    pose_exists[:, 1] &= valid
    return transform, points, exists, poses, opening, pose_exists, valid, observed, inferred, max(reprojection_errors, default=0.)


def pack(work):
    manifest, annotation, validation = load_and_validate(work)
    if validation['errors']:
        print(json.dumps(validation, ensure_ascii=False))
        return False
    if (work/'input.npz').exists() or (work/'input-provenance.json').exists():
        raise FileExistsError('Packed output already exists; use a new attempt instead of replacing it')
    transform, points, exists, poses, opening, pose_exists, valid, observed, inferred, error = geometry(manifest, annotation)
    if error > 1e-5:
        raise ValueError('Coordinate conversion does not reproduce the supplied landmark pixels')
    grid = np.asarray(annotation['depth']['values_m'], dtype=np.float32)
    unknown = ~np.isfinite(grid)
    grid[unknown] = 0
    depth = np.asarray(Image.fromarray(grid).resize((224, 224), Image.Resampling.NEAREST)).copy()
    with Image.open(inside(work, manifest['history'][-1]['path'])) as image:
        rgb = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).copy()
    intrinsic = np.asarray(annotation['camera']['intrinsic'], dtype=float).copy()
    intrinsic[0] *= 224/manifest['source']['width']
    intrinsic[1] *= 224/manifest['source']['height']
    v, u = np.nonzero(depth > 0)
    camera = (np.column_stack([u+.5, v+.5, np.ones(len(u))])@np.linalg.inv(intrinsic).T)*depth[v, u, None]
    inverse = np.linalg.inv(transform)
    xyz = (camera@inverse[:3, :3].T+inverse[:3, 3]).astype(np.float32)
    # Беру имена/формы массивов и pixel-center convention из scene_flow/io.py и inference/preprocessing.py.
    arrays = {'xyz0': xyz, 'colors': rgb[v, u], 'domain': np.asarray('libero'),
              'instruction': np.asarray(annotation['instruction']),
              'batch.history_robot_flows': points[None], 'batch.history_robot_exists': exists[None],
              'batch.history_eef_poses': poses[None], 'batch.history_eef_open': opening[None],
              'batch.history_eef_exists': pose_exists[None], 'batch.history_valid_mask': valid[None],
              'batch.history_frame_indices': np.arange(-20, 1, dtype=np.int64)[None]}
    for name in ['gripper', 'eef', 'scene']:
        arrays[f'batch.{name}_flow_fps'] = np.array([20.], np.float32)
        arrays[f'batch.{name}_flow_horizon'] = np.array([20], np.int64)
    for i in range(3):
        arrays[f'batch.cam{i}_exists'] = np.array([i == 0])
        arrays[f'batch.cam{i}_initial_rgb'] = (rgb if i == 0 else np.zeros_like(rgb))[None]
        arrays[f'batch.cam{i}_initial_depth'] = (depth if i == 0 else np.zeros_like(depth))[None]
        arrays[f'batch.cam{i}_intrinsic'] = (intrinsic if i == 0 else np.eye(3)).astype(np.float32)[None]
        arrays[f'batch.cam{i}_extrinsic'] = (transform if i == 0 else np.eye(4)).astype(np.float32)[None]
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in 'fc'):
        raise ValueError('Cannot pack non-finite numeric arrays')
    np.savez_compressed(work/'input.npz', **arrays)
    np.savez_compressed(work/'estimate-masks.npz', depth_unknown_coarse=unknown,
                        depth_known_224=depth > 0, hand_visible=observed, hand_inferred=inferred)
    provenance = {'annotation_sha256': validation['annotation_sha256'], 'manifest_sha256': validation['manifest_sha256'],
                  'input_sha256': digest(work/'input.npz'), 'helper_sha256': digest(__file__),
                  'depth_is_estimated': True, 'domain_is_container_compatibility_only': True,
                  'human_landmark_ids': list(range(1, 21)), 'height_m': manifest['cup_height_m'],
                  'model_to_camera': transform.tolist(), 'camera_to_model': inverse.tolist(),
                  'max_landmark_reprojection_error_px': float(error),
                  'history_timestamps': [{key: entry[key] for key in ['sample', 'target_s', 'actual_s', 'repeated']}
                                         for entry in manifest['history']],
                  'depth_known_fraction': float((depth > 0).mean()), 'query_points': len(xyz),
                  'warnings': validation['warnings'], 'inference_executed': False}
    write_json(work/'input-provenance.json', provenance)
    print(json.dumps({'status': 'packed_experimental_input', 'path': str(work/'input.npz'), 'points': len(xyz)}))
    return True


HAND_EDGES = [(0, base) for base in [1, 5, 9, 13, 17]] + [(base+i, base+i+1) for base in [1, 5, 9, 13, 17] for i in range(3)]


def clean_text(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def render(work):
    manifest, annotation, validation = load_and_validate(work)
    for entry in manifest['history']:
        path = inside(work, entry['path'])
        if (not path.is_relative_to((work/'frames').resolve()) or path.suffix.lower() != '.png'
                or entry['actual_s'] > min(entry['target_s'], manifest['cutoff_s'])+1e-7
                or digest(path) != entry['sha256']):
            raise ValueError('Unsafe or modified history frame; rendering refused')
    figures = work/'figures'
    figures.mkdir(exist_ok=True)
    history = manifest['history']
    width, height = manifest['source']['width'], manifest['source']['height']
    sheet = Image.new('RGB', (7*320, 3*204), '#f3f5f7')
    sheet_draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(history):
        with Image.open(inside(work, entry['path'])) as image:
            thumbnail = image.convert('RGB')
            thumbnail.thumbnail((320, 180))
        x, y = (i%7)*320, (i//7)*204
        sheet.paste(thumbnail, (x, y))
        sheet_draw.text((x+5, y+183), f'{i:02d}: {entry["actual_s"]:.3f}s'+(' repeat' if entry['repeated'] else ''), fill='black')
    sheet.save(figures/'history.png')
    with Image.open(inside(work, history[-1]['path'])) as image:
        image.convert('RGB').save(figures/'current.png')
        overlay = image.convert('RGB')
    draw = ImageDraw.Draw(overlay)
    colors = ['#00ffff', '#ffd700', '#ff55bb', '#66ee66', '#ff9944']
    objects = annotation.get('objects', [])
    semantic = np.zeros((height, width), np.uint16)
    semantic_image = Image.fromarray(semantic)
    semantic_draw = ImageDraw.Draw(semantic_image)
    for i, obj in enumerate(objects):
        polygon = obj.get('polygon_uv', [])
        if len(polygon) < 3 or not all(uv_valid(p) for p in polygon):
            continue
        pixel = [(p[0]*width-.5, p[1]*height-.5) for p in polygon]
        draw.line(pixel+[pixel[0]], fill=colors[i%len(colors)], width=max(2, width//500))
        label = str(obj.get('id', i)).encode('ascii', 'replace').decode('ascii')
        draw.text(pixel[0], label, fill=colors[i%len(colors)], stroke_width=1, stroke_fill='black')
        semantic_draw.polygon(pixel, fill=i+1)
    semantic_image.save(figures/'object-polygons.png')
    states = annotation.get('history', [])
    if len(states) == 21:
        points = {p.get('id'): p for p in states[-1].get('landmarks', [])}
        for a, b in HAND_EDGES:
            if a in points and b in points and all(uv_valid(points[k].get('uv')) for k in [a, b]):
                draw.line([(points[k]['uv'][0]*width-.5, points[k]['uv'][1]*height-.5) for k in [a, b]],
                          fill='#ff55bb', width=max(2, width//500))
        for p in points.values():
            if not uv_valid(p.get('uv')):
                continue
            x, y = p['uv'][0]*width-.5, p['uv'][1]*height-.5
            color = '#ffee00' if p.get('status') == 'visible' else '#ff55bb'
            r = max(3, width//350)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
            draw.text((x+r, y), str(p['id']), fill=color, stroke_width=1, stroke_fill='black')
    overlay.save(figures/'annotations.png')
    write_json(figures/'object-labels.json', {str(i+1): obj.get('id') for i, obj in enumerate(objects)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depth_ready = False
    try:
        grid = np.asarray(annotation.get('depth', {}).get('values_m'), dtype=float)
        if grid.shape == (36, 64):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            axes[0].set_facecolor('#cccccc')
            im = axes[0].imshow(np.ma.masked_invalid(grid), cmap='viridis', interpolation='nearest')
            axes[0].set_title('LLM-estimated camera-z, NOT measured depth')
            fig.colorbar(im, ax=axes[0], label='metres')
            axes[1].imshow(~np.isfinite(grid), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[1].set_title('Unknown cells: white')
            for ax in axes:
                ax.set(xlabel='grid column', ylabel='grid row')
            fig.savefig(figures/'depth.png', dpi=150)
            plt.close(fig)
            depth_ready = True
    except (ValueError, TypeError):
        pass
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.imshow(Image.open(figures/'current.png'), alpha=.55)
    for landmark in range(21):
        locations = []
        for i, state in enumerate(states):
            points = state.get('landmarks', [])
            point = next((p for p in points if p.get('id') == landmark), {})
            locations.append([point['uv'][0]*width-.5, point['uv'][1]*height-.5]
                             if uv_valid(point.get('uv')) else [np.nan, np.nan])
        if locations:
            locations = np.asarray(locations)
            ax.plot(locations[:, 0], locations[:, 1], linewidth=1, alpha=.8)
    ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5), title='Annotated past hand paths on current RGB; not a forecast')
    fig.savefig(figures/'hand-history.png', dpi=150)
    plt.close(fig)
    provenance = read_json(work/'input-provenance.json') if (work/'input-provenance.json').exists() else None
    packed_matches = bool(provenance and (work/'input.npz').exists()
                          and provenance.get('annotation_sha256') == validation['annotation_sha256']
                          and provenance.get('manifest_sha256') == validation['manifest_sha256']
                          and provenance.get('input_sha256') == digest(work/'input.npz'))
    status = 'NOT READY' if validation['errors'] else ('PACKED — EXPERIMENTAL' if packed_matches else 'READY TO PACK — EXPERIMENTAL')
    if provenance and not packed_matches:
        status = 'NOT READY — PACKED FILE IS STALE'
    # Отчёт строю из реально записанного JSON, а не из декоративно сгенерированных картинок.
    lines = ['# Подготовка собственной сцены прямым анализом модели', '', f'**Статус: {status}**', '',
             '> Глубина, геометрия и точки руки оценены моделью. Это не измеренный RGB-D и не эталон.', '',
             '## Разрешённые исходные данные', '',
             f'Видео: `{clean_text(manifest["source"]["name"])}`, {width}×{height}; длительность {manifest["source"]["duration_s"]:.3f} с.',
             f'Граница прогноза: **{manifest["cutoff_s"]:.3f} с**; вход — предшествующая секунда. История: 21 целевое состояние при 20 Гц.',
             f'Уникальных исходных кадров: {len({e["source_frame_index"] for e in history})}. Реальные времена и повторы сохранены в manifest.', '',
             '[Входной фрагмент без аудио](input.mp4)', '',
             '<video controls src="input.mp4"></video>', '',
             '![Все разрешённые кадры истории](figures/history.png)', '',
             'Следующая секунда сохранена отдельно и не использовалась для разметки. Содержание отложенного фрагмента здесь не показывается.', '',
             '## Текущий кадр и разметка', '',
             '![Исходный текущий кадр](figures/current.png)', '',
             '![Полигоны объектов и анатомические точки руки](figures/annotations.png)', '',
             'Жёлтые точки — видимые по оценке разметчика, розовые — предположенные. Даже у видимых точек глубина оценена.', '',
             '| Объект | Статус | Перекрытия |', '|---|---|---|']
    lines += [f'| {clean_text(obj.get("id"))}: {clean_text(obj.get("label", ""))} | {clean_text(obj.get("status"))} | {clean_text(obj.get("occlusion_notes", ""))} |' for obj in objects]
    lines += ['', 'Маска [object-polygons.png](figures/object-polygons.png) растеризует заданные моделью полигоны; при перекрытии последний полигон записывается поверх предыдущего. Это не независимая пиксельная сегментация.', '',
              '## Оценённая глубина', '']
    if depth_ready:
        lines += ['![Карта camera-z и неизвестные области](figures/depth.png)', '']
    else:
        lines += ['Карта глубины отсутствует или имеет некорректную форму.', '']
    lines += ['Сетка 64×36; увеличение до 224×224 при упаковке не добавляет измерений или деталей.', '',
              '## История руки', '', '![Прошлые пути размеченных точек](figures/hand-history.png)', '',
              'Во вход передаются 20 точек пальцев; запястье сохранено отдельно. Человеческая рука отличается от поверхности обучающего захвата Panda. Условная EEF-поза и одно раскрытие не описывают все суставы пальцев.', '',
              '## Что измерено и что предположено', '', '| Величина | Источник |', '|---|---|',
              f'| Высота кружки: {manifest["cup_height_m"]*100:g} см | Измерение пользователя |',
              '| RGB и временные метки | Исходное видео |',
              '| Глубина, K, стол, точки руки, EEF | Оценки модели |',
              '| Дисторсия отсутствует, камера неподвижна | Допущение / визуальная оценка; не инструментальная калибровка |', '',
              f'Команда: `{clean_text(annotation.get("instruction", ""))}`.', '', '### Допущения', '']
    lines += ['- '+clean_text(value) for value in annotation.get('assumptions', [])]
    lines += ['', '### Ограничения', '']
    lines += ['- '+clean_text(value) for value in annotation.get('limitations', [])]
    lines += ['', '## Проверки и готовность', '']
    lines += ['- '+clean_text(value) for value in validation['errors']+validation['warnings']]
    if packed_matches:
        lines += ['', f'Сохранён [input.npz](input.npz), {provenance["query_points"]} query-точек; покрытие глубины {provenance["depth_known_fraction"]:.1%}.',
                  f'Максимальный остаток обратной проекции точек: {provenance["max_landmark_reprojection_error_px"]:.3g} пикселя. Это проверка алгебры, не точности оценок.',
                  '[Преобразования, времена и происхождение](input-provenance.json).']
    lines += ['', '[Основные оценки](annotations.json) · [Манифест](manifest.json) · [Проверки](validation.json)', '',
              'UNITAS не запускалась; точность будущего, ADE/FDE и успешность задачи здесь не оценивались.', '']
    (work/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(work/'report.md'), 'status': status}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['inspect', 'preview', 'extract']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--video', type=Path, required=True)
        cmd.add_argument('--ffmpeg')
        if name != 'inspect':
            cmd.add_argument('--work-dir', type=Path, required=True)
        if name == 'preview':
            cmd.add_argument('--until', type=float, required=True)
        if name == 'extract':
            cmd.add_argument('--cutoff', type=float, required=True)
            cmd.add_argument('--cup-height-m', type=float, default=.11)
    for name in ['validate', 'render', 'pack']:
        commands.add_parser(name).add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps(inspect_video(args.video, ffmpeg_path(args.ffmpeg)), ensure_ascii=False, indent=2))
    elif args.command == 'preview':
        preview(args)
    elif args.command == 'extract':
        extract(args)
    elif args.command == 'validate':
        result = load_and_validate(args.work_dir.resolve())[2]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(bool(result['errors']))
    elif args.command == 'render':
        render(args.work_dir.resolve())
    elif args.command == 'pack':
        return int(not pack(args.work_dir.resolve()))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
