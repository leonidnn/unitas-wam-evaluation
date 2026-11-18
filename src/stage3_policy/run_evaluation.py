"""Run one fixed LIBERO evaluation condition with the original server/client."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from replay_data import sha256


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/content/UNITAS'))
    parser.add_argument('--model-python', default='/content/miniforge/envs/unitas/bin/python')
    parser.add_argument('--sim-python', default='/content/miniforge/envs/libero/bin/python')
    parser.add_argument('--condition', choices=['baseline', 'dim_light', 'stress'], required=True)
    parser.add_argument('--light-scale', type=float, default=None)
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--mug-occlusion', action='store_true')
    parser.add_argument('--mug-friction-scale', type=float, default=1)
    parser.add_argument('--push-force-n', type=float, default=0)
    parser.add_argument('--push-start-step', type=int, default=60)
    parser.add_argument('--push-duration-steps', type=int, default=3)
    parser.add_argument('--checkpoint', default='checkpoints/unitas_libero_finetuned.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    light_scale = args.light_scale
    if light_scale is None:
        if args.condition == 'stress':
            parser.error('--condition stress requires --light-scale')
        light_scale = 1.0 if args.condition == 'baseline' else 0.5
    if not math.isfinite(light_scale) or not 0 < light_scale <= 1:
        parser.error('--light-scale must be in (0, 1]')
    if (not math.isfinite(args.push_force_n) or args.push_force_n < 0
            or args.push_start_step < 0 or args.push_duration_steps < 1):
        parser.error('Invalid push parameters')
    if args.push_force_n and args.condition != 'stress':
        parser.error('Use --condition stress for an external push')
    if args.condition != 'stress' and light_scale != (1.0 if args.condition == 'baseline' else 0.5):
        parser.error('Use --condition stress for a custom light scale')
    if not 0 < args.mug_friction_scale <= 1:
        parser.error('--mug-friction-scale must be in (0, 1]')
    if args.mug_friction_scale != 1 and (
            args.condition != 'stress' or args.push_force_n or light_scale != 1):
        parser.error('Reduced friction requires a separate stress series at normal light')
    if args.mug_occlusion and (
            args.condition != 'stress' or args.push_force_n
            or args.mug_friction_scale != 1 or light_scale != 1):
        parser.error('Occlusion requires a separate stress series at normal light')
    root, output = args.root.resolve(), args.output_dir.resolve()
    if args.episodes <= 0 or args.episode_start < 0:
        parser.error('Invalid episode range')
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(('127.0.0.1', 8000)) == 0:
            raise RuntimeError('Port 8000 is occupied; will not reuse or stop an unknown server')
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    env = {**os.environ, 'PYTHONPATH': '/content/LIBERO',
           'LIBERO_CONFIG_PATH': '/content/libero-config', 'MUJOCO_GL': 'egl',
           'PYOPENGL_PLATFORM': 'egl', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'UNITAS_LIGHT_SCALE': str(light_scale),
           'UNITAS_MUG_OCCLUSION': str(int(args.mug_occlusion)),
           'UNITAS_MUG_FRICTION_SCALE': str(args.mug_friction_scale),
           'UNITAS_PUSH_FORCE_N': str(args.push_force_n),
           'UNITAS_PUSH_START_STEP': str(args.push_start_step),
           'UNITAS_PUSH_DURATION_STEPS': str(args.push_duration_steps),
           'UNITAS_TIMING_PATH': str(output / 'server-policy-timing.jsonl')}
    task_code = ('from libero.libero import benchmark; '
                 's=benchmark.get_benchmark_dict()["libero_90"](); '
                 'print(next(i for i in range(s.n_tasks) '
                 'if s.get_task(i).language=="put the white mug on the plate"))')
    if args.task_id is None:
        task_output = subprocess.check_output([args.sim_python, '-c', task_code], env=env, text=True)
        # Беру последнюю строку: LIBERO печатает предупреждение перед номером задачи.
        task_id = int(task_output.strip().splitlines()[-1])
    else:
        task_id = args.task_id
    server_command = [args.model_python, '-u', str(scripts / 'serve_timed.py'), '--domain', 'libero',
                      '--checkpoint', args.checkpoint,
                      '--florence-dir', 'checkpoints/Florence-2-large', '--device', 'cuda',
                      '--steps', '10', '--execute-steps', '10', '--seed', '160484', '--port', '8000']
    client_command = [args.sim_python, '-u', str(scripts / 'record_libero.py'), '--suite', 'libero_90',
                      '--task-id', str(task_id), '--episode-start', str(args.episode_start),
                      '--num-episodes', str(args.episodes), '--seed', '42', '--settle-steps', '10',
                      '--max-steps', '800', '--no-save-videos', '--output-dir', str(output)]
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'condition': args.condition, 'light_scale': float(env['UNITAS_LIGHT_SCALE']),
                'task_id': task_id, 'mug_occlusion': args.mug_occlusion,
                'mug_friction_scale': args.mug_friction_scale,
                'server_command': server_command, 'client_command': client_command,
                'script_sha256': {p.name: sha256(p) for p in scripts.glob('*.py')},
                'upstream_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}
    manifest = output / 'run.json'
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    server = client = None

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (output / 'server.log').open('w') as server_log, (output / 'client.log').open('w') as client_log:
            server = subprocess.Popen(server_command, cwd=root, env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited: {server.returncode}; see server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('domain') != 'libero':
                        raise ValueError('Unexpected server domain')
                    (output / 'health.json').write_text(json.dumps(health, indent=2))
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Policy server readiness timeout')
                    time.sleep(5)
            print('POLICY_READY', flush=True)
            client = subprocess.Popen(client_command, cwd=root, env=env, stdout=client_log,
                                      stderr=subprocess.STDOUT)
            code = client.wait()
            metadata['client_exit_code'] = code
            if code:
                raise RuntimeError(f'LIBERO client exited: {code}; see client.log')
            metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop(client)
        stop(server)
        metadata['ended_utc'] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
