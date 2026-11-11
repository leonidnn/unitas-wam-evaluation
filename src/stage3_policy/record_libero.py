"""Run the original UNITAS LIBERO client with simulator-state recording.

Accepts the same command-line arguments as evaluation/libero/client.py.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np

from replay_data import StateRecording, sha256


def set_lighting(environment, scale):
    if scale == 1:
        return []
    from robosuite.utils.mjmod import LightingModder

    modder = LightingModder(environment.sim)
    lights = list(modder.light_names)
    if not lights:
        raise ValueError('Cannot change illumination: no named lights')
    # Использую штатные setters LightingModder; случайные параметры не выбираю.
    for name in lights:
        modder.set_ambient(name, modder.get_ambient(name).copy() * scale)
        modder.set_diffuse(name, modder.get_diffuse(name).copy() * scale)
        modder.set_specular(name, modder.get_specular(name).copy() * scale)
    return lights


class RecordingEnvironment:
    def __init__(self, environment, recording):
        self.environment, self.recording = environment, recording
        self.phase = 1

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def step(self, action):
        observation = self.environment.step(action)
        self.recording.capture(action, self.phase)
        return observation

    def close(self):
        try:
            self.recording.save()
        finally:
            self.environment.close()


def main():
    # Использую исходный main/run_episode из evaluation/libero/client.py без копирования цикла.
    import evaluation.libero.client as client

    original_create = client.create_environment
    original_settle = client.settle_then_enable_absolute_control
    original_run = client.run_episode
    original_act = client.PolicyClient.act
    original_protocol = client.record_protocol
    active = {}
    current = {}
    light_scale = float(os.environ.get('UNITAS_LIGHT_SCALE', '1'))
    if not np.isfinite(light_scale) or not 0 < light_scale <= 1:
        raise ValueError('UNITAS_LIGHT_SCALE must be in (0, 1]')

    def record_protocol(path, protocol):
        protocol['scene_condition'] = {'light_scale': light_scale}
        protocol['recording_wrapper_sha256'] = sha256(__file__)
        return original_protocol(path, protocol)

    def create(suite, task_id, episode_id, seed):
        env, observation, instruction, task_name = original_create(suite, task_id, episode_id, seed)
        destination = Path(current['args'].output_dir) / (
            f'{suite}_task{task_id:02d}_episode{episode_id:03d}_replay'
        )
        try:
            lights = set_lighting(env, light_scale)
            if light_scale != 1:
                observation = env.env._get_observations(force_update=True)
            recording = StateRecording(destination, env.sim.model._model, env.sim.data._data, {
                'suite': suite, 'task_id': task_id, 'episode_id': episode_id, 'seed': seed,
                'instruction': instruction, 'task_name': task_name,
                'upstream_client_sha256': sha256(client.__file__),
                'wrapper_sha256': sha256(__file__), 'policy_protocol': current['args'].protocol,
                'condition': {'light_scale': light_scale, 'modified_lights': lights},
            })
        except BaseException:
            env.close()
            raise
        active['recording'] = recording
        return RecordingEnvironment(env, recording), observation, instruction, task_name

    def settle(env, observation, steps):
        result = original_settle(env, observation, steps)
        env.phase = 2
        return result

    def timed_act(connection, observation, instruction, history):
        start = time.perf_counter()
        status = 'error'
        count = None
        try:
            actions = original_act(connection, observation, instruction, history)
            count, status = len(actions), 'ok'
            return actions
        finally:
            entry = {'elapsed_seconds': time.perf_counter()-start,
                     'measurement': 'client_roundtrip_including_serialization',
                     'status': status, 'action_count': count,
                     'suite': active['recording'].metadata['suite'],
                     'task_id': active['recording'].metadata['task_id'],
                     'episode_id': active['recording'].metadata['episode_id']}
            path = Path(current['args'].output_dir) / 'policy-request-timing.jsonl'
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry)+'\n')

    def run(args, episode_id):
        current['args'] = args
        active.clear()
        try:
            result = original_run(args, episode_id)
        except BaseException as error:
            if 'recording' in active:
                active['recording'].finish(error=error)
            raise
        else:
            active['recording'].finish(result=result)
            return result

    client.create_environment = create
    client.settle_then_enable_absolute_control = settle
    client.run_episode = run
    client.PolicyClient.act = timed_act
    client.record_protocol = record_protocol
    try:
        client.main()
    finally:
        client.create_environment = original_create
        client.settle_then_enable_absolute_control = original_settle
        client.run_episode = original_run
        client.PolicyClient.act = original_act
        client.record_protocol = original_protocol


if __name__ == '__main__':
    main()
