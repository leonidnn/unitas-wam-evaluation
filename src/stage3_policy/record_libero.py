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
    def __init__(self, environment, recording, push=None, occlusion=None):
        self.environment, self.recording = environment, recording
        self.occlusion = occlusion
        self.phase = 1
        self.push, self.policy_steps = push, 0

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def step(self, action):
        if self.phase == 2 and self.push is not None:
            self.push.apply(self.policy_steps)
        observation = self.environment.step(action)
        self.recording.capture(action, self.phase)
        if self.phase == 2:
            self.policy_steps += 1
        return observation

    def close(self):
        try:
            if self.occlusion is not None:
                self.occlusion.close()
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
    occlusion_enabled = os.environ.get('UNITAS_MUG_OCCLUSION') == '1'
    friction_scale = float(os.environ.get('UNITAS_MUG_FRICTION_SCALE', '1'))
    push_force = float(os.environ.get('UNITAS_PUSH_FORCE_N', '0'))
    push_step = int(os.environ.get('UNITAS_PUSH_START_STEP', '60'))
    push_steps = int(os.environ.get('UNITAS_PUSH_DURATION_STEPS', '3'))
    if not np.isfinite(push_force) or push_force < 0 or push_step < 0 or push_steps < 1:
        raise ValueError('Invalid object-push parameters')
    if not np.isfinite(light_scale) or not 0 < light_scale <= 1:
        raise ValueError('UNITAS_LIGHT_SCALE must be in (0, 1]')

    def record_protocol(path, protocol):
        protocol['scene_condition'] = {'light_scale': light_scale}
        if push_force:
            protocol['scene_condition']['push'] = {'force_x_n': push_force,
                                                  'start_step': push_step, 'steps': push_steps}
        if occlusion_enabled:
            protocol['scene_condition']['mug_occlusion'] = True
        if friction_scale != 1:
            protocol['scene_condition']['mug_friction_scale'] = friction_scale
        protocol['recording_wrapper_sha256'] = sha256(__file__)
        return original_protocol(path, protocol)

    def create(suite, task_id, episode_id, seed):
        env, observation, instruction, task_name = original_create(suite, task_id, episode_id, seed)
        destination = Path(current['args'].output_dir) / (
            f'{suite}_task{task_id:02d}_episode{episode_id:03d}_replay'
        )
        occlusion = None
        try:
            if occlusion_enabled:
                from occlusion import MugOcclusion
                occlusion = MugOcclusion(env)
                observation = env.env._get_observations(force_update=True)
            lights = set_lighting(env, light_scale)
            friction = None
            if friction_scale != 1:
                from interventions import lower_mug_friction
                friction = lower_mug_friction(env, friction_scale)
                observation = env.env._get_observations(force_update=True)
            push = None
            if push_force:
                from interventions import ObjectPush
                push = ObjectPush(env.sim.model._model, env.sim.data._data,
                                  push_force, push_step, push_steps)
            if light_scale != 1:
                observation = env.env._get_observations(force_update=True)
            recording = StateRecording(destination, env.sim.model._model, env.sim.data._data, {
                'suite': suite, 'task_id': task_id, 'episode_id': episode_id, 'seed': seed,
                'instruction': instruction, 'task_name': task_name,
                'upstream_client_sha256': sha256(client.__file__),
                'wrapper_sha256': sha256(__file__), 'policy_protocol': current['args'].protocol,
                'condition': {'light_scale': light_scale, 'modified_lights': lights},
                'visual_occluder': occlusion.spec if occlusion is not None else None,
                'contact_modification': friction,
                'disturbance': push.spec if push is not None else None,
            })
        except BaseException:
            if occlusion is not None:
                occlusion.close()
            env.close()
            raise
        wrapper = RecordingEnvironment(env, recording, push, occlusion)
        active.update(recording=recording, wrapper=wrapper)
        return wrapper, observation, instruction, task_name

    def settle(env, observation, steps):
        observation, displacement = original_settle(env, observation, steps)
        if env.occlusion is not None:
            env.occlusion.calibrate()
            env.recording.metadata['visual_occluder'] = env.occlusion.spec
            env.recording.save_model()
            observation = env.environment.env._get_observations(force_update=True)
        env.phase = 2
        return observation, displacement

    def timed_act(connection, observation, instruction, history):
        wrapper = active['wrapper']
        if wrapper.occlusion is not None and wrapper.policy_steps <= 100:
            entry = {'policy_step': wrapper.policy_steps, 'visibility': wrapper.occlusion.measure()}
            path = wrapper.recording.directory / 'occlusion-observations.jsonl'
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry) + '\n')
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
