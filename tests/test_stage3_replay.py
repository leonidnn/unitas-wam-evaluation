import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'src/stage3_policy'
TEST_OUTPUT = ROOT / 'outputs/stage3-tests'
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPTS))
from replay_data import StateRecording, load_recording, restore_frame  # noqa: E402
from record_libero import RecordingEnvironment  # noqa: E402
import record_libero  # noqa: E402

XML = '''<mujoco>
  <size nuserdata="2"/>
  <asset><texture name="pattern" type="2d" builtin="checker" width="16" height="16"
    rgb1="1 0 0" rgb2="0 1 0"/><material name="surface" texture="pattern"/></asset>
  <worldbody>
    <geom type="plane" size="1 1 .1" material="surface"/>
    <body name="object" pos="0 0 .4"><freejoint/><geom type="sphere" size=".03"/></body>
    <body pos=".2 0 .2"><joint name="hinge" type="hinge"/>
      <geom type="capsule" size=".02 .1"/></body>
    <body mocap="true" pos=".3 0 .3"><geom type="sphere" size=".01"/></body>
  </worldbody>
  <actuator><general joint="hinge" dyntype="filter" dynprm=".1"/></actuator>
</mujoco>'''


class Environment:
    def __init__(self, model, data):
        self.model, self.data, self.closed = model, data, False

    def step(self, action):
        self.data.ctrl[0] = action[0]
        mujoco.mj_step(self.model, self.data)
        return {'observation': float(self.data.time)}, 0.0, False, {}

    def close(self):
        self.closed = True


class ReplayTests(unittest.TestCase):
    def make_recording(self, directory):
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.userdata[:] = [3.0, 4.0]
        data.mocap_pos[:] = [.1, .2, .3]
        mujoco.mj_forward(model, data)
        recording = StateRecording(directory, model, data, {'instruction': 'test'})
        return model, data, recording

    def test_round_trip_and_observer_does_not_modify_state(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            model, data, recording = self.make_recording(Path(folder) / 'episode')
            expected = [data.qpos.copy()]
            positions = []
            for step in range(8):
                data.ctrl[:] = step / 10
                mujoco.mj_step(model, data)
                before = np.empty(recording.metadata['state_width'])
                mujoco.mj_getState(model, data, before, recording.spec)
                recording.capture(np.full(7, step / 10), 2)
                after = np.empty_like(before)
                mujoco.mj_getState(model, data, after, recording.spec)
                np.testing.assert_array_equal(before, after)
                expected.append(data.qpos.copy())
                scratch = mujoco.MjData(model)
                mujoco.mj_setState(model, scratch, before, recording.spec)
                mujoco.mj_forward(model, scratch)
                positions.append(scratch.xpos.copy())
            recording.finish(result={'success': True})
            restored, playback, metadata, arrays = load_recording(recording.directory)
            self.assertEqual(restored.ntex, 1)
            self.assertEqual(metadata['frames'], 9)
            self.assertEqual(metadata['status'], 'completed')
            self.assertTrue(np.isnan(arrays['actions'][0]).all())
            for index in range(9):
                restore_frame(restored, playback, metadata, arrays, index)
                np.testing.assert_array_equal(playback.qpos, expected[index])
                np.testing.assert_array_equal(playback.userdata, [3.0, 4.0])
                np.testing.assert_array_equal(playback.mocap_pos, [[.1, .2, .3]])
                if index:
                    np.testing.assert_allclose(playback.xpos, positions[index-1], atol=1e-14)
                self.assertEqual(playback.time, arrays['times'][index])
            np.testing.assert_array_equal(arrays['actions'][-1], np.full(7, .7))

    def test_proxy_captures_initial_settle_and_final_policy_state(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            model, data, recording = self.make_recording(Path(folder) / 'episode')
            original = Environment(model, data)
            proxy = RecordingEnvironment(original, recording)
            self.assertEqual(proxy.step(np.zeros(7))[1:], (0.0, False, {}))
            proxy.phase = 2
            proxy.step(np.ones(7))
            final = data.qpos.copy()
            proxy.close()
            recording.finish(result={'success': False})
            m, d, metadata, arrays = load_recording(recording.directory)
            self.assertTrue(original.closed)
            np.testing.assert_array_equal(arrays['phases'], [0, 1, 2])
            self.assertEqual(len(arrays['states']), 3)
            restore_frame(m, d, metadata, arrays, 2)
            np.testing.assert_array_equal(d.qpos, final)

    def test_error_keeps_partial_recording(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            _, _, recording = self.make_recording(Path(folder) / 'episode')
            recording.finish(error=RuntimeError('policy unavailable'))
            _, _, metadata, arrays = load_recording(recording.directory)
            self.assertEqual(metadata['status'], 'error')
            self.assertEqual(len(arrays['states']), 1)

    def test_version_and_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            _, _, recording = self.make_recording(Path(folder) / 'episode')
            recording.finish()
            manifest = recording.directory / 'manifest.json'
            original = manifest.read_text()
            changed = json.loads(original)
            changed['mujoco_version'] = 'invalid'
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Install mujoco'):
                load_recording(recording.directory)
            manifest.write_text(original)
            with (recording.directory / 'states.npz').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_recording(recording.directory)

    def test_recorder_refuses_to_overwrite_an_episode(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            model, data, recording = self.make_recording(Path(folder) / 'episode')
            with self.assertRaises(FileExistsError):
                StateRecording(recording.directory, model, data, {})

    def test_wrapper_reuses_original_episode_loop_and_restores_functions(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as folder:
            model = mujoco.MjModel.from_xml_string(XML)
            data = mujoco.MjData(model)
            env = Environment(model, data)
            env.sim = types.SimpleNamespace(model=types.SimpleNamespace(_model=model),
                                            data=types.SimpleNamespace(_data=data))
            client = types.ModuleType('evaluation.libero.client')
            client.__file__ = __file__
            client.PolicyClient = types.SimpleNamespace(act=lambda *args: np.zeros((1, 7)))
            client.record_protocol = lambda *args: None
            client.create_environment = lambda *args: (env, {}, 'test', 'test_task')

            def settle(environment, observation, steps):
                for _ in range(steps):
                    observation = environment.step(np.zeros(7))[0]
                return observation, 0.0

            def episode(args, episode_id):
                environment, observation, _, _ = client.create_environment('test', 0, episode_id, 42)
                try:
                    client.settle_then_enable_absolute_control(environment, observation, 2)
                    actions = client.PolicyClient.act(None, {}, 'test', [])
                    environment.step(actions[0])
                    return {'success': True}
                finally:
                    environment.close()

            client.settle_then_enable_absolute_control = settle
            client.run_episode = episode
            client.main = lambda: client.run_episode(types.SimpleNamespace(output_dir=folder,
                                                        protocol={'test': True}), 0)
            package = types.ModuleType('evaluation')
            package.__path__ = []
            subpackage = types.ModuleType('evaluation.libero')
            subpackage.__path__ = []
            package.libero = subpackage
            subpackage.client = client
            original_create = client.create_environment
            with patch.dict(sys.modules, {'evaluation': package, 'evaluation.libero': subpackage,
                                          'evaluation.libero.client': client}):
                record_libero.main()
            self.assertIs(client.run_episode, episode)
            self.assertIs(client.create_environment, original_create)
            self.assertIs(client.settle_then_enable_absolute_control, settle)
            _, _, metadata, arrays = load_recording(Path(folder) / 'test_task00_episode000_replay')
            self.assertEqual(metadata['status'], 'completed')
            np.testing.assert_array_equal(arrays['phases'], [0, 1, 1, 2])
            timing = json.loads((Path(folder) / 'policy-request-timing.jsonl').read_text())
            self.assertEqual(timing['episode_id'], 0)
            self.assertEqual(timing['status'], 'ok')
            self.assertEqual(timing['action_count'], 1)


if __name__ == '__main__':
    unittest.main()
