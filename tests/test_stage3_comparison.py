import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/stage3_policy'))
from compare_checkpoints import compare
import run_evaluation


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / 'outputs/stage3-tests'
        output.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=output)
        self.root = Path(self.temp.name)
        # Задаю тестовый протокол явно, чтобы проверка работала в чистом клоне.
        self.protocol = {
            'suite': 'libero_90', 'scene_condition': {'light_scale': 1.0},
            'server': {
                'checkpoint': 'unitas_libero_finetuned.pt', 'checkpoint_sha256': 'fixture',
                'sampling_seed': 160484,
                'temporal_spec': dict(prediction_horizon=20, history_horizon=20,
                                      gripper_flow_fps=20.0, scene_flow_horizon=20,
                                      scene_flow_fps=20.0),
            },
        }
        for model in ('finetuned', 'scratch'):
            for condition, scale in (('baseline', 1.), ('dim_light', .5)):
                folder = self.root / model / condition
                folder.mkdir(parents=True)
                p = copy.deepcopy(self.protocol)
                p['scene_condition']['light_scale'] = scale
                p['server']['checkpoint'] = f'unitas_libero_{model}.pt'
                p['server']['checkpoint_sha256'] = model
                (folder / 'libero_90_protocol.json').write_text(json.dumps(p))
                row = dict(suite='libero_90', task_id=72, episode_id=0, max_steps=800,
                           settle_steps=10, success=model == 'finetuned', steps=158)
                (folder / 'libero_90_task72_episode000.json').write_text(json.dumps(row))
                replay = folder / 'libero_90_task72_episode000_replay'
                replay.mkdir()
                np.savez(replay / 'states.npz', states=np.zeros((11, 4)), times=np.arange(11),
                         phases=np.array([0] + [1] * 10))

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_and_incomplete(self):
        result = compare(self.root / 'finetuned', self.root / 'scratch', count=1)
        self.assertTrue(result['complete'])
        self.assertFalse(result['paired']['baseline'][0]['scratch_success'])
        (self.root / 'scratch/baseline/libero_90_task72_episode000.json').unlink()
        result = compare(self.root / 'finetuned', self.root / 'scratch', count=1)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['groups']['scratch/baseline']['success_rate'])

    def test_reject_changed_protocol(self):
        p = self.root / 'scratch/baseline/libero_90_protocol.json'
        value = json.loads(p.read_text())
        value['server']['sampling_seed'] += 1
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Incompatible protocol'):
            compare(self.root / 'finetuned', self.root / 'scratch', count=1)

    def test_native_temporal_config_requires_explicit_option(self):
        for condition in ('baseline', 'dim_light'):
            p = self.root / 'scratch' / condition / 'libero_90_protocol.json'
            value = json.loads(p.read_text())
            for key in ('prediction_horizon', 'history_horizon', 'scene_flow_horizon'):
                value['server']['temporal_spec'][key] = 30
            p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Incompatible protocol'):
            compare(self.root / 'finetuned', self.root / 'scratch', count=1)
        self.assertTrue(compare(self.root / 'finetuned', self.root / 'scratch', count=1,
                                native_temporal_config=True)['complete'])
        p = self.root / 'scratch/baseline/libero_90_protocol.json'
        value = json.loads(p.read_text())
        value['server']['temporal_spec']['history_horizon'] = 31
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Unexpected native temporal spec'):
            compare(self.root / 'finetuned', self.root / 'scratch', count=1,
                    native_temporal_config=True)

    def test_reject_initial_state_mismatch(self):
        p = self.root / 'scratch/baseline/libero_90_task72_episode000_replay/states.npz'
        np.savez(p, states=np.ones((11, 4)), times=np.arange(11), phases=np.array([0]+[1]*10))
        with self.assertRaisesRegex(ValueError, 'Initial/settle mismatch'):
            compare(self.root / 'finetuned', self.root / 'scratch', count=1)

    def test_selected_checkpoint_reaches_server_command(self):
        output = self.root / 'mock-run'
        connection = MagicMock()
        connection.__enter__.return_value.connect_ex.return_value = 1
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"domain":"libero"}'
        server, client = MagicMock(), MagicMock()
        server.poll.return_value = None
        client.wait.return_value = 0
        argv = ['run_evaluation.py', '--condition', 'baseline', '--checkpoint', 'scratch.pt',
                '--root', str(ROOT), '--output-dir', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(run_evaluation.socket, 'socket', return_value=connection), \
             patch.object(run_evaluation.subprocess, 'check_output', side_effect=['72\n', 'revision\n']), \
             patch.object(run_evaluation.subprocess, 'Popen', side_effect=[server, client]) as launch, \
             patch.object(run_evaluation.urllib.request, 'urlopen', return_value=response), \
             patch.object(run_evaluation, 'stop'), patch.object(run_evaluation.signal, 'signal'):
            run_evaluation.main()
        command = launch.call_args_list[0].args[0]
        self.assertEqual(command[command.index('--checkpoint')+1], 'scratch.pt')
        self.assertEqual(json.loads((output/'run.json').read_text())['server_command'], command)


if __name__ == '__main__':
    unittest.main()
