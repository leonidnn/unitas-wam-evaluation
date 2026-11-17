import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'src/stage5_metrics/score.py'
spec = importlib.util.spec_from_file_location('scene_metrics', path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MetricTests(unittest.TestCase):
    def test_stationary_predictor_and_occluded_final_point(self):
        # Две точки: первая движется на 2/4 см, вторая остаётся на месте.
        truth = np.zeros((3, 2, 3))
        truth[1:, 0, 0] = [0.02, 0.04]
        prediction = np.zeros_like(truth)
        visible = np.ones((3, 2), bool)
        visible[-1, 0] = False
        result = metrics.score(prediction, truth, visible)
        full = result['groups']['all_correspondences']
        self.assertAlmostEqual(full['moving']['unitas']['ade_m'], 0.03)
        self.assertAlmostEqual(full['all']['unitas']['ade_m'], 0.015)
        self.assertAlmostEqual(full['moving']['unitas']['fde_m'], 0.04)
        self.assertEqual(full['static']['unitas']['fde_m'], 0)
        self.assertEqual(full['all']['unitas'], full['all']['stationary'])
        self.assertIsNone(result['groups']['visible']['moving']['unitas']['fde_m'])
        self.assertAlmostEqual(result['groups']['visible']['moving']['unitas']['ade_m'], 0.02)

    def test_wrong_coordinate_frame_is_rejected(self):
        truth = np.zeros((3, 2, 3))
        with self.assertRaisesRegex(ValueError, 'Initial corresponding'):
            metrics.score(truth + 0.1, truth, np.ones((3, 2), bool))

    def test_perfect_prediction_and_empty_moving_group(self):
        truth = np.zeros((3, 2, 3))
        result = metrics.score(truth, truth, np.ones((3, 2), bool))
        self.assertEqual(result['groups']['visible']['all']['unitas']['ade_m'], 0)
        self.assertIsNone(result['groups']['visible']['moving']['unitas']['ade_m'])

    def test_rigidity_separates_translation_from_deformation(self):
        initial = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        rigid = np.stack([initial, initial + [0.2, 0.3, 0.4]])
        self.assertLess(metrics.rigidity(rigid, [1, 1, 1])['1']['max_rigid_fit_rms_m'], 1e-12)
        rigid[1, 0, 0] += 0.1
        self.assertGreater(metrics.rigidity(rigid, [1, 1, 1])['1']['max_rigid_fit_rms_m'], 0.01)


if __name__ == '__main__':
    unittest.main()
