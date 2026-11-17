import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'src/stage5_metrics/window.py'
spec = importlib.util.spec_from_file_location('scene_window', path)
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


class VisibilityTests(unittest.TestCase):
    def test_pixel_centers_depth_occlusion_and_camera_pose(self):
        camera = {'intrinsic': np.asarray([[1, 0, .5], [0, 1, .5], [0, 0, 1.]]),
                  'camera_to_world': np.eye(4), 'depth': np.ones((2, 2))}
        points = np.asarray([[0, 0, 1], [1, 0, 1], [0, 0, -1], [0, 0, 2]])
        np.testing.assert_array_equal(window.visibility(points, {'cam': camera}),
                                      [True, True, False, False])
        camera['camera_to_world'][0, 3] = 1
        np.testing.assert_array_equal(window.visibility(points[:2], {'cam': camera}),
                                      [False, True])


if __name__ == '__main__':
    unittest.main()
