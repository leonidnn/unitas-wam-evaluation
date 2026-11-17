from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/stage3_policy'))
from interventions import ObjectPush  # noqa: E402


class PushTests(unittest.TestCase):
    def test_known_impulse_and_force_reset(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep="0.05" gravity="0 0 0" integrator="Euler"/>
          <worldbody><body name="object" pos="0 0 1"><freejoint/>
          <geom type="sphere" size="0.05" mass="1"/></body></worldbody>
        </mujoco>''')
        data = mujoco.MjData(model)
        push = ObjectPush(model, data, 2.0, start_step=1, duration_steps=1, body_name='object')
        push.apply(0)
        mujoco.mj_step(model, data)
        np.testing.assert_array_equal(data.qvel, 0)
        push.apply(1)
        mujoco.mj_step(model, data)
        self.assertAlmostEqual(data.qvel[0], 0.1)
        push.apply(2)
        mujoco.mj_step(model, data)
        self.assertAlmostEqual(data.qvel[0], 0.1)
        np.testing.assert_array_equal(data.xfrc_applied, 0)


if __name__ == '__main__':
    unittest.main()
