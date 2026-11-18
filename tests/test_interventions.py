from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/stage3_policy'))
from interventions import ObjectPush, lower_mug_friction  # noqa: E402


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


class FrictionTests(unittest.TestCase):
    def test_finger_override_preserves_other_contact_parameters(self):
        xml = """<mujoco><worldbody>
          <geom name="gripper0_finger" type="box" size=".1 .1 .1"
                friction="2 .02 .003" condim="4" solref=".02 1.5"/>
          <body name="mug" pos=".18 0 0"><freejoint/>
            <geom name="mug_surface" type="sphere" size=".1" mass=".03"
                  friction=".95 .03 .004" solref=".001 1"/>
          </body>
        </worldbody></mujoco>"""
        env = SimpleNamespace()

        def compile_xml(source):
            model = mujoco.MjModel.from_xml_string(source)
            data = mujoco.MjData(model)
            env.sim = SimpleNamespace(model=SimpleNamespace(_model=model, get_xml=lambda: source),
                                      data=SimpleNamespace(_data=data),
                                      forward=lambda: mujoco.mj_forward(model, data))
            env.sim.forward()

        env.env = SimpleNamespace(reset_from_xml_string=compile_xml)
        compile_xml(xml)
        original = env.sim.data._data.contact[0]
        original_friction = original.friction.copy()
        original_solref, original_solimp = original.solref.copy(), original.solimp.copy()
        original_dim = original.dim
        lower_mug_friction(env, 0.2, body_name='mug')
        modified = env.sim.data._data.contact[0]
        np.testing.assert_allclose(modified.friction[:2], original_friction[:2] * 0.2)
        np.testing.assert_allclose(modified.friction[2:], original_friction[2:])
        np.testing.assert_allclose(modified.solref, original_solref)
        np.testing.assert_allclose(modified.solimp, original_solimp)
        self.assertEqual(modified.dim, original_dim)


if __name__ == '__main__':
    unittest.main()
