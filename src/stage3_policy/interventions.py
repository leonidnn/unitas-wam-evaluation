"""Controlled disturbances applied to the simulator, independent of policy outputs."""

from __future__ import annotations

import mujoco
import numpy as np


class ObjectPush:
    def __init__(self, model, data, force_n, start_step=60, duration_steps=3,
                 body_name='porcelain_mug_1_main'):
        if not np.isfinite(force_n) or force_n <= 0:
            raise ValueError('Push force must be finite and positive')
        if start_step < 0 or duration_steps < 1:
            raise ValueError('Invalid disturbance interval')
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f'Cannot push absent body {body_name}')
        self.data = data
        self.force_n, self.start_step, self.duration_steps = force_n, start_step, duration_steps
        self.spec = {'type': 'external_force', 'body': body_name, 'world_force_n': [force_n, 0, 0],
                     'start_policy_step': start_step, 'duration_control_steps': duration_steps,
                     'control_frequency_hz': 20}

    def apply(self, policy_step):
        self.data.xfrc_applied[self.body_id] = 0
        if self.start_step <= policy_step < self.start_step + self.duration_steps:
            self.data.xfrc_applied[self.body_id, 0] = self.force_n
