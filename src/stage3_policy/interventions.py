"""Controlled disturbances applied to the simulator, independent of policy outputs."""

from __future__ import annotations

import xml.etree.ElementTree as ET

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


def reload_scene(environment, xml):
    """Recompile a scene edit that adds no degrees of freedom."""
    sim = environment.sim
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, spec))
    mujoco.mj_getState(sim.model._model, sim.data._data, state, spec)
    environment.env.reset_from_xml_string(ET.tostring(xml, encoding='unicode'))
    sim = environment.sim
    mujoco.mj_setState(sim.model._model, sim.data._data, state, spec)
    sim.forward()
    return sim


def lower_mug_friction(environment, scale, body_name='porcelain_mug_1_main'):
    """Override tangential friction for finger?mug contact pairs only."""
    if not np.isfinite(scale) or not 0 < scale < 1:
        raise ValueError('Friction scale must be in (0, 1)')
    sim = environment.sim
    model = sim.model._model
    body_id = model.body(body_name).id
    collidable = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    target = [i for i in collidable if model.geom_bodyid[i] == body_id]
    fingers = [i for i in collidable if 'finger' in model.geom(i).name
               and model.geom(i).name.startswith('gripper')]
    if not target or not fingers:
        raise ValueError('Expected target and finger collision geometries')
    xml = ET.fromstring(sim.model.get_xml())
    section = xml.find('contact')
    if section is None:
        section = ET.SubElement(xml, 'contact')
    existing = {frozenset((a, b)) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    changed = []
    for a in target:
        for b in fingers:
            if not ((model.geom_contype[a] & model.geom_conaffinity[b])
                    or (model.geom_contype[b] & model.geom_conaffinity[a])):
                continue
            if frozenset((a, b)) in existing:
                raise ValueError('Unexpected explicit finger?mug contact pair')
            first, second = model.geom(a).name, model.geom(b).name
            pa, pb = model.geom_priority[[a, b]]
            if pa != pb:
                idx = a if pa > pb else b
                condim = model.geom_condim[idx]
                solref, solimp = model.geom_solref[idx], model.geom_solimp[idx]
                friction = model.geom_friction[idx]
            else:
                condim = max(model.geom_condim[[a, b]])
                weights = np.where(model.geom_solmix[[a, b]] >= mujoco.mjMINVAL,
                                   model.geom_solmix[[a, b]], 0)
                weights = weights / weights.sum() if weights.any() else [0.5, 0.5]
                refs = model.geom_solref[[a, b]]
                solref = weights @ refs if (refs[:, 0] > 0).all() else refs.min(axis=0)
                solimp = weights @ model.geom_solimp[[a, b]]
                friction = model.geom_friction[[a, b]].max(axis=0)
            sliding, torsion, rolling = friction
            override = [sliding * scale, sliding * scale, torsion, rolling, rolling]
            ET.SubElement(section, 'pair', geom1=first, geom2=second, condim=str(condim),
                          friction=_xml_values(override), solref=_xml_values(solref),
                          solimp=_xml_values(solimp), margin=str(max(model.geom_margin[[a, b]])),
                          gap=str(max(model.geom_gap[[a, b]])))
            changed.append({'geom1': first, 'geom2': second, 'reference': float(sliding),
                            'sliding': float(override[0])})
    if not changed:
        raise ValueError('No collision-enabled finger?mug contact pairs')
    reload_scene(environment, xml)
    return {'type': 'low_mug_friction', 'body': body_name, 'scale': scale, 'pairs': changed}


def _xml_values(values):
    return ' '.join(str(float(value)) for value in values)
