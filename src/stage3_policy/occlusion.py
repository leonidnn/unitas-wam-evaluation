"""Calibrate and measure a visual-only static occluder in LIBERO."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from interventions import reload_scene


NAME = 'unitas_occluder'
CAMERAS = ('agentview', 'robot0_eye_in_hand')


class MugOcclusion:
    def __init__(self, environment):
        sim = environment.sim
        xml = ET.fromstring(sim.model.get_xml())
        ET.SubElement(xml.find('worldbody'), 'geom', name=NAME, type='cylinder',
                      size='0.022 0.080', pos='0 0 -4', rgba='0.19 0.32 0.52 1',
                      group='1', contype='0', conaffinity='0')
        sim = reload_scene(environment, xml)
        self.model, self.data = sim.model._model, sim.data._data
        self.geom = self.model.geom(NAME).id
        self.target = self.model.body('porcelain_mug_1_main').id
        self.targets = np.flatnonzero(self.model.geom_bodyid == self.target)
        self.hidden = np.array([0., 0., -4.])
        self.renderer = None
        self.position = None
        self.spec = {'type': 'visual_only_cylinder', 'camera': CAMERAS[0],
                     'radius_m': 0.022, 'height_m': 0.160}

    def _move(self, position):
        self.model.geom_pos[self.geom] = position
        mujoco.mj_forward(self.model, self.data)

    def _mask(self, camera):
        self.renderer.update_scene(self.data, camera=camera)
        ids = self.renderer.render()[:, :, 0]
        return np.isin(ids, self.targets), ids == self.geom

    def calibrate(self):
        target = self.data.xpos[self.target].copy()
        camera = self.data.cam_xpos[self.model.camera(CAMERAS[0]).id]
        direction = camera[:2] - target[:2]
        direction /= np.linalg.norm(direction)
        lateral = np.array([-direction[1], direction[0]])
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.renderer.enable_segmentation_rendering()
        original, _ = self._mask(CAMERAS[0])
        if not original.any():
            raise ValueError('Target mug is invisible at calibration')
        candidates = []
        for distance in (0.18, 0.21, 0.15):
            for offset in np.linspace(-0.095, 0.095, 39):
                xy = target[:2] + distance * direction + offset * lateral
                geom = np.array([-1], dtype=np.int32)
                depth = mujoco.mj_ray(self.model, self.data, np.r_[xy, target[2] + 0.05],
                                      np.array([0., 0., -1.]), None, 1, -1, geom)
                if depth < 0:
                    continue
                surface = target[2] + 0.05 - depth
                if not target[2] - 0.09 < surface < target[2] + 0.015:
                    continue
                candidate = np.r_[xy, surface + 0.080]
                self._move(candidate)
                _, occluder = self._mask(CAMERAS[0])
                ratio = np.count_nonzero(original & occluder) / np.count_nonzero(original)
                if 0.20 <= ratio <= 0.30:
                    candidates.append((abs(ratio - 0.25), candidate, ratio))
                self._move(self.hidden)
        if not candidates:
            raise ValueError('No valid cylinder position with 20–30% initial occlusion')
        _, self.position, ratio = min(candidates, key=lambda candidate: candidate[0])
        self._move(self.position)
        self.spec.update(world_position_m=self.position.tolist(), calibration_overlap=ratio)
    def measure(self):
        result = {}
        try:
            for camera in CAMERAS:
                _, occluder = self._mask(camera)
                self._move(self.hidden)
                original, _ = self._mask(camera)
                self._move(self.position)
                count = int(original.sum())
                result[camera] = {'target_pixels_without_occluder': count,
                                  'occluded_target_pixels': int((original & occluder).sum())}
        finally:
            self._move(self.position)
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
