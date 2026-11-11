"""Save and restore MuJoCo states at LIBERO control-step boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StateRecording:
    def __init__(self, directory, model, data, metadata):
        import mujoco

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model, self.data = model, data
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.frames, self.times, self.actions, self.phases = [], [], [], []
        self.saved = False
        # Беру штатные mj_saveModel и mjSTATE_INTEGRATION из MuJoCo.
        mujoco.mj_saveModel(model, str(self.directory / 'scene.mjb'))
        self.metadata = {
            **metadata, 'format': 'unitas-libero-replay-v1', 'mujoco_version': mujoco.__version__,
            'state_spec': self.spec, 'state_width': mujoco.mj_stateSize(model, self.spec),
            'scene_sha256': sha256(self.directory / 'scene.mjb'),
            'status': 'incomplete', 'sampling': 'initial state and after each env.step',
            'phase_codes': {'0': 'initial', '1': 'settle', '2': 'policy'},
            'action_alignment': 'actions[i] led from states[i-1] to states[i]; row 0 is NaN',
            'recorder_sha256': sha256(__file__),
        }
        self.write_metadata()
        self.capture(None, 0)

    def write_metadata(self):
        path = self.directory / 'manifest.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        temporary.replace(path)

    def capture(self, action, phase):
        import mujoco

        state = np.empty(self.metadata['state_width'], dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self.spec)
        if not np.isfinite(state).all():
            raise ValueError('Cannot record non-finite simulator state')
        self.frames.append(state)
        self.times.append(float(self.data.time))
        self.actions.append(np.full(7, np.nan) if action is None else np.array(action, copy=True))
        self.phases.append(phase)

    def save(self):
        if self.saved:
            return
        path = self.directory / 'states.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, states=np.stack(self.frames),
                                times=np.asarray(self.times), actions=np.stack(self.actions),
                                phases=np.asarray(self.phases, dtype=np.uint8))
        temporary.replace(path)
        self.metadata.update(frames=len(self.frames), trajectory_sha256=sha256(path))
        self.write_metadata()
        self.saved = True

    def finish(self, result=None, error=None):
        self.save()
        if error is not None:
            self.metadata.update(status='error', error_type=type(error).__name__, error=str(error))
        else:
            self.metadata.update(status='completed', result=result)
        self.write_metadata()


def load_recording(directory):
    import mujoco

    directory = Path(directory)
    metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if metadata['format'] != 'unitas-libero-replay-v1':
        raise ValueError('Unknown recording format')
    if metadata['mujoco_version'] != mujoco.__version__:
        raise ValueError(f"Install mujoco=={metadata['mujoco_version']} for this recording")
    if sha256(directory / 'scene.mjb') != metadata['scene_sha256']:
        raise ValueError('Scene checksum mismatch')
    if sha256(directory / 'states.npz') != metadata['trajectory_sha256']:
        raise ValueError('Trajectory checksum mismatch')
    model = mujoco.MjModel.from_binary_path(str(directory / 'scene.mjb'))
    data = mujoco.MjData(model)
    with np.load(directory / 'states.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('states', 'times', 'actions', 'phases')}
    states, times = arrays['states'], arrays['times']
    width = mujoco.mj_stateSize(model, metadata['state_spec'])
    if states.shape != (metadata['frames'], width) or len(states) == 0:
        raise ValueError('Unexpected recorded state dimensions')
    if times.shape != (len(states),) or arrays['actions'].shape != (len(states), 7):
        raise ValueError('Misaligned timestamps/actions')
    if arrays['phases'].shape != times.shape or not np.isin(arrays['phases'], [0, 1, 2]).all():
        raise ValueError('Invalid phase data')
    if not np.isfinite(states).all() or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('Invalid state values/timestamps')
    return model, data, metadata, arrays


def restore_frame(model, data, metadata, arrays, index):
    import mujoco

    # Восстанавливаю записанное состояние; новый шаг физики не выполняю.
    mujoco.mj_setState(model, data, arrays['states'][index], metadata['state_spec'])
    mujoco.mj_forward(model, data)
