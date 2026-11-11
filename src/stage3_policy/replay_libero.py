"""Replay a recorded MuJoCo scene locally without UNITAS or LIBERO."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import numpy as np

from replay_data import load_recording, restore_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Episode *_replay directory')
    parser.add_argument('--check-only', action='store_true', help='Restore every frame without GUI')
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    model, data, metadata, arrays = load_recording(args.input)
    total = len(arrays['states'])
    print(f"{metadata['instruction']} | {total} states | status={metadata['status']}")
    if args.check_only:
        for index in range(total):
            restore_frame(model, data, metadata, arrays, index)
            if not np.isfinite(data.xpos).all():
                raise ValueError(f'Invalid body positions in frame {index}')
        print('All frames restored; no physics steps or model inference performed')
        return

    import mujoco.viewer

    lock = threading.Lock()
    control = {'frame': 0, 'play': False, 'speed': args.speed}

    def key_callback(key):
        with lock:
            if key == 32:
                control['play'] = not control['play']
            elif key in (262, 263, 268, 269):
                control['play'] = False
                if key == 268:
                    control['frame'] = 0
                elif key == 269:
                    control['frame'] = total - 1
                else:
                    control['frame'] = int(np.clip(control['frame'] + (1 if key == 262 else -1), 0, total-1))
            elif key in (61, 45):
                control['speed'] = float(np.clip(control['speed'] * (2 if key == 61 else .5), .0625, 16))
            print(f"Frame {control['frame']}/{total-1}, t={arrays['times'][control['frame']]:.3f}s, "
                  f"speed={control['speed']:g}, play={control['play']}")

    print('Mouse: rotate/zoom/pan. Space: play/pause. Arrows: step. Home/End: first/last. +/-: speed.')
    restore_frame(model, data, metadata, arrays, 0)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        with viewer.lock():
            # Беру видимые группы из robosuite/environments/base.py: коллизии 0 скрываю, поверхности 1 показываю.
            viewer.opt.geomgroup[0] = 0
            viewer.opt.geomgroup[1] = 1
        previous = time.monotonic()
        last_frame = 0
        elapsed = 0.0
        was_playing = False
        while viewer.is_running():
            now = time.monotonic()
            with lock:
                index = control['frame']
                playing = control['play']
                if index != last_frame or not playing or not was_playing:
                    elapsed = 0.0
                if playing:
                    elapsed += (now - previous) * control['speed']
                    while index < total-1:
                        duration = max(float(arrays['times'][index+1] - arrays['times'][index]), .001)
                        if elapsed < duration:
                            break
                        elapsed -= duration
                        index += 1
                    control['frame'] = index
                    if index == total-1:
                        control['play'] = False
                was_playing = control['play']
                last_frame = index
            previous = now
            with viewer.lock():
                restore_frame(model, data, metadata, arrays, index)
            viewer.sync()
            time.sleep(.01)


if __name__ == '__main__':
    main()
