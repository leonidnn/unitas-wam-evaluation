"""Launch the original UNITAS server and measure synchronized policy latency."""

import json
import os
from pathlib import Path
import time


def main():
    import torch
    from unitas.inference.policy import Policy
    from unitas.inference.server import main as serve

    destination = Path(os.environ.get('UNITAS_TIMING_PATH', 'outputs/stage3/server-timing.jsonl'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = Policy.act

    def measured(self, observation, instruction):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        status = 'error'
        try:
            actions = original(self, observation, instruction)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            status = 'ok'
            return actions
        finally:
            # Измеряю весь Policy.act, включая подготовку наблюдения и перевод в команды.
            entry = {'elapsed_seconds': time.perf_counter()-start, 'status': status,
                     'measurement': 'synchronized_policy_act_including_pre_and_postprocessing'}
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry)+'\n')

    Policy.act = measured
    try:
        serve()
    finally:
        Policy.act = original


if __name__ == '__main__':
    main()
