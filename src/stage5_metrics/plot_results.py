"""Plot visible-point trajectory errors for the two fixed evaluation windows."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, label, group in zip(axes, ('050', '050', '100'), ('all', 'moving', 'all')):
        with np.load(args.root/f'window{label}/scene_flow.npz', allow_pickle=False) as p, \
                np.load(args.root/f'window{label}-execution/reference.npz', allow_pickle=False) as r:
            truth, prediction, visible = r['scene'], p['scene'], r['visibility'][1:]
            moving = np.linalg.norm(truth[1:]-truth[:1], axis=-1).max(axis=0) > .01
            if group == 'moving':
                visible = visible & moving[None]
            times = np.arange(1, len(truth))/float(r['fps'])
            for name, error, color, style in (
                ('UNITAS', np.linalg.norm(prediction[1:]-truth[1:], axis=-1), '#1465b0', '-'),
                ('Без движения', np.linalg.norm(truth[1:]-truth[:1], axis=-1), '#63666a', '--')):
                count = visible.sum(axis=1)
                mean = np.divide(np.where(visible, error, 0).sum(axis=1), count,
                                 out=np.full(len(times), np.nan), where=count > 0)
                axis.plot(times, mean*1000, label=name, color=color, linestyle=style, linewidth=2)
        axis.set_title(f'Окно {int(label)}: '+('подвижные точки' if group == 'moving' else 'все точки'))
        axis.set_xlabel('Время прогноза, с')
        axis.set_ylabel('Средняя ошибка, мм')
        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
