"""Plot descriptive paired LIBERO outcomes and synchronized policy timings."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(args.summary.read_text(encoding='utf-8'))
    if not data['complete']:
        raise ValueError('Complete both fixed episode lists before plotting outcomes')
    conditions = [data['baseline'], data['changed']]
    labels, colors = ['Baseline', 'Dim light'], ['#0072B2', '#D55E00']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].bar(labels, [row['success_rate'] for row in conditions], color=colors, width=.55)
    axes[0].set(ylim=(0, 1.15), ylabel='Success rate', title='One task, fixed initial states')
    for index, row in enumerate(conditions):
        axes[0].text(index, row['success_rate']+.03,
                     f"{row['successes']}/{len(row['expected_episodes'])}", ha='center')
    # Соединяю одинаковые начальные состояния, а не сортирую по успешности.
    for base, changed in zip(conditions[0]['episodes'], conditions[1]['episodes']):
        if base['episode_id'] != changed['episode_id']:
            raise ValueError('Episode order differs')
        axes[1].plot([0, 1], [base['steps'], changed['steps']], 'o-', color='#666666', alpha=.55)
    axes[1].set(xticks=[0, 1], xticklabels=labels, ylabel='Executed policy steps',
                title='Paired episode lengths')
    for label, color, row in zip(labels, colors, conditions):
        directory = args.summary.parent / Path(row['directory']).name
        timings = [json.loads(line) for line in
                   (directory/'server-policy-timing.jsonl').read_text().splitlines() if line.strip()]
        values = [entry['elapsed_seconds'] for entry in timings if entry['status']=='ok']
        axes[2].plot(range(1, len(values)+1), values, label=label, color=color, linewidth=1)
    axes[2].axhline(.5, color='black', linestyle='--', linewidth=1, label='10 actions / 20 Hz')
    axes[2].set(xlabel='Policy request index within condition', ylabel='Seconds',
                title='Synchronized Policy.act, T4 FP32')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
