"""Plot measured/optimized joint traces and their chronological train/test split."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit', required=True, type=Path)
parser.add_argument('--baseline_comparison', type=Path,
                    help='Earlier native replay comparison containing the unfitted half-reference model')
args = parser.parse_args()
report = json.loads((args.fit/'evaluation.json').read_text())
data = np.load(args.fit/'evaluation.npz')
t = np.arange(len(data['real_q']))*float(data['dt'])
split = int(data['train_steps'])*float(data['dt'])
baseline = None
if args.baseline_comparison:
    baseline_report = json.loads((args.baseline_comparison/'comparison.json').read_text())
    baseline_data = np.load(args.baseline_comparison/'trajectories.npz')
    baseline_index = next(i for i, row in enumerate(baseline_report['rows'])
                          if row['variant'] == 'UW_reference_0.5x_delay_4ms')
    for signal in ('q', 'v'):
        assert np.array_equal(baseline_data['real_'+signal], data['real_'+signal])
    baseline = {signal: baseline_data['simulated_'+signal][:, baseline_index] for signal in ('q', 'v')}
for signal, unit, name in [('q', 'degrees', 'angles'), ('v', 'degrees/s', 'speeds')]:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    actual = np.degrees(data['real_'+signal])
    pred = np.degrees(data['simulated_'+signal])
    for j, ax in enumerate(axes.flat):
        ax.axvspan(split, t[-1], color='#edf4ff', label='Held out')
        ax.plot(t, actual[:, j], color='#222222', linewidth=1.4, label='Measured hardware')
        baseline_curve = pred[:, -1, j] if baseline is None else np.degrees(baseline[signal][:, j])
        ax.plot(t, baseline_curve, color='#999999', linewidth=1, alpha=.8,
                label='Before local refinement' if baseline is None else 'Unfitted half-reference model')
        ax.fill_between(t, pred[:, :8, j].min(axis=1), pred[:, :8, j].max(axis=1), color='#df7228', alpha=.18)
        ax.plot(t, pred[:, 0, j], color='#cd5f12', linewidth=1.2, label='Best fitted model')
        ax.axvline(split, color='#6683a1', linewidth=.8, linestyle='--')
        ax.set_title(report['joint_names'][j].replace('_joint', '').replace('_', ' ').title())
        ax.set_ylabel(unit); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[-1, 0].set_xlabel('Seconds from first command'); axes[-1, 1].set_xlabel('Seconds from first command')
    fig.suptitle('Thunder partial-run fit — '+name+'\nFirst 3.2 seconds optimized; final 0.86 seconds predicted without resetting', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.fit/('fit_'+name+'.png'), dpi=150); plt.close(fig)
fit = json.loads((args.fit/'best_fit.json').read_text())
history = fit['history']
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot([h['iteration'] for h in history], [h['best_score'] for h in history], marker='o', markersize=3)
ax.set_xlabel('Optimization generation (128 candidates each)')
ax.set_ylabel('Angle + weighted speed error'); ax.set_yscale('log'); ax.grid(alpha=.2)
ax.set_title('Training objective — lower is better; held-out data excluded')
fig.tight_layout(); fig.savefig(args.fit/'optimization_progress.png', dpi=150); plt.close(fig)
print(json.dumps({'fit': report['rows'][0], 'starting_reference': report['rows'][-1]}, indent=2))
