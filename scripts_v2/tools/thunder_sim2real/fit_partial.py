"""Fit saved hardware data, optionally leaving a chronological suffix untouched.

Uses UWLab's native simulator, controller and CMA optimizer. The recording stays
unchanged; this experiment does not export a deployment profile.
"""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from isaaclab.app import AppLauncher
from records import JOINT_NAMES, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--record', type=Path, required=True)
parser.add_argument('--warm_start', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=128)
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--train_steps', type=int, default=1600,
                    help='Number of samples used for fitting; 0 uses the complete recording')
parser.add_argument('--velocity_weight_s', type=float, default=.05)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', action='store_true', help='Resume this locally generated experiment checkpoint')
parser.add_argument('--evaluate_only', action='store_true', help='Evaluate the existing training checkpoint')
parser.add_argument('--patience', type=int, default=8, help='Stop after this many generations improve the best score by less than 1 percent')
parser.add_argument('--parameterization', choices=('linear', 'logit'), default='linear')
parser.add_argument('--search_scale', type=float, default=1., help='Scale the initial search standard deviation')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
record = torch.load(args.record, weights_only=True, map_location='cpu')
n = len(record['joint_positions'])
if args.train_steps == 0:
    args.train_steps = n
assert record['robot'] == 'thunder' and record['joint_names'] == JOINT_NAMES
assert record['sample_phase'] == 'pre_command' and record['dt'] == .002
assert 0 < args.train_steps <= n and args.num_envs >= 9 and args.iterations > 0
assert np.isfinite(args.velocity_weight_s) and args.velocity_weight_s > 0
assert np.isfinite(args.search_scale) and args.search_scale > 0
for key, width in [('joint_positions', 6), ('joint_velocities', 6),
                   ('waypoint_target_pos', 3), ('waypoint_target_quat', 4)]:
    assert record[key].shape == (n, width) and torch.isfinite(record[key]).all(), key
warm_document = json.loads(args.warm_start.read_text())
warm = np.asarray(warm_document['best_params'], dtype=float)
assert warm.shape == (25,) and np.isfinite(warm).all()
args.output.mkdir(parents=True, exist_ok=args.resume or args.evaluate_only)
app = AppLauncher(args).app

from cmaes import CMA
from replay import ThunderReplay, parameter_dict
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_sim2real_cfg import ThunderSysidCfg


def write(name, value):
    target = args.output / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(target)


def metrics(q, v, start, stop):
    real_q = record['joint_positions'].numpy()[start:stop]
    real_v = record['joint_velocities'].numpy()[start:stop]
    qe, ve = q[start:stop] - real_q, v[start:stop] - real_v
    return dict(samples=stop-start,
                position_rmse_deg=np.degrees(np.sqrt(np.mean(qe**2, axis=0))).tolist(),
                velocity_rmse_deg_s=np.degrees(np.sqrt(np.mean(ve**2, axis=0))).tolist(),
                position_rmse_all_deg=float(np.degrees(np.sqrt(np.mean(qe**2)))),
                velocity_rmse_all_deg_s=float(np.degrees(np.sqrt(np.mean(ve**2)))),
                weighted_score=float(np.mean(qe**2 + args.velocity_weight_s**2*ve**2, axis=0).sum()),
                predicted_peak_speed_deg_s=np.degrees(np.abs(v[start:stop]).max(axis=0)).tolist(),
                measured_peak_speed_deg_s=np.degrees(np.abs(real_v).max(axis=0)).tolist())


def main():
    base = dict(record_path=str(args.record.resolve()), record_sha256=sha256(args.record),
                record_completed=record['completed'], record_failure=record['failure'],
                source_kind=record['source_kind'], controller=record['osc_params'],
                joint_names=record['joint_names'], train_steps=args.train_steps, total_steps=n,
                held_out_steps=n-args.train_steps, comparison_phase='pre_command', dt=.002,
                velocity_weight_s=args.velocity_weight_s, seed=args.seed,
                parameterization=args.parameterization, search_scale=args.search_scale,
                warm_start_path=str(args.warm_start.resolve()), warm_start_sha256=sha256(args.warm_start),
                scope=('Single-record dynamics estimate for planning the next collection. '
                       'Only this recording contributes to the objective. Command-index alignment '
                       'at nominal 2 ms; original timing and samples are unchanged. No deployment-profile export. '
                       + ('All samples are used for fitting; reported errors are training errors.'
                          if args.train_steps == n else
                          'Chronological suffix is never used by the optimizer.')))
    high = np.array([10.]*6 + [20.]*6 + [1.]*6 + [60.]*6 + [5.])
    low = np.array([1e-4]*6 + [0.]*19)
    if not np.all((warm >= low) & (warm <= high)):
        raise ValueError('Warm-start parameters are outside the physical search bounds')
    std = np.r_[.5*warm[:6]+.02, .5*warm[6:12]+.2, [0.15]*6,
                .5*warm[18:24]+.3, 1.2]
    if args.parameterization == 'logit':
        fraction = np.clip((warm-low)/(high-low), 1e-6, 1-1e-6)
        mean = np.log(fraction/(1-fraction))
        # Search smoothly inside the physical bounds. This avoids the repeated
        # multivariate rejection that biases a linear Gaussian away from zero.
        std_search = np.clip(std / ((high-low)*fraction*(1-fraction)), .2, 1.)*args.search_scale
        def decode(x):
            return low+(high-low)/(1+np.exp(-np.clip(x, -40, 40)))
        optimizer = CMA(mean=mean, sigma=1., cov=np.diag(std_search**2),
                        population_size=args.num_envs, seed=args.seed)
    else:
        mean = warm/high
        def decode(x):
            return x*high
        optimizer = CMA(mean=mean, sigma=1., cov=np.diag((args.search_scale*std/high)**2),
                        bounds=np.column_stack([low/high, np.ones(25)]),
                        population_size=args.num_envs, seed=args.seed)
    cfg = ThunderSysidCfg()
    cfg.scene.robot.actuators['arm'].velocity_limit = 1000.
    cfg.scene.robot.actuators['arm'].velocity_limit_sim = 1000.
    replay = ThunderReplay(record, num_envs=args.num_envs, device=args.device,
                          delay_max=5, seed=args.seed, env_cfg=cfg)
    elite, history = [], []
    if args.resume or args.evaluate_only:
        saved = json.loads((args.output/'best_fit.json').read_text())
        for key in ('record_sha256', 'train_steps', 'velocity_weight_s', 'seed'):
            assert saved[key] == base[key], f'Resume mismatch: {key}'
        assert saved.get('parameterization', 'linear') == args.parameterization
        elite, history = saved['elite'], saved['history']
        if args.resume:
            # Only resume a checkpoint created locally by this program.
            optimizer = pickle.loads((args.output/'optimizer.pkl').read_bytes())
            assert optimizer.population_size == args.num_envs
    try:
        limits = replay.robot.root_physx_view.get_dof_max_velocities()[:, replay.joint_ids].cpu().numpy()
        assert np.allclose(limits, 1000.)
        base['simulation_velocity_limits_rad_s'] = limits[0].tolist()
        for iteration in range(len(history), args.iterations if not args.evaluate_only else len(history)):
            started = time.monotonic()
            normalized = np.array([optimizer.ask() for _ in range(args.num_envs)])
            if iteration == 0:
                normalized[0] = mean
            candidates = decode(normalized)
            result = replay.run(candidates, max_steps=args.train_steps,
                                velocity_weight_s=args.velocity_weight_s)
            scores = np.nan_to_num(result['scores'], nan=1e12, posinf=1e12, neginf=1e12)
            optimizer.tell(list(zip(normalized, scores.astype(float))))
            for i in np.argsort(scores)[:8]:
                if scores[i] < 1e11:
                    elite.append(dict(best_params=candidates[i].tolist(), score=float(scores[i]),
                                      iteration=iteration+1,
                                      position_rmse_deg=np.degrees(result['per_joint_rmse_rad'][i]).tolist(),
                                      velocity_rmse_deg_s=np.degrees(result['per_joint_velocity_rmse_rad_s'][i]).tolist()))
            elite = sorted(elite, key=lambda e: e['score'])[:16]
            if not elite:
                raise RuntimeError('No finite simulation candidate')
            best = elite[0]
            history.append(dict(iteration=iteration+1, best_score=best['score'],
                                population_median=float(np.median(scores)), seconds=time.monotonic()-started))
            write('best_fit.json', {**base, **best, 'completed_generations': iteration+1,
                                   'delay_steps': int(np.rint(best['best_params'][24])),
                                   'sysid': parameter_dict(best['best_params']), 'bounds_lower': low.tolist(),
                                   'bounds_upper': high.tolist(), 'history': history, 'elite': elite})
            checkpoint = args.output / 'optimizer.tmp'
            checkpoint.write_bytes(pickle.dumps(optimizer))
            checkpoint.replace(args.output / 'optimizer.pkl')
            print(f"Iteration {iteration+1}/{args.iterations}: score {best['score']:.6g}; "
                  f"q RMSE {np.sqrt(np.mean(np.square(best['position_rmse_deg']))):.3f} deg; "
                  f"speed RMSE {np.sqrt(np.mean(np.square(best['velocity_rmse_deg_s']))):.2f} deg/s; "
                  f"{history[-1]['seconds']:.1f} s", flush=True)
            if len(history) >= max(12, args.patience+1):
                previous = history[-args.patience-1]['best_score']
                if best['score'] >= .99*previous:
                    print(f'Stopping: improvement below 1 percent over {args.patience} generations.', flush=True)
                    break
        # All held-out evaluations happen after the last optimizer update. Each
        # trajectory continues from the initial state through the split boundary.
        chosen = elite[:8] + [dict(best_params=warm.tolist(), score=None, iteration=0)]
        params = np.array([e['best_params'] for e in chosen])
        padded = np.concatenate([params, np.tile(params[0], (args.num_envs-len(params), 1))])
        result = replay.run(padded, trajectory=True, velocity_weight_s=args.velocity_weight_s)
        q = result['joint_positions'][:, :len(chosen)]
        v = result['pre_command_joint_velocities'][:, :len(chosen)]
        rows = []
        for i, item in enumerate(chosen):
            train = metrics(q[:, i], v[:, i], 0, args.train_steps)
            if i == 0:
                assert np.isclose(train['weighted_score'], elite[0]['score'], rtol=.02, atol=1e-7), (train, elite[0])
            rows.append(dict(name=f'fit_train_rank_{i+1}' if i < 8 else 'warm_reference',
                             params=item['best_params'], train=train,
                             held_out=(metrics(q[:, i], v[:, i], args.train_steps, n)
                                       if args.train_steps < n else None)))
        write('evaluation.json', {**base, 'rows': rows})
        np.savez_compressed(args.output/'evaluation.npz', real_q=record['joint_positions'].numpy(),
                            real_v=record['joint_velocities'].numpy(), simulated_q=q, simulated_v=v,
                            train_steps=args.train_steps, dt=.002)
        write('fitted_variants.json', dict(rows=[dict(variant=row['name'], params=row['params']) for row in rows[:8]],
                                         source_fit=str((args.output/'best_fit.json').resolve()),
                                         selection=('Top scores on the sole input recording.' if args.train_steps == n
                                                    else 'Top training scores only; suffix did not select candidates.')))
        print(json.dumps({'evaluation': rows[0], 'baseline': rows[-1]}, indent=2), flush=True)
    finally:
        replay.close()


try:
    main()
except BaseException as exc:
    write('failure.json', dict(error=str(exc)))
    raise
finally:
    app.close()
