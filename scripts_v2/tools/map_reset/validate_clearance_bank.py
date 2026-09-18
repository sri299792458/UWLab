"""Reload saved resets in Isaac Sim, audit geometry, and measure one-second holds.

This is validation only: it neither filters nor rewrites the input reset banks.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=256)
p.add_argument('--families', nargs='+', default=['ObjectAnywhereEEAnywhere', 'ObjectAnywhereEEGrasped',
    'ObjectPartiallyAssembledEEGrasped', 'ObjectRestingEEGrasped'])
p.add_argument('--seed', type=int, default=73)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args, multi_gpu=False).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_reset_cfg import TABLE_Z
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.umi_sphere_geometry import UmiSphereGeometry
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.umi_map_reset import AtlasSeeds


def leaves(node, prefix=''):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, prefix+'/'+k)
    else:
        yield prefix, node


def stack(node):
    if isinstance(node, dict):
        return {k: stack(v) for k, v in node.items()}
    return torch.stack(node)


def take(node, indices):
    if isinstance(node, dict):
        return {k: take(v, indices) for k, v in node.items()}
    return node[indices].to(args.device)


def support_gap(env):
    top, bottom = env.scene['insertive_object'], env.scene['receptive_object']
    th = .03 * math_utils.matrix_from_quat(top.data.root_quat_w).abs().sum(dim=2)
    bh = .03 * math_utils.matrix_from_quat(bottom.data.root_quat_w).abs().sum(dim=2)
    overlap = ((top.data.root_pos_w[:, :2] - bottom.data.root_pos_w[:, :2]).abs() <= th[:, :2]+bh[:, :2]).all(dim=1)
    table = env.scene.env_origins[:, 2] + TABLE_Z
    support = torch.where(overlap, torch.maximum(table, bottom.data.root_pos_w[:, 2]+bh[:, 2]), table)
    return top.data.root_pos_w[:, 2] - th[:, 2] - support


cfg = UmiCubeTrainCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.events.reset_from_reset_states = None
cfg.terminations.abnormal_robot = None
cfg.episode_length_s = 60
env = gym.make('OmniReset-UMI-Defaults-State-Train-v0', cfg=cfg).unwrapped
env.reset()
robot = env.scene['robot']
geometry = UmiSphereGeometry(robot.body_names, env.device)
atlas = AtlasSeeds(env)
report = dict(dataset_dir=str(args.dataset_dir), families={}, seed=args.seed,
    kp=list(cfg.actions.arm.motion_stiffness),
    kd=[2 * kp**.5 * ratio for kp, ratio in zip(cfg.actions.arm.motion_stiffness, cfg.actions.arm.motion_damping_ratio)],
    physics_dt=env.physics_dt, policy_dt=env.step_dt, all_saved_states_valid=False,
    hold_probe='One second with zero arm actions and saved hand opening/closing; freshly randomized task masses. A controller probe, not policy performance.')
args.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


try:
    for family in args.families:
        path = args.dataset_dir / 'Resets/InsertiveAprilCube60__ReceptiveAprilCube60' / f'resets_{family}.pt'
        raw = torch.load(path, map_location='cpu', weights_only=False)
        state = stack(raw['initial_state'])
        count = len(state['articulation']['robot']['joint_position'])
        tensors = dict(leaves(state))
        assert all(len(v) == count and torch.isfinite(v).all() for v in tensors.values())
        roots = state['articulation']['robot']['root_pose'][:, 3:7]
        expected_root = torch.tensor(cfg.scene.robot.init_state.rot, dtype=roots.dtype)
        root_dot = (roots @ expected_root).abs() / (roots.norm(dim=1) * expected_root.norm())
        mount_error = float((1-root_dot).abs().max())
        assert mount_error < 1e-6, (family, 'Reset bank uses a different robot mounting', mount_error)
        flat = torch.cat([v.reshape(count, -1) for v in tensors.values()], dim=1)
        unique = len(torch.unique(flat, dim=0))
        assert unique == count, (family, unique, count)
        result = dict(count=count, unique=unique, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      robot_mount_quaternion_dot_error=mount_error,
                      self_blocked_rows=[], world_blocked_rows=[], edge_blocked_rows=[], minimum_cube_edge_clearance_m=1e6, maximum_reload_error=0.)
        report['families'][family] = result
        for start in range(0, count, env.num_envs):
            rows = np.arange(start, min(start+env.num_envs, count))
            padded = np.pad(rows, (0, env.num_envs-len(rows)), mode='edge')
            desired = take(state, padded)
            env.scene.reset_to(desired, is_relative=True)
            env.sim.forward()
            env.scene.update(0.)
            actual = dict(leaves(env.scene.get_state(is_relative=True)))
            err = max(float((actual[k][:len(rows)] - v[:len(rows)]).abs().max())
                      for k, v in leaves(desired))
            result['maximum_reload_error'] = max(result['maximum_reload_error'], err)
            sc, wc = geometry.evaluate(robot.data.body_link_pos_w, math_utils.matrix_from_quat(robot.data.body_link_quat_w), env.scene.env_origins)
            result['self_blocked_rows'] += rows[(~sc[:len(rows)]).cpu().numpy()].tolist()
            result['world_blocked_rows'] += rows[(~wc[:len(rows)]).cpu().numpy()].tolist()
            edge_clear = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            for name in ('insertive_object', 'receptive_object'):
                cube = env.scene[name]
                cp = cube.data.root_pos_w - env.scene.env_origins
                cq = cube.data.root_quat_w
                edge_clear &= atlas.cube_edge_clear(cp, cq)
                half = .03 * math_utils.matrix_from_quat(cq).abs().sum(dim=2)[:, :2]
                lo = torch.tensor(atlas.cfg['table_bounds_world_m']['min'][:2], device=env.device)
                hi = torch.tensor(atlas.cfg['table_bounds_world_m']['max'][:2], device=env.device)
                gap = torch.minimum(cp[:, :2] - half - lo, hi - cp[:, :2] - half)
                result['minimum_cube_edge_clearance_m'] = min(result['minimum_cube_edge_clearance_m'], float(gap[:len(rows)].min()))
            result['edge_blocked_rows'] += rows[(~edge_clear[:len(rows)]).cpu().numpy()].tolist()
        result['saved_geometry_valid'] = not (result['self_blocked_rows'] or result['world_blocked_rows'] or result['edge_blocked_rows'])
        assert result['maximum_reload_error'] < 5e-5, (family, result['maximum_reload_error'])
        print('RELOADED', family, json.dumps(result), flush=True)
        save()

        rng = np.random.default_rng(args.seed)
        rows = rng.choice(count, min(env.num_envs, count), replace=False)
        n = len(rows)
        padded = np.pad(rows, (0, env.num_envs-n), mode='edge')
        env.reset()
        env.scene.reset_to(take(state, padded), is_relative=True)
        env.sim.forward()
        env.scene.update(0.)
        ids = torch.arange(env.num_envs, device=env.device)
        env.action_manager.reset(ids)
        env.command_manager.reset(ids)
        env.reward_manager.reset(ids)
        env.termination_manager.reset(ids)
        env.episode_length_buf.zero_()
        initial_top = env.scene['insertive_object'].data.root_pos_w.clone()
        initial_q = robot.data.joint_pos.clone()
        initial_gap = support_gap(env).clone()
        finger = robot.joint_names.index('finger_joint')
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, -1] = torch.where(initial_q[:, finger] < .1, 1., -1.)
        if family in ('ObjectAnywhereEEGrasped', 'ObjectRestingEEGrasped'):
            actions[:, -1] = -1
        finite = True
        for _ in range(10):
            obs, reward, terminated, truncated, extra = env.step(actions)
            finite &= all(bool(torch.isfinite(v).all()) for _, v in leaves(env.scene.get_state(is_relative=True)))
        drift = (env.scene['insertive_object'].data.root_pos_w - initial_top).norm(dim=1)
        q_drift = (robot.data.joint_pos - initial_q).abs().amax(dim=1)
        gap = support_gap(env)
        result['hold_probe'] = dict(rows=rows.tolist(), all_finite=finite,
            cube_drift_m_quantiles=torch.quantile(drift[:n], torch.tensor([0., .5, .9, 1.], device=env.device)).cpu().tolist(),
            max_joint_change_rad_quantiles=torch.quantile(q_drift[:n], torch.tensor([0., .5, .9, 1.], device=env.device)).cpu().tolist(),
            initially_airborne=int((initial_gap[:n] > .01).sum()), still_airborne=int((gap[:n] > .01).sum()),
            cube_drift_over_1cm_rows=rows[(drift[:n] > .01).cpu().numpy()].tolist(),
            top_mass_kg=env.scene['insertive_object'].root_physx_view.get_masses()[:n].cpu().flatten().tolist())
        assert finite
        print('HOLD_PROBE', family, json.dumps(result['hold_probe']), flush=True)
        save()
    report['all_saved_states_valid'] = all(v['saved_geometry_valid'] for v in report['families'].values())
    save()
    assert report['all_saved_states_valid'], 'Saved-state geometry mismatch; inspect reported rows'
except Exception as exc:
    traceback.print_exc()
    report['validation_error'] = repr(exc)
    save()
    raise
finally:
    env.close()
    app.close()
