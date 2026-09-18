"""Measure loaded table/air grasps with a fixed controller sequence; never filter."""
import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab.utils.math as mu
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg

cfg = UmiCubeTrainCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 181
cfg.events.reset_from_reset_states.params['dataset_dir'] = str(args.dataset_dir)
# Observe the same saved start throughout the probe, including poor outcomes.
cfg.terminations.abnormal_robot = None
cfg.episode_length_s = 60
env = gym.make('OmniReset-UMI-Defaults-State-Train-v0', cfg=cfg).unwrapped
robot, top = env.scene['robot'], env.scene['insertive_object']
base_id = robot.body_names.index('robotiq_base_link')
finger_id = robot.joint_names.index('finger_joint')
resetter = env.event_manager.get_term_cfg('reset_from_reset_states').func
families = cfg.events.reset_from_reset_states.params['reset_types']
report = dict(dataset_dir=str(args.dataset_dir), environments=env.num_envs,
    sequence='Close and hold for 1 s; command +10 mm world-Z target increment per policy step for 4 s; hold 1 s; open for 1 s.',
    scope='Controller probe with existing training mass/material randomization; no policy, bank filtering, or new reset criterion. '
          'Abnormal-robot termination is disabled and the time limit extended only in this probe to avoid replacing a tested state mid-sequence.',
    measurements='Ever lifted: cube rises at least 30 mm above its loaded height. '
                 'Lifted with hand: cube and hand each rise 30 mm, and their relative translation changes by less than 30 mm from the end of the initial hold. '
                 'These are diagnostic measurements, not a contact or trajectory certificate.',
    families={})
args.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


try:
    for family in ('ObjectRestingEEGrasped', 'ObjectAnywhereEEGrasped'):
        index = families.index(family)
        torch.manual_seed(181 + index)
        resetter.probs.zero_()
        resetter.probs[index] = 1
        env.reset()
        assert (resetter.task_id == index).all()
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, -1] = -1
        initial_top = top.data.root_pos_w.clone()
        initial_hand = robot.data.body_pos_w[:, base_id].clone()
        initial_finger = robot.data.joint_pos[:, finger_id].clone()
        maximum_cube_rise = torch.zeros(env.num_envs, device=env.device)
        maximum_hand_rise = torch.zeros_like(maximum_cube_rise)
        lifted_with_hand = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        finite = torch.ones_like(lifted_with_hand)
        close_relative = None
        for step in range(60):
            actions[:, :3] = 0
            if 10 <= step < 50:
                up = torch.zeros((env.num_envs, 3), device=env.device)
                up[:, 2] = .01
                actions[:, :3] = mu.quat_apply_inverse(robot.data.root_quat_w, up) / torch.tensor(
                    cfg.actions.arm.scale_xyz_axisangle[:3], device=env.device)
            env.step(actions)
            cube_rise = top.data.root_pos_w[:, 2] - initial_top[:, 2]
            hand_rise = robot.data.body_pos_w[:, base_id, 2] - initial_hand[:, 2]
            relative = top.data.root_pos_w - robot.data.body_pos_w[:, base_id]
            finite &= torch.isfinite(robot.data.joint_pos).all(dim=1) & torch.isfinite(top.data.root_pose_w).all(dim=1)
            maximum_cube_rise = torch.maximum(maximum_cube_rise, cube_rise)
            maximum_hand_rise = torch.maximum(maximum_hand_rise, hand_rise)
            if step == 9:
                close_relative = relative.clone()
            if step >= 10:
                lifted_with_hand |= (cube_rise >= .03) & (hand_rise >= .03) & ((relative - close_relative).norm(dim=1) < .03)
        height_before_opening = top.data.root_pos_w[:, 2].clone()
        actions.zero_()
        actions[:, -1] = 1
        for _ in range(10):
            env.step(actions)
            finite &= torch.isfinite(robot.data.joint_pos).all(dim=1) & torch.isfinite(top.data.root_pose_w).all(dim=1)
        drop_after_opening = height_before_opening - top.data.root_pos_w[:, 2]
        path = args.dataset_dir / 'Resets/InsertiveAprilCube60__ReceptiveAprilCube60' / ('resets_' + family + '.pt')
        result = dict(seed=181 + index, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            finite_environments=int(finite.sum()), initially_fully_closed=int((initial_finger > .75).sum()),
            cube_ever_rose_30mm=int(((maximum_cube_rise >= .03) & finite).sum()),
            hand_ever_rose_30mm=int(((maximum_hand_rise >= .03) & finite).sum()),
            lifted_with_hand=int((lifted_with_hand & finite).sum()),
            rose_and_dropped_after_opening=int((lifted_with_hand & (drop_after_opening >= .01) & finite).sum()),
            median_maximum_cube_rise_m=float(maximum_cube_rise[finite].median()),
            median_maximum_hand_rise_m=float(maximum_hand_rise[finite].median()),
            top_mass_kg=top.root_physx_view.get_masses().cpu().flatten().tolist())
        report['families'][family] = result
        save()
        print('GRASP_PROBE ' + family + ' ' + json.dumps({k: v for k, v in result.items() if k != 'top_mass_kg'}), flush=True)
    report['complete'] = True
    save()
finally:
    env.close()
    app.close()
