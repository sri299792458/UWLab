"""Closed-loop Thunder replay through UWLab's existing in-environment OSC.

Import after AppLauncher starts Isaac Sim. Candidate parameters are installed
after reset, because a delayed actuator resets its delay buffers and time lags.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import uwlab_tasks  # noqa: F401
from isaaclab.utils.math import subtract_frame_transforms
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_sim2real_cfg import ThunderSysidCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import settle_robot, target_pose_to_action

from records import JOINT_NAMES


def parameter_dict(params):
    p = np.asarray(params)
    return {key: p[start:start+6].tolist() for key, start in
            (("armature", 0), ("static_friction", 6), ("dynamic_ratio", 12), ("viscous_friction", 18))}


class ThunderReplay:
    def __init__(self, record, *, num_envs, device, delay_max, seed=42, env_cfg=None, render_mode=None):
        cfg = ThunderSysidCfg() if env_cfg is None else env_cfg
        cfg.seed = seed
        cfg.scene.num_envs = num_envs
        cfg.sim.device = device
        ctrl = record["osc_params"]
        cfg.actions.arm.motion_stiffness = tuple(ctrl["motion_stiffness"])
        cfg.actions.arm.motion_damping_ratio = tuple(ctrl["motion_damping_ratio"])
        cfg.actions.arm.torque_limit = tuple(ctrl["torque_max"])
        cfg.scene.robot.actuators["arm"].min_delay = 0
        cfg.scene.robot.actuators["arm"].max_delay = delay_max
        self.env = gym.make("OmniReset-Thunder-UMI-Sysid-v0", cfg=cfg, render_mode=render_mode).unwrapped
        self.env.reset()
        self.robot = self.env.scene["robot"]
        self.joint_ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError(f"Unexpected arm joint order: {names}")
        self.ee_id = self.robot.find_bodies("wrist_3_link")[0][0]
        self.n = num_envs
        self.device = self.env.device
        self.dt = cfg.sim.dt
        self.delay_max = delay_max
        if self.dt != record["dt"]:
            raise ValueError("Simulation and recording dt disagree")
        self.q0 = record["initial_joint_pos"].to(self.device).float()
        self.qd0 = record["initial_joint_vel"].to(self.device).float()
        self.targets_pos = record["waypoint_target_pos"].to(self.device).float()
        self.targets_quat = record["waypoint_target_quat"].to(self.device).float()
        self.real_q = record["joint_positions"].to(self.device).float()

    def prepare(self, params):
        params = torch.as_tensor(params, device=self.device, dtype=torch.float32)
        if params.shape != (self.n, 25) or not torch.isfinite(params).all():
            raise ValueError("Expected one finite 25-parameter candidate per environment")
        if (params < 0).any() or (params[:, 12:18] > 1).any():
            raise ValueError("Invalid dynamics candidate")
        delays = torch.round(params[:, 24]).to(torch.int)
        if (delays > self.delay_max).any():
            raise ValueError("Candidate delay exceeds actuator buffer capacity")
        self.env.reset()
        q = self.robot.data.default_joint_pos.clone()
        qd = torch.zeros_like(q)
        q[:, self.joint_ids] = self.q0
        qd[:, self.joint_ids] = self.qd0
        self.robot.set_joint_effort_target(torch.zeros_like(q))
        settle_robot(self.robot, self.env.sim, q, qd, self.joint_ids, self.dt, headless=True)
        self.env.action_manager.reset()
        # Clear the history AFTER settling and BEFORE assigning candidate delays.
        ids = torch.arange(self.n, device=self.device)
        actuator = self.robot.actuators["arm"]
        actuator.reset(ids)
        self.robot.write_joint_armature_to_sim(params[:, :6], joint_ids=self.joint_ids, env_ids=ids)
        self.robot.write_joint_friction_coefficient_to_sim(
            params[:, 6:12], joint_dynamic_friction_coeff=params[:, 6:12] * params[:, 12:18],
            joint_viscous_friction_coeff=params[:, 18:24], joint_ids=self.joint_ids, env_ids=ids,
        )
        for buffer in (actuator.positions_delay_buffer, actuator.velocities_delay_buffer, actuator.efforts_delay_buffer):
            buffer.set_time_lag(delays)
        return delays

    def run(self, params, *, trajectory=False, max_steps=None, capture_bodies=False, progress_every=0):
        delays = self.prepare(params)
        count = len(self.real_q) if max_steps is None else min(max_steps, len(self.real_q))
        sum_sq = torch.zeros(self.n, 6, device=self.device)
        legacy_sum_sq = torch.zeros_like(sum_sq)
        before_rows, after_rows, torque_rows = [], [], []
        body_rows, full_joint_rows, velocity_rows = [], [], []
        for t in range(count):
            # record[t] is the robot state before command[t]. Compare the state
            # at the same phase, then apply that command and advance by dt.
            before = self.robot.data.joint_pos[:, self.joint_ids].clone()
            if capture_bodies:
                body_rows.append(self.robot.data.body_pose_w.cpu().numpy().copy())
                full_joint_rows.append(self.robot.data.joint_pos.cpu().numpy().copy())
                velocity_rows.append(self.robot.data.joint_vel[:, self.joint_ids].cpu().numpy().copy())
            sum_sq += (before - self.real_q[t]) ** 2
            pos, quat = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w,
                self.robot.data.body_pos_w[:, self.ee_id], self.robot.data.body_quat_w[:, self.ee_id],
            )
            arm = target_pose_to_action(pos, quat, self.targets_pos[t].expand(self.n, -1),
                                        self.targets_quat[t].expand(self.n, -1))
            # Positive binary command holds the physical open-gripper condition.
            action = torch.cat([arm, torch.ones(self.n, 1, device=self.device)], dim=-1)
            self.env.step(action)
            after = self.robot.data.joint_pos[:, self.joint_ids]
            legacy_sum_sq += (after - self.real_q[t]) ** 2
            if trajectory:
                before_rows.append(before.cpu().numpy())
                after_rows.append(after.cpu().numpy().copy())
                torque_rows.append(self.robot.data.computed_torque[:, self.joint_ids].cpu().numpy().copy())
            if progress_every and (t+1) % progress_every == 0:
                print(f"Replayed {t+1}/{count} steps", flush=True)
        result = {"scores": (sum_sq.sum(dim=1) / count).cpu().numpy(),
                  "per_joint_rmse_rad": torch.sqrt(sum_sq / count).cpu().numpy(),
                  "upstream_post_step_scores": (legacy_sum_sq.sum(dim=1) / count).cpu().numpy(),
                  "delay_steps": delays.cpu().tolist(), "steps": count}
        if trajectory:
            result.update(joint_positions=np.array(before_rows), post_step_joint_positions=np.array(after_rows),
                          computed_torques=np.array(torque_rows))
        if capture_bodies:
            body_rows.append(self.robot.data.body_pose_w.cpu().numpy().copy())
            full_joint_rows.append(self.robot.data.joint_pos.cpu().numpy().copy())
            velocity_rows.append(self.robot.data.joint_vel[:, self.joint_ids].cpu().numpy().copy())
            result.update(body_poses=np.array(body_rows), full_joint_positions=np.array(full_joint_rows),
                          joint_velocities=np.array(velocity_rows), body_names=self.robot.body_names,
                          joint_names=self.robot.joint_names, env_origins=self.env.scene.env_origins.cpu().numpy())
        return result

    def close(self):
        self.env.close()
