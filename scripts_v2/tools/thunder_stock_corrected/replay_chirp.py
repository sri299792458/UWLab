"""Replay the recorded Thunder chirp with two fixed gains at one chosen rate.

Reuses the existing ThunderReplay reset, native parameter installation, target
conversion and pre-command comparison conventions. No fitting is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=Path, required=True)
parser.add_argument("--fit", type=Path, required=True)
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--hz", type=int, choices=(120, 500), required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args, multi_gpu=False).app

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import uwlab_tasks
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import subtract_frame_transforms
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.sysid_cfg import SysidEnvCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import settle_robot, target_pose_to_action

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    record = torch.load(args.record, weights_only=True, map_location="cpu")
    fit = json.loads(args.fit.read_text())
    assert record["completed"] and record["sample_phase"] == "pre_command"
    assert record["joint_names"] == JOINTS and record["dt"] == .002
    assert fit["record_sha256"] == sha(args.record)
    source_t = np.arange(len(record["joint_positions"])) * record["dt"]
    duration = len(source_t) * record["dt"]
    times = np.arange(round(duration * args.hz)) / args.hz
    def interp(key):
        values = record[key].numpy()
        return np.column_stack([np.interp(times, source_t, values[:, i]) for i in range(values.shape[1])])
    pos_targets = interp("waypoint_target_pos")
    source_quat = record["waypoint_target_quat"].numpy()
    rotation = Rotation.from_quat(source_quat[:, [1, 2, 3, 0]])
    quat_targets = Slerp(source_t, rotation)(times).as_quat()[:, [3, 0, 1, 2]]
    real_q, real_v = interp("joint_positions"), interp("joint_velocities")
    delay_s = fit["delay_steps"] * record["dt"]
    delay_steps = int(round(delay_s * args.hz))
    cfg = SysidEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.sim.dt = 1 / args.hz
    cfg.decimation = 1
    cfg.sim.render_interval = 100000
    cfg.scene.ground = None
    cfg.scene.robot.spawn.usd_path = str(args.asset.resolve())
    cfg.scene.robot.spawn.mass_props = None
    # The existing identification replay used an inactive high speed cap, so
    # clipping cannot conceal a divergent response in this comparison either.
    cfg.scene.robot.actuators["arm"].velocity_limit = 1000.
    cfg.scene.robot.actuators["arm"].velocity_limit_sim = 1000.
    cfg.scene.robot.actuators["arm"].min_delay = 0
    cfg.scene.robot.actuators["arm"].max_delay = delay_steps
    env = gym.make("OmniReset-Ur5eRobotiq2f85-Sysid-v0", cfg=cfg).unwrapped
    dump_yaml(str(args.output / "env.yaml"), cfg)
    try:
        env.reset(seed=42)
        robot = env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS
        arm = env.action_manager.get_term("arm")
        kp = torch.tensor([[200., 200., 200., 3., 3., 3.], record["osc_params"]["motion_stiffness"]], device=env.device)
        ratio = torch.tensor([[3., 3., 3., 1., 1., 1.], record["osc_params"]["motion_damping_ratio"]], device=env.device)
        arm._kp[:] = kp
        arm._kd[:] = 2 * torch.sqrt(kp) * ratio
        q0 = robot.data.default_joint_pos.clone()
        v0 = torch.zeros_like(q0)
        q0[:, ids] = record["initial_joint_pos"].to(device=env.device, dtype=q0.dtype)
        v0[:, ids] = record["initial_joint_vel"].to(device=env.device, dtype=v0.dtype)
        robot.set_joint_effort_target(torch.zeros_like(q0))
        settle_robot(robot, env.sim, q0, v0, ids, cfg.sim.dt, headless=True)
        env.action_manager.reset()
        actuator = robot.actuators["arm"]
        env_ids = torch.arange(2, device=env.device)
        actuator.reset(env_ids)
        params = torch.tensor(fit["best_params"], dtype=torch.float32, device=env.device)[None].repeat(2, 1)
        robot.write_joint_armature_to_sim(params[:, :6], joint_ids=ids, env_ids=env_ids)
        robot.write_joint_friction_coefficient_to_sim(
            params[:, 6:12], joint_dynamic_friction_coeff=params[:, 6:12] * params[:, 12:18],
            joint_viscous_friction_coeff=params[:, 18:24], joint_ids=ids, env_ids=env_ids)
        delays = torch.full((2,), delay_steps, dtype=torch.int, device=env.device)
        for buffer in (actuator.positions_delay_buffer, actuator.velocities_delay_buffer, actuator.efforts_delay_buffer):
            buffer.set_time_lag(delays)
        target_pos = torch.tensor(pos_targets, dtype=torch.float32, device=env.device)
        target_quat = torch.tensor(quat_targets, dtype=torch.float32, device=env.device)
        initial = {"joint_position":robot.data.joint_pos.cpu().numpy().copy(),
                   "joint_velocity":robot.data.joint_vel.cpu().numpy().copy(),
                   "masses":robot.root_physx_view.get_masses().cpu().numpy(),
                   "inertias":robot.root_physx_view.get_inertias().cpu().numpy(),
                   "coms":robot.root_physx_view.get_coms().cpu().numpy(),
                   "armature":robot.root_physx_view.get_dof_armatures().cpu().numpy(),
                   "static_friction":robot.data.joint_friction_coeff.cpu().numpy().copy(),
                   "dynamic_friction":robot.data.joint_dynamic_friction_coeff.cpu().numpy().copy(),
                   "viscous_friction":robot.data.joint_viscous_friction_coeff.cpu().numpy().copy()}
        np.savez_compressed(args.output / "initial.npz", **initial)
        rows_q, rows_v, rows_tau, rows_applied, rows_ee, rows_target_error = [], [], [], [], [], []
        started = time.monotonic()
        for step in range(len(times)):
            rows_q.append(robot.data.joint_pos[:, ids].cpu().numpy().copy())
            rows_v.append(robot.data.joint_vel[:, ids].cpu().numpy().copy())
            position, orientation = arm._get_ee_pose_root_frame()
            rows_ee.append(torch.cat([position, orientation], dim=1).cpu().numpy().copy())
            delta = target_pose_to_action(position, orientation, target_pos[step].expand(2, -1), target_quat[step].expand(2, -1))
            rows_target_error.append(delta.cpu().numpy().copy())
            action = torch.cat([delta, torch.ones((2, 1), device=env.device)], dim=1)
            _, _, done, timeout, _ = env.step(action)
            assert not (done | timeout).any()
            assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all()
            rows_tau.append(robot.data.joint_effort_target[:, ids].cpu().numpy().copy())
            rows_applied.append(robot.data.applied_torque[:, ids].cpu().numpy().copy())
            if (step+1) % args.hz == 0:
                print("REPLAY_SECONDS", (step+1)/args.hz, flush=True)
        arrays = dict(t=times, target_pos=pos_targets, target_quat=quat_targets, real_q=real_q, real_v=real_v,
                      q=np.asarray(rows_q), v=np.asarray(rows_v), requested_torque=np.asarray(rows_tau),
                      applied_torque=np.asarray(rows_applied), ee=np.asarray(rows_ee), target_error=np.asarray(rows_target_error))
        np.savez_compressed(args.output / "trace.npz", **arrays)
        reports = []
        for i, name in enumerate(("uwlab", "thunder")):
            q_error, v_error = arrays["q"][:, i] - real_q, arrays["v"][:, i] - real_v
            reports.append({"gains":name, "kp":arm._kp[i].tolist(), "kd":arm._kd[i].tolist(),
                            "position_rmse_vs_hardware_deg":float(np.rad2deg(np.sqrt(np.mean(q_error**2)))),
                            "velocity_rmse_vs_hardware_deg_s":float(np.rad2deg(np.sqrt(np.mean(v_error**2)))),
                            "position_tracking_rms_mm":float(np.sqrt(np.mean(np.sum(arrays["target_error"][:, i, :3]**2, axis=1)))*1000),
                            "rotation_tracking_rms_deg":float(np.rad2deg(np.sqrt(np.mean(np.sum(arrays["target_error"][:, i, 3:]**2, axis=1))))),
                            "peak_joint_speed_deg_s":np.rad2deg(np.abs(arrays["v"][:, i]).max(axis=0)).tolist(),
                            "torque_saturation_fraction":float(np.mean(np.abs(arrays["requested_torque"][:, i]) >= np.asarray(record["osc_params"]["torque_max"])*.999))})
        report = {"status":"complete", "hz":args.hz, "steps":len(times), "duration_s":duration,
                  "record":str(args.record), "record_sha256":sha(args.record), "fit":str(args.fit), "fit_sha256":sha(args.fit),
                  "asset":str(args.asset), "asset_sha256":sha(args.asset), "fitted_delay_s":delay_s,
                  "implemented_delay_steps":delay_steps, "implemented_delay_s":delay_steps/args.hz,
                  "physical_recording_hand":"UMI; simulation is corrected stock", "rows":reports,
                  "wall_seconds":time.monotonic()-started, "joint_names":names,
                  "timing":"Same nominal command-index alignment as the existing fit. Linear target-position interpolation and quaternion SLERP at each rate; raw recording is unchanged. Targets update every servo step for this identification chirp, not at the 10 Hz policy rate.",
                  "scope":"Two fixed gain candidates, no refitting, no PPO, no contact task. Hardware agreement is descriptive: the fit used this recording, and the hardware hand differs. Rate comparison includes delay quantization."}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("CHIRP_COMPLETE", json.dumps(report), flush=True)
    finally:
        env.close()


try:
    with torch.inference_mode():
        main()
except BaseException as exc:
    (args.output / "failure.json").write_text(json.dumps({"error":repr(exc)}, indent=2) + "\n")
    raise
finally:
    app.close()
