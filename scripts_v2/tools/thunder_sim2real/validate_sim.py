"""Native simulation checks. All generated data is labeled synthetic_test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch
from isaaclab.app import AppLauncher

from records import HERE, profile_api, validate_record
from test_contract import synthetic_record, synthetic_profile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["replay", "stage2", "stage2eval"], required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app


def replay_check():
    from replay import ThunderReplay
    sys.path.insert(0, str(HERE / "workstation"))
    import collect_thunder
    collect_thunder.install_calibration()
    nsteps = 400
    record = synthetic_record(nsteps)
    q0 = record["initial_joint_pos"].numpy()
    pos, quat = collect_thunder.kin.get_ee_pose(q0)
    t = np.arange(nsteps) * 0.002
    offset = np.stack([0.02*np.sin(2*np.pi*1.1*t), 0.015*np.sin(2*np.pi*1.7*t),
                       0.01*np.sin(2*np.pi*0.7*t)], axis=1)
    record["waypoint_target_pos"] = torch.tensor(pos + offset)
    record["waypoint_target_quat"] = torch.tensor(quat).repeat(nsteps, 1)
    validate_record(record, allow_synthetic=True)
    replay = ThunderReplay(record, num_envs=6, device=args.device, delay_max=5)
    params = np.array([0.2]*6 + [0.5]*6 + [0.8]*6 + [0.4]*6 + [2.0])
    try:
        truth = replay.run(np.tile(params, (6, 1)), trajectory=True)
        record["joint_positions"] = torch.tensor(truth["joint_positions"][:, 0], dtype=torch.float64)
        record["joint_torques"] = torch.tensor(truth["computed_torques"][:, 0], dtype=torch.float64)
        replay.real_q = record["joint_positions"].to(replay.device).float()
        candidates = np.tile(params, (6, 1)); candidates[:, 24] = np.arange(6)
        result = replay.run(candidates, trajectory=True)
        errors = np.degrees(np.sqrt(result["scores"] / 6))
        legacy = np.degrees(np.sqrt(result["upstream_post_step_scores"] / 6))
        best = int(np.argmin(errors))
        if best != 2 or errors[2] > 0.01:
            raise AssertionError(f"Known-delay recovery failed: {errors}")
        if not legacy[2] > max(5*errors[2], 1e-4):
            raise AssertionError(f"Expected post-step comparison to expose the indexing offset: {errors}, {legacy}")
        # Compare the simulator's wrist frame and analytical Jacobian against the
        # independent NumPy implementation shipped to the robot workstation.
        replay.prepare(np.tile(params, (6, 1)))
        from isaaclab.utils.math import subtract_frame_transforms
        actual_pos, actual_quat = subtract_frame_transforms(
            replay.robot.data.root_pos_w, replay.robot.data.root_quat_w,
            replay.robot.data.body_pos_w[:, replay.ee_id], replay.robot.data.body_quat_w[:, replay.ee_id])
        pos_error = float(np.max(np.abs(actual_pos[0].cpu().numpy() - pos)))
        quat_dot = float(abs(np.dot(actual_quat[0].cpu().numpy(), quat)))
        from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical
        jac = compute_jacobian_analytical(torch.tensor(q0, device=replay.device).float()[None],
                                          device=str(replay.device), usd_path=replay.robot.cfg.spawn.usd_path)
        jac_error = float(np.max(np.abs(jac[0].cpu().numpy() - collect_thunder.kin.compute_jacobian_calibrated(q0))))
        if pos_error > 1e-5 or abs(1-quat_dot) > 1e-5 or jac_error > 1e-5:
            raise AssertionError(f"Workstation/simulation kinematics differ: {pos_error}, {quat_dot}, {jac_error}")
        report = {"status": "PASS", "source_kind": "synthetic_test", "known_delay_steps": 2,
                  "recovered_delay_steps": best, "pre_command_rmse_deg_by_delay": errors.tolist(),
                  "upstream_post_step_rmse_deg_by_delay": legacy.tolist(),
                  "wrist_position_max_error_m": pos_error, "wrist_quaternion_dot": quat_dot,
                  "jacobian_max_error": jac_error, "steps": nsteps,
                  "scope": "Indexing and known-delay recovery with other dynamics fixed; not real-robot fit validation"}
        torch.save(record, args.output / "synthetic_record.pt")
        np.savez_compressed(args.output / "synthetic_replay.npz", truth=record["joint_positions"].numpy(),
                            replay=result["joint_positions"], delays=np.arange(6), dt=0.002)
        return report
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        replay.close()


def stage2_check():
    import gymnasium as gym
    import isaaclab
    import uwlab_tasks
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import umi_sim2real_cfg as cfgmod
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg
    profile = synthetic_profile()
    profile_api().validate_profile(profile, allow_synthetic=True)
    # Test-only injection. Production load_profile rejects synthetic profiles.
    with patch.object(cfgmod, "load_profile", return_value=(profile, "synthetic-test-only")):
        cfg = cfgmod.UmiCubeFinetuneCfg()
        evaluation = cfgmod.UmiCubeFinetuneEvalCfg()
    baseline = UmiCubeTrainCfg()
    assert cfg.sim.dt == evaluation.sim.dt == baseline.sim.dt == 1/120
    assert cfg.decimation == evaluation.decimation == baseline.decimation == 12
    assert cfg.scene.robot.spawn.usd_path == evaluation.scene.robot.spawn.usd_path == baseline.scene.robot.spawn.usd_path
    assert cfg.scene.insertive_object.spawn.usd_path == baseline.scene.insertive_object.spawn.usd_path
    assert cfg.scene.receptive_object.spawn.usd_path == baseline.scene.receptive_object.spawn.usd_path
    assert cfg.scene.receptive_object.spawn.rigid_props.kinematic_enabled
    assert cfg.actions.arm.motion_stiffness == baseline.actions.arm.motion_stiffness
    assert tuple(evaluation.actions.arm.motion_stiffness) == tuple(profile["controller"]["motion_stiffness"])
    assert tuple(evaluation.actions.arm.scale_xyz_axisangle) == tuple(cfg.curriculum.action_scale.params["target_scales"])
    assert cfg.events.reset_from_reset_states.params["dataset_dir"] == baseline.events.reset_from_reset_states.params["dataset_dir"]
    assert cfg.events.reset_from_reset_states.params["probs"] == baseline.events.reset_from_reset_states.params["probs"]
    assert cfg.rewards.to_dict() == baseline.rewards.to_dict()
    assert cfg.observations.to_dict() == baseline.observations.to_dict()
    is_eval = args.mode == "stage2eval"
    if is_eval:
        cfg = evaluation
    cfg.scene.num_envs = 64; cfg.sim.device = args.device; cfg.seed = 42
    task = "OmniReset-UMI-Defaults-State-Finetune-Play-v0" if is_eval else "OmniReset-UMI-Defaults-State-Finetune-v0"
    env = gym.make(task, cfg=cfg).unwrapped
    phases = []
    try:
        env.reset()
        print("Stage-2 environment reset completed", flush=True)
        sysid = env.event_manager.get_term_cfg("randomize_arm_sysid").func
        gains = env.event_manager.get_term_cfg("randomize_osc_gains").func
        for progress in ((1.0,) if is_eval else (0.0, 0.5, 1.0)):
            print(f"Checking Stage-2 progress={progress}", flush=True)
            env.reset()
            sysid.scale_progress = gains.scale_progress = progress
            for term_name in ("randomize_arm_sysid", "randomize_osc_gains"):
                term = env.event_manager.get_term_cfg(term_name)
                term.func(env, env_ids=torch.arange(64, device=env.device), **term.params)
            arm = env.action_manager.get_term("arm")
            kp_now, kd_now = arm._kp.clone(), arm._kd.clone()
            if progress == 0:
                torch.testing.assert_close(kp_now, torch.tensor(baseline.actions.arm.motion_stiffness, device=env.device).expand(64, -1))
            if progress == 1:
                terminal = torch.tensor(profile["controller"]["motion_stiffness"], device=env.device)
                assert ((kp_now >= 0.8*terminal) & (kp_now <= 1.2*terminal)).all()
            for _ in range(8):
                observations, reward, terminated, truncated, _ = env.step(torch.zeros(64, 7, device=env.device))
                assert torch.isfinite(reward).all()
                for value in observations.values():
                    assert torch.isfinite(value).all()
            assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
            phases.append({"progress": progress, "mean_reward": float(reward.mean()),
                           "mean_kp": kp_now.mean(dim=0).cpu().tolist(),
                           "mean_kd": kd_now.mean(dim=0).cpu().tolist()})
        dataset = Path(cfg.events.reset_from_reset_states.params["dataset_dir"])
        families = cfg.events.reset_from_reset_states.params["reset_types"]
        resetter = env.event_manager.get_term_cfg("reset_from_reset_states").func
        bank_paths = {family: dataset / 'Resets/InsertiveAprilCube60__ReceptiveAprilCube60' / f'resets_{family}.pt'
                      for family in families}
        return {"status": "PASS", "source_kind": "synthetic_test", "num_envs": 64,
                "task": task,
                "isaaclab_module_path": isaaclab.__file__,
                "task_module_path": uwlab_tasks.__file__,
                "robot_usd": cfg.scene.robot.spawn.usd_path,
                "table_usd": cfg.scene.table.spawn.usd_path,
                "robot_mount_position_world_m": list(cfg.scene.robot.init_state.pos),
                "robot_mount_quaternion_world_wxyz": list(cfg.scene.robot.init_state.rot),
                "dataset_dir": str(dataset),
                "dataset_counts": dict(zip(families, resetter.num_states.cpu().tolist())),
                "dataset_sha256": {family: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for family, path in bank_paths.items()},
                "physics_hz": 120, "policy_hz": 10, "phases": phases,
                "preserved": ["Thunder/UMI assets", "cube motion modes", "reset bank and probabilities",
                              "rewards", "observations", "initial gains"],
                "scope": "Configuration and native reset/step validation; no real fitted parameters or learned transfer claim"}
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        env.close()


try:
    result = replay_check() if args.mode == "replay" else stage2_check()
    (args.output / f"{args.mode}_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
except BaseException as exc:
    traceback.print_exc()
    (args.output / f"{args.mode}_failure.json").write_text(json.dumps({"status": "FAIL", "error": str(exc)}) + "\n")
    raise
finally:
    app.close()
