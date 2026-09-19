"""Replay the collection waveform for atlas poses in the complete static lab."""
import os
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from isaaclab.app import AppLauncher

from records import HERE
from test_contract import synthetic_record

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--amplitude_scale", type=float, default=1.0)
parser.add_argument("--f0_hz", type=float, default=.1)
parser.add_argument("--f1_hz", type=float, default=3.)
parser.add_argument("--velocity_limit_mode", choices=("diagnostic", "asset"), default="diagnostic",
                    help="Diagnostic uses a loose simulation-only cap and reports any cap activity")
parser.add_argument("--diagnostic_velocity_limit_rad_s", type=float, default=1000.)
parser.add_argument("--review_speed_deg_s", type=float, default=120.,
                    help="Offline screening target, not a hardware safety setting")
parser.add_argument("--dynamics_grid", action="store_true",
                    help="Test 50/100/120 percent reference dynamics at 0/4/10 ms delay")
parser.add_argument("--dynamics_variants", type=Path,
                    help="JSON rows with variant and params, replacing the assumed dynamics grid")
parser.add_argument("--stationary_baseline", action="store_true",
                    help="Also test zero added dynamics with a stationary target")
parser.add_argument("--stationary_reference_baselines", action="store_true",
                    help="Also test each nonzero reference dynamics variant with a stationary target")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not np.isfinite([args.amplitude_scale, args.f0_hz, args.f1_hz,
                    args.diagnostic_velocity_limit_rad_s, args.review_speed_deg_s]).all():
    parser.error("Motion scale and speed values must be finite")
if args.amplitude_scale <= 0 or args.diagnostic_velocity_limit_rad_s <= 0 or args.review_speed_deg_s <= 0:
    parser.error("Motion scale and speed values must be positive")
if not 0 < args.f0_hz <= args.f1_hz < 250:
    parser.error("Use 0 < f0_hz <= f1_hz < 250")
if args.velocity_limit_mode == "diagnostic" and args.diagnostic_velocity_limit_rad_s <= np.radians(191.):
    parser.error("The diagnostic simulation cap must exceed the displayed hardware setting")
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

from isaaclab.utils.math import matrix_from_quat
from replay import ThunderReplay
from motion_geometry import UmiSphereGeometry, clearances
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_sim2real_cfg import ThunderSysidCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg
import yaml

sys.path.insert(0, str(HERE / "workstation"))
import collect_thunder


def lab_config():
    cfg = ThunderSysidCfg()
    if args.velocity_limit_mode == "diagnostic":
        # This applies only to this preview instance. Do not change the shared
        # robot asset, training configuration, or physical robot settings.
        cfg.scene.robot.actuators["arm"].velocity_limit = args.diagnostic_velocity_limit_rad_s
        cfg.scene.robot.actuators["arm"].velocity_limit_sim = args.diagnostic_velocity_limit_rad_s
    lab = UmiCubeTrainCfg()
    cfg.scene.robot.init_state.pos = lab.scene.robot.init_state.pos
    cfg.scene.robot.init_state.rot = lab.scene.robot.init_state.rot
    cfg.scene.table = copy.deepcopy(lab.scene.table)
    cfg.scene.ground = copy.deepcopy(lab.scene.ground)
    cfg.scene.env_spacing = 3.5
    return cfg


def main():
    candidates = json.loads(args.candidates.read_text())["candidates"]
    motion = {"duration_s": 8., "f0_hz": args.f0_hz, "f1_hz": args.f1_hz,
              "amplitudes_m_rad": (args.amplitude_scale*np.array([.1, .1, .1*1.5, .5, .25, .5])).tolist()}
    offsets = collect_thunder.generate_offsets(motion)
    collect_thunder.install_calibration()
    record = synthetic_record(len(offsets))
    metadata = yaml.safe_load(Path(os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets') + '/thunder_d405_umi_rigid_asset/metadata.yaml').read_text())
    nominal = metadata["sysid"]
    ref = np.array(sum([nominal[k] for k in ("armature", "static_friction", "dynamic_ratio", "viscous_friction")], []))
    variants = [{"name": "zero_added_friction_inertia_delay", "params": np.zeros(25).tolist()},
                {"name": "UW_reference_dynamics_delay_4ms", "params": np.r_[ref, 2].tolist()},
                {"name": "UW_reference_120percent_delay_10ms", "params": np.r_[ref, 5].tolist()}]
    for section in (slice(0, 12), slice(18, 24)):
        values = np.array(variants[2]["params"]); values[section] *= 1.2; variants[2]["params"] = values.tolist()
    if args.dynamics_grid:
        variants = variants[:1]
        for scale in (.5, 1., 1.2):
            values = ref.copy()
            for section in (slice(0, 12), slice(18, 24)):
                values[section] *= scale
            for delay in (0, 2, 5):
                variants.append({"name": f"UW_reference_{scale:g}x_delay_{delay*2}ms",
                                 "params": np.r_[values, delay].tolist()})
    if args.dynamics_variants:
        if args.dynamics_grid:
            raise ValueError("Choose dynamics_variants or dynamics_grid")
        fitted = json.loads(args.dynamics_variants.read_text())
        variants = [{"name": row["variant"], "params": row["params"]} for row in fitted["rows"]]
        for variant in variants:
            params = np.asarray(variant["params"])
            if params.shape != (25,) or not np.isfinite(params).all():
                raise ValueError("Expected 25 finite parameters for each fitted variant")
        if not variants:
            raise ValueError("No dynamics variants provided")
    if args.stationary_reference_baselines:
        variants += [{"name": "stationary_" + v["name"], "params": v["params"], "stationary": True}
                     for v in variants if v["name"].startswith("UW_reference")]
    if args.stationary_baseline:
        variants.append({"name": "stationary_zero_added_friction_inertia_delay",
                         "params": np.zeros(25).tolist(), "stationary": True})
    rows = [{"candidate": c["candidate"], "variant": v["name"], "q0": c["joint_positions_rad"],
             "params": v["params"], "stationary": v.get("stationary", False)}
            for c in candidates for v in variants]
    count = len(rows)
    replay = ThunderReplay(record, num_envs=count, device=args.device, delay_max=5, env_cfg=lab_config())
    try:
        physics_limits = replay.robot.root_physx_view.get_dof_max_velocities()[:, replay.joint_ids].cpu().numpy()
        if args.velocity_limit_mode == "diagnostic" and not np.allclose(physics_limits, args.diagnostic_velocity_limit_rad_s):
            raise RuntimeError("PhysX did not receive the requested diagnostic speed limits")
        replay.q0 = torch.tensor([row["q0"] for row in rows], device=replay.device, dtype=torch.float32)
        replay.qd0 = torch.zeros_like(replay.q0)
        positions, quats = [], []
        for row in rows:
            p0, r0 = collect_thunder.kin.get_ee_pose(np.array(row["q0"]))
            row_offsets = np.zeros_like(offsets) if row["stationary"] else offsets
            positions.append(p0+row_offsets[:, :3])
            quats.append(np.array([collect_thunder.quat_multiply(collect_thunder.kin.axis_angle_to_quat(o[3:]), r0) for o in row_offsets]))
        replay.targets_pos = torch.tensor(np.stack(positions, axis=1), device=replay.device, dtype=torch.float32)
        replay.targets_quat = torch.tensor(np.stack(quats, axis=1), device=replay.device, dtype=torch.float32)
        result = replay.run(np.array([row["params"] for row in rows]), trajectory=True, capture_bodies=True, progress_every=500)
        np.savez_compressed(args.output / "trajectories.npz", **{key: result[key] for key in
                            ("body_poses", "full_joint_positions", "joint_velocities", "computed_torques", "env_origins")},
                            targets_pos=replay.targets_pos.cpu().numpy(), targets_quat=replay.targets_quat.cpu().numpy())
        adapter = UmiSphereGeometry(result["body_names"], replay.device)
        all_metrics = {key: [] for key in ("self_margin_m", "world_margin_m", "world_gap_m")}
        # Evaluate every 2 ms recorded state. Batch the geometry computation so
        # simulation stepping and geometry verification use the same trajectory.
        for start in range(0, len(result["body_poses"]), 25):
            poses = torch.tensor(result["body_poses"][start:start+25], device=replay.device)
            flat = poses.reshape(-1, poses.shape[-2], 7)
            origins = torch.tensor(np.tile(result["env_origins"], (len(poses), 1)), device=replay.device)
            metrics = clearances(adapter, flat[..., :3], matrix_from_quat(flat[..., 3:]), origins)
            for key in all_metrics:
                all_metrics[key].append(metrics[key].reshape(len(poses), count).cpu().numpy())
        all_metrics = {key: np.concatenate(value) for key, value in all_metrics.items()}
        np.savez_compressed(args.output / "clearance_samples.npz", **all_metrics)
        for index, row in enumerate(rows):
            row.update({key: float(value[:, index].min()) for key, value in all_metrics.items()})
            row["sphere_clear_all_samples"] = row["self_margin_m"] >= 0 and row["world_margin_m"] >= 0
            row["max_joint_speed_rad_s"] = np.abs(result["joint_velocities"][:, index]).max(axis=0).tolist()
            row["max_joint_excursion_rad"] = np.abs(result["full_joint_positions"][:, index, :6]-row["q0"]).max(axis=0).tolist()
            speed = np.abs(result["joint_velocities"][:, index])
            row["max_joint_speed_deg_s"] = np.degrees(speed.max(axis=0)).tolist()
            row["peak_speed_time_s"] = (speed.argmax(axis=0)*.002).tolist()
            row["physics_velocity_limits_rad_s"] = physics_limits[index].tolist()
            row["samples_near_physics_speed_cap"] = (speed >= .99*physics_limits[index]).sum(axis=0).tolist()
            row["samples_above_191_deg_s"] = (speed > np.radians(191.)).sum(axis=0).tolist()
            row["samples_above_review_speed"] = (speed > np.radians(args.review_speed_deg_s)).sum(axis=0).tolist()
            row["speed_review_pass"] = bool(np.isfinite(speed).all()
                and np.max(speed) <= np.radians(args.review_speed_deg_s)
                and not any(row["samples_near_physics_speed_cap"]))
        summary = {"motion": motion, "controller": record["osc_params"], "dt": .002,
                   "velocity_limit_mode": args.velocity_limit_mode,
                   "review_speed_deg_s": args.review_speed_deg_s,
                   "displayed_hardware_speed_setting_deg_s": 191.,
                   "body_names": result["body_names"], "joint_names": result["joint_names"],
                   "rows": rows, "candidates": candidates,
                   "dynamics_variants_source": str(args.dynamics_variants.resolve()) if args.dynamics_variants else None,
                   "scope": ("Simulated sweep against the accepted map geometry using supplied fitted parameters. The candidate ensemble is not a confidence bound."
                             if args.dynamics_variants else
                             "Simulated sweep against the accepted map geometry. Reference dynamics are stress scenarios, not Thunder measurements.")}
        (args.output / "preview.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({"checked_states": len(result["body_poses"])*count,
                          "rows": [{k:v for k,v in row.items() if k not in ("params", "q0")} for row in rows]}, indent=2), flush=True)
    except BaseException as exc:
        traceback.print_exc()
        (args.output / "failure.json").write_text(json.dumps({"error": str(exc)})+"\n")
        raise
    finally:
        replay.close()


try:
    main()
finally:
    app.close()
