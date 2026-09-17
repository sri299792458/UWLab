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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
    lab = UmiCubeTrainCfg()
    cfg.scene.robot.init_state.pos = lab.scene.robot.init_state.pos
    cfg.scene.robot.init_state.rot = lab.scene.robot.init_state.rot
    cfg.scene.table = copy.deepcopy(lab.scene.table)
    cfg.scene.ground = copy.deepcopy(lab.scene.ground)
    cfg.scene.env_spacing = 3.5
    return cfg


def main():
    candidates = json.loads(args.candidates.read_text())["candidates"]
    motion = {"duration_s": 8., "f0_hz": .1, "f1_hz": 3.,
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
    rows = [{"candidate": c["candidate"], "variant": v["name"], "q0": c["joint_positions_rad"], "params": v["params"]}
            for c in candidates for v in variants]
    count = len(rows)
    replay = ThunderReplay(record, num_envs=count, device=args.device, delay_max=5, env_cfg=lab_config())
    try:
        replay.q0 = torch.tensor([row["q0"] for row in rows], device=replay.device, dtype=torch.float32)
        replay.qd0 = torch.zeros_like(replay.q0)
        positions, quats = [], []
        for row in rows:
            p0, r0 = collect_thunder.kin.get_ee_pose(np.array(row["q0"]))
            positions.append(p0+offsets[:, :3])
            quats.append(np.array([collect_thunder.quat_multiply(collect_thunder.kin.axis_angle_to_quat(o[3:]), r0) for o in offsets]))
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
        summary = {"motion": motion, "controller": record["osc_params"], "dt": .002,
                   "body_names": result["body_names"], "joint_names": result["joint_names"],
                   "rows": rows, "candidates": candidates,
                   "scope": "Simulated sweep against the accepted map geometry. Reference dynamics are stress scenarios, not Thunder measurements."}
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
