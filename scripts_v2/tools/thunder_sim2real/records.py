"""Validate recorded inputs and export a traceable Stage-2 profile on CPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_api():
    path = REPO / "source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/sysid_profile.py"
    spec = importlib.util.spec_from_file_location("thunder_profile_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array(record, key, shape):
    value = np.asarray(record[key], dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key}: expected finite {shape}, received {value.shape}")
    return value


def validate_record(record, *, allow_synthetic=False):
    expected = {
        "schema_version": 1, "robot": "thunder", "sample_phase": "pre_command",
        "pose_frame": "base_link", "quaternion_order": "wxyz", "control_freq": 500,
        "completed": True, "joint_names": JOINT_NAMES,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    kind = record.get("source_kind")
    if kind != "real_robot" and not (allow_synthetic and kind == "synthetic_test"):
        raise ValueError("Expected real-robot data (synthetic data is restricted to explicit tests)")
    if record.get("failure") or record.get("cleanup_errors"):
        raise ValueError("Recording or controller cleanup did not complete cleanly")
    if record.get("dt") != 0.002:
        raise ValueError("Expected 500 Hz data; do not silently resample or reinterpret delay steps")
    expected_calibration = json.loads((HERE / "workstation/thunder_calibration.json").read_text())
    if record.get("calibration") != expected_calibration:
        raise ValueError("Recording uses a different robot calibration")
    if record.get("calibration_sha256") != sha256(HERE / "workstation/thunder_calibration.json"):
        raise ValueError("Calibration fingerprint does not match")
    if record.get("ur_rtde_version") != "1.6.2":
        raise ValueError("Recording did not use the pinned UWLab RTDE version")
    expected_controller = json.loads((HERE / "workstation/vendor/PROVENANCE.json").read_text())
    if record.get("controller_provenance") != expected_controller:
        raise ValueError("Controller provenance differs from the pinned collector")
    if record.get("collection_config", {}).get("gripper_configuration") != "open_empty":
        raise ValueError("Expected open, unloaded gripper")
    n = len(record["joint_positions"])
    if n < 2:
        raise ValueError("At least two recorded samples are required")
    joints = array(record, "joint_positions", (n, 6))
    array(record, "joint_velocities", (n, 6))
    array(record, "joint_torques", (n, 6))
    initial = array(record, "initial_joint_pos", (6,))
    array(record, "initial_joint_vel", (6,))
    if not np.allclose(initial, joints[0], atol=1e-7, rtol=0):
        raise ValueError("Initial state must be the first pre-command sample")
    indices = array(record, "waypoint_step_indices", (n,))
    if not np.array_equal(indices, np.arange(n)) or record["num_waypoints"] != n:
        raise ValueError("Expected exactly one command per 500 Hz sample")
    array(record, "waypoint_target_pos", (n, 3))
    quat = array(record, "waypoint_target_quat", (n, 4))
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-5):
        raise ValueError("Target quaternions must be normalized wxyz values")
    api = profile_api()
    for key in ("motion_stiffness", "motion_damping_ratio", "torque_max"):
        api.six_numbers(record["osc_params"][key], key, positive=True)
    times = array(record, "robot_sample_times_s", (n,))
    robot_after = array(record, "robot_after_read_times_s", (n,))
    host_sample = array(record, "host_sample_times_s", (n,))
    host_command = array(record, "host_command_times_s", (n,))
    intervals = np.diff(times)
    # This fixed-step replay has no missing-sample interpolation model. Refuse
    # gaps rather than treating irregular robot samples as exact 2 ms samples.
    if np.any(np.abs(intervals - 0.002) > 0.0004):
        raise ValueError("Robot timestamps contain duplicate/missing samples; fixed 500 Hz replay is invalid")
    if np.any(np.diff(host_sample) <= 0) or np.any(host_command < host_sample):
        raise ValueError("Host timestamps are not ordered sample-before-command")
    if np.any(robot_after != times):
        raise ValueError("A robot update crossed the Q/Qd read; recollect to obtain consistent state samples")
    return {"samples": n, "duration_s": n * 0.002, "source_kind": kind,
            "sample_phase": "pre_command", "robot_dt_min_s": float(intervals.min()),
            "robot_dt_max_s": float(intervals.max()),
            "joint_excursion_rad": np.max(np.abs(joints - joints[0]), axis=0).tolist(),
            "max_torque_nm": np.max(np.abs(record["joint_torques"].numpy()), axis=0).tolist()}


def load_record(path, *, allow_synthetic=False):
    record = torch.load(path, map_location="cpu", weights_only=True)
    return record, validate_record(record, allow_synthetic=allow_synthetic)


def export_profile(fit_path, output):
    fit_path = Path(fit_path).resolve()
    fit = json.loads(fit_path.read_text())
    if fit.get("source_kind") != "real_robot" or fit.get("comparison_phase") != "pre_command":
        raise ValueError("Export requires a real Thunder fit using the declared pre-command alignment")
    record_path = Path(fit["record_path"])
    record, _ = load_record(record_path)
    if sha256(record_path) != fit["record_sha256"]:
        raise ValueError("Source recording changed after fitting")
    profile = {
        "schema_version": 1, "robot": "thunder", "source_kind": "real_robot",
        "calibration_hash": record["calibration"]["kinematics_calibration"]["hash"],
        "sysid": fit["sysid"], "controller": record["osc_params"],
        "fit_dt_s": record["dt"], "fitted_delay_steps": fit["delay_steps"],
        "fitted_delay_seconds": fit["delay_steps"] * record["dt"],
        "stage2_delay": {"mode": "upstream_range", "physics_hz": 120, "steps": [0, 1]},
        "record_sha256": fit["record_sha256"], "fit_sha256": sha256(fit_path),
        "fit_path": str(fit_path), "validation_status": "fitted_candidate",
        "note": "Fit error is not proof of real task transfer. Evaluate held-out motions before selecting this profile.",
    }
    profile_api().validate_profile(profile)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("validate")
    check.add_argument("record", type=Path)
    export = sub.add_parser("export-profile")
    export.add_argument("--fit", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = load_record(args.record)[1] if args.mode == "validate" else export_profile(args.fit, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
