"""Standalone Thunder recorder. Without --execute this only prints a motion plan.

The vendored UWLab OSC is unmodified. Hardware-specific parameters are installed
in memory. Recorded joint positions follow UWLab's pre-command convention.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from vendor import ur5e_kinematics as kin

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "thunder_calibration.json"
DT = 1 / 500
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def install_calibration():
    data = json.loads(CALIBRATION.read_text())
    joints, inertials = data["calibrated_joints"], data["link_inertials"]
    kin.CALIBRATED_JOINTS = [
        {"xyz": np.array(x), "rpy": np.array(r)}
        for x, r in zip(joints["xyz"], joints["rpy"])
    ]
    kin.LINK_INERTIAS = [
        {"mass": m, "com": np.array(c), "I": np.array(i)}
        for m, c, i in zip(inertials["masses"], inertials["coms"], inertials["inertias"])
    ]
    return data


def vector(config, key, n, *, positive=False):
    value = np.asarray(config.get(key), dtype=float)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {n} finite numbers")
    if positive and np.any(value <= 0):
        raise ValueError(f"{key} must be positive")
    return value


def make_plan(config):
    # Deliberately require a measured payload and a reviewed start pose. No moveJ
    # to an inherited reference-robot pose is issued by this recorder.
    if not config.get("robot_ip") or not config.get("polyscope_version"):
        raise ValueError("Fill robot_ip and polyscope_version in the collection config")
    mass = config.get("payload_mass_kg")
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ValueError("payload_mass_kg must be the configured Thunder tool payload")
    vector(config, "payload_cog_m", 3)
    vector(config, "start_joint_positions_rad", 6)
    vector(config, "joint_excursion_limit_rad", 6, positive=True)
    amps = vector(config, "amplitudes_m_rad", 6)
    if np.any(amps < 0) or not np.any(amps > 0):
        raise ValueError("Choose nonnegative excitation amplitudes, with at least one nonzero axis")
    for key in ("motion_stiffness", "motion_damping_ratio", "torque_max"):
        vector(config, key, 6, positive=True)
    if np.any(vector(config, "torque_max", 6) > [150, 150, 150, 28, 28, 28]):
        raise ValueError("torque_max exceeds the UWLab UR5e limits")
    if config.get("gripper_configuration") != "open_empty":
        raise ValueError("This fitting setup expects the gripper open, with no held object")
    duration = float(config["duration_s"])
    f0, f1 = float(config["f0_hz"]), float(config["f1_hz"])
    if not np.isfinite([duration, f0, f1]).all() or duration < 5 or not 0 < f0 <= f1 < 250:
        raise ValueError("Use duration >= 5 seconds and 0 < f0 <= f1 < 250 Hz")
    tolerance = float(config["start_joint_tolerance_rad"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("start_joint_tolerance_rad must be positive")
    return generate_offsets(config)


def generate_offsets(config):
    """The exact command waveform, shared by the collector and lab preview."""
    duration = float(config["duration_s"])
    f0, f1 = float(config["f0_hz"]), float(config["f1_hz"])
    amps = np.asarray(config["amplitudes_m_rad"], dtype=float)
    # Preserve the pinned UW collector's sample and ramp construction exactly.
    # Control/recording still runs at 500 Hz; this is the waveform parameter.
    count = int(duration / DT)
    t = np.linspace(0, duration, count)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    envelope = np.ones(count)
    up, down = int(2. / DT), int(3. / DT)
    envelope[:up] = np.linspace(0, 1, up)
    envelope[-down:] = np.linspace(1, 0, down)
    phases = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    offsets = np.zeros((count, 6))
    for i in range(6):
        offsets[:, i] = amps[i] * envelope * np.sin(phase + phases[i])
    return offsets


def quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def collect(config, offsets, output):
    # Importing this module and planning never open a robot connection.
    import torch
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    version = importlib.metadata.version("ur-rtde")
    if version != "1.6.2":
        raise RuntimeError(f"Expected UWLab's ur-rtde 1.6.2, found {version}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration = install_calibration()
    kp = np.asarray(config["motion_stiffness"])
    dr = np.asarray(config["motion_damping_ratio"])
    osc = kin.OperationalSpaceController(
        motion_stiffness=kp, motion_damping_ratio=dr,
        torque_max=np.asarray(config["torque_max"]),
    )
    control = receive = None
    rows = []
    initial_q = None
    completed = False
    failure = None
    cleanup_errors = []
    try:
        receive = RTDEReceiveInterface(config["robot_ip"], 500)
        initial_q = np.asarray(receive.getActualQ())
        if initial_q.shape != (6,) or not np.isfinite(initial_q).all():
            raise RuntimeError("Received invalid initial joint positions")
        expected = np.asarray(config["start_joint_positions_rad"])
        if np.max(np.abs(initial_q - expected)) > config["start_joint_tolerance_rad"]:
            raise RuntimeError("Robot is outside the configured start-pose tolerance; no motion sent")
        initial_qd = np.asarray(receive.getActualQd())
        if initial_qd.shape != (6,) or not np.isfinite(initial_qd).all():
            raise RuntimeError("Received invalid initial joint velocities")
        if np.max(np.abs(initial_qd)) > 0.01:
            raise RuntimeError("Robot must be stationary before collection")
        center_pos, center_quat = kin.get_ee_pose(initial_q)
        control = RTDEControlInterface(
            config["robot_ip"], 500,
            RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT,
        )
        if not control.setPayload(config["payload_mass_kg"], config["payload_cog_m"]):
            raise RuntimeError("setPayload failed")
        for offset in offsets:
            period = control.initPeriod()
            # Timestamps bracket the receive calls; they expose duplicates and
            # overruns without pretending host and robot clocks are identical.
            host_before = time.monotonic()
            robot_before = receive.getTimestamp()
            q = np.asarray(receive.getActualQ())
            qd = np.asarray(receive.getActualQd())
            robot_after = receive.getTimestamp()
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise RuntimeError("Received non-finite robot state")
            if np.any(np.abs(q - initial_q) > config["joint_excursion_limit_rad"]):
                raise RuntimeError("Configured joint excursion exceeded")
            pos, quat = kin.get_ee_pose(q)
            jacobian = kin.compute_jacobian_calibrated(q)
            target_pos = center_pos + offset[:3]
            target_quat = quat_multiply(kin.axis_angle_to_quat(offset[3:]), center_quat)
            osc.set_target(target_pos, target_quat)
            torque = osc.compute(pos, quat, jacobian @ qd, jacobian)
            if not np.isfinite(torque).all():
                raise RuntimeError("Non-finite torque command")
            command_time = time.monotonic()
            if not control.directTorque(torque.tolist(), friction_comp=False):
                raise RuntimeError("directTorque returned failure")
            rows.append((q.copy(), qd.copy(), torque.copy(), target_pos.copy(), target_quat.copy(),
                         host_before, command_time, robot_before, robot_after))
            control.waitPeriod(period)
        completed = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if control is not None:
            # Same torque-to-hold handoff used by the pinned UWLab collector.
            try:
                control.directTorque([0.0] * 6, friction_comp=False)
                q_hold = receive.getActualQ()
                control.servoJ(q_hold, 0.5, 0.5, 0.1, 0.1, 300)
                control.servoStop()
            except Exception as exc:
                cleanup_errors.append(str(exc))
            try:
                control.stopScript()
            except Exception as exc:
                cleanup_errors.append(str(exc))
            try:
                control.disconnect()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if receive is not None:
            try:
                receive.disconnect()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if rows:
            columns = list(zip(*rows))
            tensor = lambda index: torch.tensor(np.asarray(columns[index]), dtype=torch.float64)
            record = {
                "schema_version": 1, "robot": "thunder", "source_kind": "real_robot",
                "joint_names": JOINT_NAMES, "sample_phase": "pre_command",
                "pose_frame": "base_link", "quaternion_order": "wxyz",
                "dt": DT, "control_freq": 500, "completed": completed,
                "failure": failure, "cleanup_errors": cleanup_errors,
                "joint_positions": tensor(0), "joint_velocities": tensor(1),
                "joint_torques": tensor(2), "initial_joint_pos": tensor(0)[0].clone(),
                "initial_joint_vel": tensor(1)[0].clone(),
                "waypoint_target_pos": tensor(3), "waypoint_target_quat": tensor(4),
                "waypoint_step_indices": torch.arange(len(rows)), "num_waypoints": len(rows),
                "host_sample_times_s": tensor(5), "host_command_times_s": tensor(6),
                "robot_sample_times_s": tensor(7), "robot_after_read_times_s": tensor(8),
                "osc_params": {k: config[k] for k in
                               ("motion_stiffness", "motion_damping_ratio", "torque_max")},
                "collection_config": config, "calibration": calibration,
                "calibration_sha256": hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
                "controller_provenance": json.loads((HERE / "vendor/PROVENANCE.json").read_text()),
                "ur_rtde_version": version,
            }
            torch.save(record, output)
            print(f"Saved {len(rows)} samples to {output}; completed={completed}")
        if cleanup_errors:
            print("Controller cleanup reported:", cleanup_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Connect and execute the displayed excitation")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    offsets = make_plan(config)
    print(json.dumps({"samples": len(offsets), "frequency_hz": 500,
                      "offset_min_m_rad": offsets.min(axis=0).tolist(),
                      "offset_max_m_rad": offsets.max(axis=0).tolist(),
                      "start_joint_positions_rad": config["start_joint_positions_rad"],
                      "kp": config["motion_stiffness"],
                      "kd": (2 * np.sqrt(config["motion_stiffness"]) * config["motion_damping_ratio"]).tolist(),
                      "execute": args.execute}, indent=2))
    if args.execute:
        if args.output is None:
            parser.error("--execute requires --output")
        collect(config, offsets, args.output)


if __name__ == "__main__":
    main()
