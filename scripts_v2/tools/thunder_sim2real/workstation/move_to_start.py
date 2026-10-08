"""Slowly return an already nearby robot to the configured joint start pose.

Without --execute, print the target without connecting. The operator must have
checked clearance for this small joint-space move; this is not collision planning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

MAX_INITIAL_ERROR_RAD = np.deg2rad(5.0)
MOVE_SPEED_RAD_S = 0.05
MOVE_ACCEL_RAD_S2 = 0.1
TARGET_TOLERANCE_RAD = 0.003
STATIONARY_SPEED_RAD_S = 0.01
TIMEOUT_S = 30.0
# PolyScope 5.25+ adds bit 11: three-position enabling (3PE) input active.
# It is informational, unlike the stop/recovery/fault flags in bits 2-10.
# Source: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
SAFETY_READY_BITS = (1 << 0) | (1 << 1)
SAFETY_ALLOWED_BITS = SAFETY_READY_BITS | (1 << 11)


def joint_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite joint values")
    return result


def read_state(receive):
    if not receive.isConnected():
        raise RuntimeError("RTDE receive connection lost")
    safety = int(receive.getSafetyStatusBits())
    if not (safety & SAFETY_READY_BITS) or (safety & ~SAFETY_ALLOWED_BITS):
        raise RuntimeError(f"Robot is not in normal/reduced safety mode (bits={safety}); no automatic reset")
    if receive.getRobotMode() != 7:
        raise RuntimeError("Robot must be powered with brakes released and ready")
    return (joint_vector(receive.getActualQ(), "Actual joints"),
            joint_vector(receive.getActualQd(), "Actual joint velocities"))


def require_nearby_stationary(q, qd, target):
    # Do not wrap angles: a different joint revolution must not become a long move.
    delta = target - q
    if np.max(np.abs(delta)) > MAX_INITIAL_ERROR_RAD:
        raise RuntimeError("Refusing move: a joint is more than 5 degrees from the start pose. "
                           f"Errors (degrees): {np.rad2deg(delta).round(3).tolist()}")
    if np.max(np.abs(qd)) > STATIONARY_SPEED_RAD_S:
        raise RuntimeError("Refusing move: robot is already moving")


def move_to_start(config, receive_factory=None, control_factory=None):
    target = joint_vector(config.get("start_joint_positions_rad"), "Start joints")
    ip = config.get("robot_ip")
    if not isinstance(ip, str) or not ip:
        raise ValueError("Configuration must specify robot_ip")
    if receive_factory is None or control_factory is None:
        from rtde_receive import RTDEReceiveInterface
        from rtde_control import RTDEControlInterface
        receive_factory = RTDEReceiveInterface
        control_factory = RTDEControlInterface
    receive = control = None
    motion_started = completed = False
    primary_error = None
    try:
        receive = receive_factory(ip, 125)
        q, qd = read_state(receive)
        require_nearby_stationary(q, qd, target)
        print("Current joints (deg):", np.rad2deg(q).round(3).tolist(), flush=True)
        print("Target joints  (deg):", np.rad2deg(target).round(3).tolist(), flush=True)
        if np.max(np.abs(target - q)) <= TARGET_TOLERANCE_RAD:
            print("Already at the start pose; no motion sent")
            return
        control = control_factory(ip)
        # Check again after the control interface/script handshake.
        q, qd = read_state(receive)
        require_nearby_stationary(q, qd, target)
        print("Moving at 0.05 rad/s (2.86 deg/s); Ctrl-C requests a stop", flush=True)
        motion_started = True
        if not control.moveJ(target.tolist(), MOVE_SPEED_RAD_S, MOVE_ACCEL_RAD_S2, True):
            raise RuntimeError("moveJ returned failure")
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            q, qd = read_state(receive)
            if (np.max(np.abs(target - q)) <= TARGET_TOLERANCE_RAD
                    and np.max(np.abs(qd)) <= 0.002 and control.isSteady()):
                completed = True
                print("At start pose and stationary. Maximum joint error (deg):",
                      round(float(np.rad2deg(np.max(np.abs(target - q)))), 3))
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Start-pose move did not finish within 30 seconds")
            time.sleep(0.02)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        errors = []
        if control is not None:
            if motion_started and not completed:
                try:
                    control.stopJ(0.5)
                except Exception as exc:
                    errors.append(f"stopJ: {exc}")
            for method in (control.stopScript, control.disconnect):
                try:
                    method()
                except Exception as exc:
                    errors.append(str(exc))
        if receive is not None:
            try:
                receive.disconnect()
            except Exception as exc:
                errors.append(f"receive disconnect: {exc}")
        if errors:
            message = "Move cleanup errors: " + "; ".join(errors)
            if primary_error is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target = joint_vector(config.get("start_joint_positions_rad"), "Start joints")
    if args.execute:
        move_to_start(config)
    else:
        print(json.dumps({"target_joints_deg": np.rad2deg(target).tolist(),
                          "max_initial_joint_error_deg": 5.0, "speed_rad_s": MOVE_SPEED_RAD_S,
                          "acceleration_rad_s2": MOVE_ACCEL_RAD_S2, "execute": False}, indent=2))


if __name__ == "__main__":
    main()
