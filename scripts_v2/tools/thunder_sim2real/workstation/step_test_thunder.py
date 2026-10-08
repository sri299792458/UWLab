"""Thunder controller step-response test (hardware counterpart of sim record R223). Without --execute it only prints the plan.

Purpose: measure how far the arm actually moves toward a HELD Cartesian target offset under the training controller
(OSC Kp 1000/1000/1000/50/50/50, damping ratio 1, torque limits 150/28), i.e. the static-friction dead band and the
"stop short" error, with UR friction compensation off (as fitted) or on (UR direct_torque viscous/coulomb scales).

It reuses collect_thunder.collect() unchanged: robot-state-locked 500 Hz torque loop, start-pose and stationary checks,
joint-excursion guard, torque-to-hold cleanup and the same record format. Only the target sequence differs: a
piecewise-constant offset from the start pose in the base_link frame (Thunder: base y = world up). For every axis, sign
and size: step the target to the offset, hold, step back to zero, hold. The joint-excursion guard defaults to the
largest joint change predicted by inverse kinematics for any step, plus 5 degrees per joint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import collect_thunder as ct
from vendor import ur5e_kinematics as kin

AXES = ("x", "y", "z", "rx", "ry", "rz")
FRICTION_MODES = {
    "off": None,   # zeros: the fitted setting
    "ur_default": {"viscous_scale": [0.9, 0.9, 0.8, 0.9, 0.9, 0.9],   # URScript direct_torque defaults (SW 5.26 manual)
                   "coulomb_scale": [0.8, 0.8, 0.7, 0.8, 0.8, 0.8]},
}
MARGIN_RAD = np.deg2rad(5.0)
POLICY_STEP_SAMPLES = 50                       # one 10 Hz policy step at the 500 Hz collector rate
MAX_TRANSLATION_M, MAX_ROTATION_RAD = 0.04, np.deg2rad(16.0)   # held-test envelope, also the policy-mode cumulative cap


def step_schedule(config):
    """List of (axis index, signed size) steps and the 500 Hz offset array implementing them."""
    sizes_t = [float(v) for v in config["translation_steps_m"]]
    sizes_r = [float(v) for v in config["rotation_steps_rad"]]
    if not sizes_t and not sizes_r:
        raise ValueError("Give at least one step size")
    if any(v <= 0 or v > 0.05 for v in sizes_t) or any(v <= 0 or v > np.deg2rad(20) for v in sizes_r):
        raise ValueError("Step sizes must be positive, <= 50 mm and <= 20 deg")
    axes = [AXES.index(a) for a in config["step_axes"]]
    initial, hold, rest = (float(config[k]) for k in ("initial_hold_s", "step_hold_s", "return_hold_s"))
    if min(initial, hold, rest) < 0.5:
        raise ValueError("Hold times must be at least 0.5 s")
    steps = []
    for ax in axes:
        for size in (sizes_t if ax < 3 else sizes_r):
            for sign in (1.0, -1.0):
                steps.append((ax, sign * size))
    n = lambda seconds: int(round(seconds / ct.DT))
    blocks, schedule, cursor = [np.zeros((n(initial), 6))], [], n(initial)
    for ax, value in steps:
        offset = np.zeros(6); offset[ax] = value
        blocks += [np.tile(offset, (n(hold), 1)), np.zeros((n(rest), 6))]
        schedule.append({"axis": AXES[ax], "value": value, "start_sample": cursor, "hold_samples": n(hold),
                         "return_start_sample": cursor + n(hold), "return_samples": n(rest)})
        cursor += n(hold) + n(rest)
    return steps, np.concatenate(blocks), schedule


def policy_schedule(config):
    """Policy-semantics trials (--mode policy). Every 0.1 s the target is re-anchored to the measured flange pose plus an
    offset, exactly how the trained policy's relative Cartesian action is applied (target = current pose + scale x action).
    single:   one 0.1 s command of size d, then zero action (target = current pose) for the rest of the trial;
    constant: the same command for `repeats` consecutive 0.1 s steps, then zero action.
    After each trial the target returns to the initial center pose and holds there. Returns (trials, offsets, anchors,
    schedule); anchors follow collect_thunder.collect (1 = re-anchor to the measured pose, 2 = initial center)."""
    p = config["policy_test"]
    steps_per_trial, n = int(p["policy_steps_per_trial"]), lambda seconds: int(round(seconds / ct.DT))
    initial, rest = float(p["initial_hold_s"]), float(p["return_hold_s"])
    if steps_per_trial < 2 or min(initial, rest) < 0.5:
        raise ValueError("Use at least 2 policy steps per trial and holds of at least 0.5 s")
    trials = []
    for ax in [AXES.index(a) for a in p["axes"]]:
        single = p["single_translation_m"] if ax < 3 else p["single_rotation_rad"]
        constant = p["constant_translation_m"] if ax < 3 else p["constant_rotation_rad"]
        cap = MAX_TRANSLATION_M if ax < 3 else MAX_ROTATION_RAD
        for size, repeats, kind in [(v, 1, "single") for v in single] + [(v, int(r), "constant") for v, r in constant]:
            if size <= 0 or repeats < 1 or repeats >= steps_per_trial or size * repeats > cap + 1e-6:
                raise ValueError(f"{kind} {AXES[ax]} {size} x {repeats} exceeds the 40 mm / 16 deg cumulative cap")
            for sign in (1.0, -1.0):
                trials.append((ax, sign * size, repeats, kind))
    offsets = [np.zeros((n(initial), 6))]; anchors = [np.zeros(n(initial), dtype=int)]; anchors[0][0] = 2
    schedule, cursor = [], n(initial)
    for ax, value, repeats, kind in trials:
        start = cursor
        for k in range(steps_per_trial):
            block = np.zeros((POLICY_STEP_SAMPLES, 6))
            if k < repeats:
                block[:, ax] = value
            code = np.zeros(POLICY_STEP_SAMPLES, dtype=int); code[0] = 1
            offsets.append(block); anchors.append(code); cursor += POLICY_STEP_SAMPLES
        code = np.zeros(n(rest), dtype=int); code[0] = 2
        offsets.append(np.zeros((n(rest), 6))); anchors.append(code)
        schedule.append({"kind": kind, "axis": AXES[ax], "value": value, "repeats": repeats, "start_sample": start,
                         "policy_steps": steps_per_trial, "step_samples": POLICY_STEP_SAMPLES,
                         "return_start_sample": cursor, "return_samples": n(rest)})
        cursor += n(rest)
    return trials, np.concatenate(offsets), np.concatenate(anchors), schedule


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def ik_step(q0, target_pos, target_quat, iters=200, damping=1e-3):
    """Damped least-squares IK with the vendored calibrated kinematics (planning only)."""
    q = q0.copy()
    for _ in range(iters):
        pos, quat = kin.get_ee_pose(q)
        err = np.concatenate([target_pos - pos, kin.quat_to_axis_angle(ct.quat_multiply(target_quat, quat_conjugate(quat)))])
        if np.linalg.norm(err) < 1e-9:
            break
        jac = kin.compute_jacobian_calibrated(q)
        q = q + jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
    return q, float(np.linalg.norm(err))


def predicted_excursions(config, steps):
    q0 = np.asarray(config["start_joint_positions_rad"], dtype=float)
    pos0, quat0 = kin.get_ee_pose(q0)
    worst, residual = np.zeros(6), 0.0
    for ax, value in steps:
        offset = np.zeros(6); offset[ax] = value
        q, err = ik_step(q0, pos0 + offset[:3], ct.quat_multiply(kin.axis_angle_to_quat(offset[3:]), quat0))
        worst = np.maximum(worst, np.abs(q - q0)); residual = max(residual, err)
    return worst, residual


def make_plan(config, friction, mode="held"):
    config = copy.deepcopy(config)
    if mode not in ("held", "policy"):
        raise ValueError("mode must be held or policy")
    if friction not in FRICTION_MODES:
        raise ValueError(f"friction must be one of {sorted(FRICTION_MODES)}")
    if FRICTION_MODES[friction] is None:
        config.pop("direct_torque_params", None)
    else:
        config["direct_torque_params"] = FRICTION_MODES[friction]
    ct.install_calibration()
    if mode == "held":
        steps, offsets, schedule = step_schedule(config)
        anchors = None
    else:
        trials, offsets, anchors, schedule = policy_schedule(config)
        steps = [(ax, value * repeats) for ax, value, repeats, _ in trials]   # worst case: every command fully tracked
    worst, residual = predicted_excursions(config, steps)
    if residual > 1e-6:
        raise ValueError(f"Planning IK did not converge (residual {residual:.2e})")
    if config.get("joint_excursion_limit_rad") is None:
        config["joint_excursion_limit_rad"] = (worst + MARGIN_RAD).tolist()
    # Same checks the sysid collector requires (address, firmware, payload, start pose, gains, open empty gripper).
    if not config.get("robot_ip") or not config.get("polyscope_version"):
        raise ValueError("Fill robot_ip and polyscope_version")
    if config.get("gripper_configuration") != "open_empty":
        raise ValueError("Run with the gripper open and empty, as in the sysid recordings")
    mass = config.get("payload_mass_kg")
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ValueError("payload_mass_kg must be the configured Thunder tool payload")
    ct.vector(config, "payload_cog_m", 3)
    ct.vector(config, "start_joint_positions_rad", 6)
    ct.vector(config, "joint_excursion_limit_rad", 6, positive=True)
    for key in ("motion_stiffness", "motion_damping_ratio", "torque_max"):
        ct.vector(config, key, 6, positive=True)
    if np.any(ct.vector(config, "torque_max", 6) > [150, 150, 150, 28, 28, 28]):
        raise ValueError("torque_max exceeds the UWLab UR5e limits")
    ct.friction_scales(config)
    config["step_test"] = {"mode": mode, "friction_mode": friction, "schedule": schedule,
                           "predicted_max_joint_excursion_rad": worst.tolist(), "excursion_margin_rad": float(MARGIN_RAD)}
    return config, offsets, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--friction", choices=sorted(FRICTION_MODES), default="off",
                        help="off = fitted setting (zero UR compensation); ur_default = URScript default scales")
    parser.add_argument("--mode", choices=["held", "policy"], default="held",
                        help="held = fixed targets from the center (sim R223 phase B); policy = 10 Hz re-anchored targets (phases A/C)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Connect and execute the displayed step sequence")
    args = parser.parse_args()
    config, offsets, anchors = make_plan(json.loads(args.config.read_text()), args.friction, args.mode)
    info = config["step_test"]
    print(json.dumps({"steps": len(info["schedule"]), "duration_s": round(len(offsets) * ct.DT, 1), "samples": len(offsets),
                      "mode": args.mode, "friction_mode": args.friction, "direct_torque_params": ct.friction_scales(config),
                      "control_script": "vendor fixed (ur_rtde 1.6.5 bug)" if args.friction != "off" else "ur_rtde compiled-in",
                      "max_offset_m_rad": np.abs(offsets).max(axis=0).round(4).tolist(),
                      "predicted_max_joint_excursion_deg": np.rad2deg(info["predicted_max_joint_excursion_rad"]).round(2).tolist(),
                      "joint_excursion_limit_deg": np.rad2deg(config["joint_excursion_limit_rad"]).round(2).tolist(),
                      "start_joint_positions_rad": config["start_joint_positions_rad"],
                      "kp": config["motion_stiffness"], "execute": args.execute}, indent=2))
    if args.execute:
        if args.output is None:
            parser.error("--execute requires --output")
        ct.collect(config, offsets, args.output, anchors=anchors)


if __name__ == "__main__":
    main()
