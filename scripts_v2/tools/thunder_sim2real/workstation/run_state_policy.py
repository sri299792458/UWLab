"""Run the R214 state policy (broad 60 mm cube stacking, UMI fingertips) on Thunder.

Modes (each later mode does everything the earlier ones do):
  track    camera only: cube poses in base_link with placement checks (table height, up face, trained region). No robot.
  check    + robot state (receive only) + gripper position: with a cube held in the closed gripper, compares the camera's cube
           pose in the wrist frame with training grasps. Catches camera-transform frame or direction errors before motion.
  dry-run  + the policy at 10 Hz on real observations; prints and logs actions and targets. Sends NO robot or gripper
           command.
  execute  + the 500 Hz torque loop (same robot-state-locked loop and OSC as collect_thunder / the step tests) and the
           gripper. Asks for confirmation before the first torque command.

What the policy sees is rebuilt exactly as in training (state_policy/obs.py, verified on 40,960 sim observations):
joint angles from RTDE, wrist pose from the calibrated kinematics, 6 gripper joints from the Robotiq position
(state_policy/gripper.py), cube poses from the L515 (state_policy/tracker.py), its own previous action, 5-step history.
Action semantics are training's: every 0.1 s target = measured wrist pose + scale * action (rotation applied in the base
frame), held for 50 robot cycles; gripper closes if action[6] < 0. Each cycle sends its torque first and then runs the
policy step, so a new target takes effect from the next cycle (2 ms after the state it was computed from).

Our additions (not in training): latency compensation of the HELD cube and the safety stops listed in
collection.state_policy.json "limits". While the gripper is commanded closed and stopped on an object, the held cube is
treated as rigidly attached to the wrist between the camera capture and now: T_base_cube(now) = T_base_wrist(now) *
inv(T_base_wrist(capture)) * T_base_cube(capture), with the wrist pose at capture time from the logged joint angles.
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only add latency
    os.environ.setdefault(_var, "1")

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import collect_thunder as ct
from vendor import ur5e_kinematics as kin
from state_policy import obs as O
from state_policy import frames as F

HERE = Path(__file__).resolve().parent
PKG = HERE / "state_policy/r214"
POLICY_CYCLES = 50                      # 0.1 s at 500 Hz
MODES = ("track", "check", "dry-run", "execute")


class SafetyStop(RuntimeError):
    pass


def load_manifest():
    m = json.loads((PKG / "manifest.json").read_text())
    if hashlib.sha256((PKG / m["policy"]["file"]).read_bytes()).hexdigest() != m["policy"]["sha256"]:
        raise RuntimeError(f"{m['policy']['file']} does not match its manifest hash")
    return m


def load_policy(m):
    """R214 actor in NumPy: h = (obs - mean) / (std + 0.01); 4 x elu(W h + b); action = W h + b. No torch needed.
    Checked at load against actions the policy produced in the training sim for stored observations."""
    w = np.load(PKG / m["policy"]["file"])
    mean, std = w["obs_mean"], w["obs_std"] + float(w["obs_eps"])
    layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(5)]

    def act(obs):
        h = (np.asarray(obs, dtype=float) - mean) / std
        for i, (W, b) in enumerate(layers):
            h = W @ h + b
            if i < 4:
                h = np.where(h > 0, h, np.expm1(h))
        return h

    error = max(float(np.abs(act(o) - a).max()) for o, a in zip(w["check_obs"], w["check_action"]))
    if error > 1e-4:
        raise RuntimeError(f"NumPy policy differs from the recorded sim actions by {error:.2e}")
    return act


def wrist_matrix(q):
    pos, quat = kin.get_ee_pose(q)
    return F.pos_quat_to_matrix(pos, quat), pos, quat


def action_to_target(pos, quat, action, scale):
    """Training's RelCartesianOSCAction.process_actions: target = current + scale*action; rotation delta applied first."""
    scaled = np.asarray(action[:6], dtype=float) * scale
    return pos + scaled[:3], ct.quat_multiply(kin.axis_angle_to_quat(scaled[3:6]), quat)


def start_pose_report(q, m, limits):
    ref = np.load(PKG / "train_start_arm_joints.npz")
    d = np.linalg.norm(ref["q"].astype(float) - q, axis=1)
    i = int(d.argmin())
    rng = m["arm_joint_range_in_rollouts_rad"]
    margin = limits["joint_range_margin_rad"]
    outside = [j for j in range(6) if not rng["min"][j] - margin <= q[j] <= rng["max"][j] + margin]
    return {"nearest_training_start_distance_rad": float(d[i]), "nearest_family": str(ref["families"][ref["family"][i]]),
            "nearest_start_rad": ref["q"][i].astype(float).round(4).tolist(),
            "bank_leave_one_out_p99_rad": float(ref["loo_nn_p99"]), "joints_outside_rollout_range": outside,
            "in_distribution": bool(d[i] <= limits["start_joint_nn_max_rad"] and not outside)}


def cube_report(cube, m):
    """Placement checks for a cube in base_link: height above table, up face, inside the trained center box."""
    scene = m["scene_in_base_link"]
    up = np.array(scene["world_up"], dtype=float)
    R = F.pos_quat_to_matrix(cube["pos"], cube["quat"])[:3, :3]
    faces = {"+X": R[:, 0], "-X": -R[:, 0], "+Y": R[:, 1], "-Y": -R[:, 1], "+Z": R[:, 2], "-Z": -R[:, 2]}
    face = max(faces, key=lambda k: faces[k] @ up)
    height = float(cube["pos"] @ up - scene["table_top_y_m"] - scene["cube_size_m"] / 2)
    box = scene["receptive_center_box_m"]
    inside = bool(np.all(cube["pos"] >= np.array(box["min"]) - 0.02) and np.all(cube["pos"] <= np.array(box["max"]) + 0.02))
    return {"pos_m": np.round(cube["pos"], 4).tolist(), "bottom_above_table_mm": round(height * 1000, 1), "up_face": face,
            "up_face_tilt_deg": round(float(np.degrees(np.arccos(np.clip(faces[face] @ up, -1, 1)))), 1),
            "inside_trained_region": inside, "reproj_px": round(cube["reproj_px"], 2), "n_tags": cube["n_tags"]}


class CubeState:
    """Turns the tracker's latest measurements into the cube poses for one policy step (and enforces freshness)."""

    def __init__(self, limits, latency_compensation):
        self.limits = limits
        self.latency_compensation = latency_compensation

    def poses(self, cubes, now, wrist_T_now, wrist_T_at, gripper_closed_on_object):
        out, info = {}, {}
        for name in ("receptive", "insertive"):
            c = cubes[name]
            if not c["valid"]:
                raise SafetyStop(f"{name} cube has not been detected")
            age = now - c["capture_t"]
            T_cap = F.pos_quat_to_matrix(c["pos"], c["quat"])
            attached = False
            if name == "insertive" and gripper_closed_on_object:
                dist = np.linalg.norm(T_cap[:3, 3] - wrist_T_now[:3, 3])
                attached = dist < self.limits["attached_cube_max_wrist_distance_m"]
            if attached and self.latency_compensation:
                T = wrist_T_now @ np.linalg.inv(wrist_T_at(c["capture_t"])) @ T_cap
                max_age = self.limits["cube_max_age_s"]["insertive_attached"]
            else:
                T = T_cap
                max_age = self.limits["cube_max_age_s"]["receptive" if name == "receptive" else "insertive_free"]
            if age > max_age:
                raise SafetyStop(f"{name} cube pose is {age:.2f} s old (limit {max_age} s, attached={attached})")
            out[name] = F.matrix_to_pos_quat(T)
            info[name] = (age, attached)
        return out, info


def gripper_on_object(position, closed_cmd, m):
    """Closed command and stopped between open and the empty-closed position, i.e. on an object."""
    real = m["gripper"]["position_to_finger_joint"]["real_position"]
    return bool(closed_cmd and real[0] + 20 < position < real[-1] - 10)


def run(args):
    cfg = json.loads(args.config.read_text())
    limits = cfg["limits"]
    m = load_manifest()
    scale = np.asarray(m["training"]["arm_action_scale"], dtype=float)
    from state_policy.gripper import GripperMap, GripperProcess
    from state_policy.tracker import CubeTracker
    gmap = GripperMap(m)
    ct.install_calibration()
    T_cam = F.load_camera_transform(args.camera_transform, args.camera_transform_frame)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"mode": args.mode, "config": cfg, "camera_transform_base_link": T_cam.tolist(),
               "camera_transform_source": str(args.camera_transform), "camera_transform_frame": args.camera_transform_frame,
               "policy_sha256": m["policy"]["sha256"], "events": []}
    tracker = CubeTracker(T_cam, {"receptive": PKG / m["cubes"]["receptive_detector"],
                                  "insertive": PKG / m["cubes"]["insertive_detector"]}, **cfg["camera"])
    gripper = receive = control = None
    try:
        tracker.start()
        if args.mode == "track":
            return track_loop(args, tracker, m, summary, out_dir)
        gripper = GripperProcess(cfg["robot_ip"], cfg["gripper"]["speed"], cfg["gripper"]["force"],
                                 allow_motion=args.mode == "execute")
        gripper.start()
        from rtde_receive import RTDEReceiveInterface
        if args.mode == "check":
            receive = RTDEReceiveInterface(cfg["robot_ip"], 500, variables=list(ct.RT_RECEIVE_VARIABLES))
            return check_loop(args, tracker, gripper, receive, m, summary, out_dir)
        return policy_loop(args, cfg, limits, m, scale, gmap, tracker, gripper, summary, out_dir)
    except KeyboardInterrupt:
        summary["stop_reason"] = "operator Ctrl-C"
    finally:
        for proc in (gripper, tracker):
            if proc is not None:
                proc.stop()
        if receive is not None:
            receive.disconnect()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
        print("wrote", out_dir / "summary.json")


def track_loop(args, tracker, m, summary, out_dir):
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        cubes, now = tracker.latest(), time.monotonic()
        line = {}
        for name, c in cubes.items():
            line[name] = cube_report(c, m) if c["valid"] else "not detected"
            if c["valid"]:
                line[name]["age_s"] = round(now - c["capture_t"], 3)
                line[name]["global_time"] = c["global_time"]
        print(json.dumps(line), flush=True)
        summary["last"] = line
        tracker.check()
        time.sleep(0.5)


def check_loop(args, tracker, gripper, receive, m, summary, out_dir):
    expect = m["scene_in_base_link"]["held_cube_center_in_wrist_m"]
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        q = np.asarray(receive.getActualQ())
        W, _, _ = wrist_matrix(q)
        cubes, (gpos, gobj, _) = tracker.latest(), gripper.latest()
        c = cubes["insertive"]
        line = {"gripper_position": gpos, "gripper_on_object": gripper_on_object(gpos, True, m),
                "start_pose": start_pose_report(q, m, json.loads(args.config.read_text())["limits"])}
        if c["valid"]:
            rel = np.linalg.inv(W) @ F.pos_quat_to_matrix(c["pos"], c["quat"])
            p = rel[:3, 3]
            ok = bool(np.all(p >= np.array(expect["p05"]) - 0.015) and np.all(p <= np.array(expect["p95"]) + 0.015))
            line.update(held_cube_in_wrist_m=np.round(p, 4).tolist(), training_median=expect["median"],
                        training_p05=expect["p05"], training_p95=expect["p95"], consistent_with_training_grasps=ok,
                        age_s=round(time.monotonic() - c["capture_t"], 3))
        else:
            line["held_cube"] = "not detected"
        print(json.dumps(line), flush=True)
        summary["last"] = line
        tracker.check(); gripper.check()
        time.sleep(0.5)


def policy_loop(args, cfg, limits, m, scale, gmap, tracker, gripper, summary, out_dir):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    execute = args.mode == "execute"
    act = load_policy(m)
    if execute:
        import importlib.metadata
        version = importlib.metadata.version("ur-rtde")
        if version != ct.REQUIRED_RTDE_VERSION:
            raise RuntimeError(f"Expected ur-rtde {ct.REQUIRED_RTDE_VERSION}, found {version}")
    ct.check_realtime_permissions()
    initial_scheduler = os.sched_getscheduler(0), os.sched_getparam(0)
    n_cycles = int(round(limits["max_episode_s"] * 500))
    n_ticks = -(-n_cycles // POLICY_CYCLES)
    log = {"q": np.zeros((n_cycles, 6)), "qd": np.zeros((n_cycles, 6)), "torque": np.zeros((n_cycles, 6)),
           "host_t": np.zeros(n_cycles), "robot_t": np.zeros(n_cycles), "target_pos": np.zeros((n_cycles, 3)),
           "target_quat": np.zeros((n_cycles, 4)),
           "tick_obs": np.zeros((n_ticks, O.OBS_DIM)), "tick_action": np.zeros((n_ticks, 7)),
           "tick_cycle": np.zeros(n_ticks, dtype=int), "tick_gripper_pos": np.zeros(n_ticks),
           "tick_cube_age": np.zeros((n_ticks, 2)), "tick_attached": np.zeros(n_ticks, dtype=bool),
           "tick_cubes": np.zeros((n_ticks, 2, 7)), "tick_compute_s": np.zeros(n_ticks)}
    osc = kin.OperationalSpaceController(motion_stiffness=np.asarray(cfg["motion_stiffness"]),
                                         motion_damping_ratio=np.asarray(cfg["motion_damping_ratio"]),
                                         torque_max=np.asarray(cfg["torque_max"]))
    zero = ct.friction_scales({})                 # UR friction compensation off, as fitted and trained
    scene = m["scene_in_base_link"]
    up = np.array(scene["world_up"], dtype=float)
    box_lo = np.array(scene["wrist_box_m"]["min"]) - limits["wrist_box_margin_m"]
    box_hi = np.array(scene["wrist_box_m"]["max"]) + limits["wrist_box_margin_m"]
    min_wrist_up = scene["table_top_y_m"] + limits["min_wrist_height_above_table_m"]
    rng = m["arm_joint_range_in_rollouts_rad"]
    j_lo = np.array(rng["min"]) - limits["joint_range_margin_rad"]
    j_hi = np.array(rng["max"]) + limits["joint_range_margin_rad"]
    cube_state = CubeState(limits, cfg["latency_compensation"])
    history = O.ObservationHistory()
    st = {"target": None, "last_action": np.zeros(7), "closed": None, "ticks": 0}
    receive = control = None
    count = 0
    torque_started = False
    gc_enabled = gc.isenabled()
    failure = None

    def wrist_T_at(t):
        i = int(np.clip(np.searchsorted(log["host_t"][:count], t), 0, max(count - 1, 0)))
        return wrist_matrix(log["q"][i])[0]

    def in_workspace(p, extra=0.0):
        return bool(np.all(p >= box_lo - extra) and np.all(p <= box_hi + extra) and p @ up >= min_wrist_up - extra)

    try:
        gc.collect()
        before = ct.thread_schedule_snapshot()
        with ct.fifo_thread_creation(ct.RT_RECEIVE_PRIORITY):
            receive = RTDEReceiveInterface(cfg["robot_ip"], 500, variables=list(ct.RT_RECEIVE_VARIABLES),
                                           rt_priority=ct.RT_RECEIVE_PRIORITY)
        ct.require_new_fifo_thread(before, ct.RT_RECEIVE_PRIORITY, "RTDEReceiveInterface")
        q0, qd0 = np.asarray(receive.getActualQ()), np.asarray(receive.getActualQd())
        if np.max(np.abs(qd0)) > 0.01:
            raise SafetyStop("Robot must be stationary at the start")
        report = start_pose_report(q0, m, limits)
        summary["start_pose"] = report
        cubes0 = tracker.latest()
        summary["cubes_at_start"] = {k: cube_report(c, m) if c["valid"] else None for k, c in cubes0.items()}
        gpos0 = gripper.latest()[0]
        summary["gripper_position_at_start"] = gpos0
        anchors = m["gripper"]["position_to_finger_joint"]["real_position"]
        want = anchors[-1] if cfg["gripper"]["start"] == "closed" else anchors[0]
        if abs(gpos0 - want) > 8:
            message = (f"Gripper at {gpos0:.0f}, expected {cfg['gripper']['start']} ({want}); training Reaching/Near-Object "
                       "starts have the gripper closed and empty")
            if execute:
                raise SafetyStop(message)
            summary["events"].append("warning: " + message)
        print(json.dumps({"start_pose": report, "cubes": summary["cubes_at_start"], "gripper_position": gpos0}, indent=1))
        if not report["in_distribution"] and not args.allow_start_outside_training:
            raise SafetyStop("Start pose is outside the training starts (see start_pose); move to the recommended start "
                             f"{scene['recommended_start_joint_positions_rad']} or pass --allow-start-outside-training")
        if not in_workspace(kin.get_ee_pose(q0)[0]):
            raise SafetyStop("Wrist is outside the trained workspace box at the start")
        if execute:
            if input("Type GO to start torque control (Ctrl-C aborts): ").strip() != "GO":
                raise SafetyStop("Not confirmed")
            before = ct.thread_schedule_snapshot()
            with ct.fifo_thread_creation(ct.RT_CONTROL_PRIORITY):
                control = RTDEControlInterface(cfg["robot_ip"], 500,
                                               RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT,
                                               rt_priority=ct.RT_CONTROL_PRIORITY)
            ct.require_new_fifo_thread(before, ct.RT_CONTROL_PRIORITY, "RTDEControlInterface")
            if not control.setPayload(cfg["payload_mass_kg"], cfg["payload_cog_m"]):
                raise RuntimeError("setPayload failed")
        gc.disable()
        ct.set_app_realtime_priority()
        last_robot_time = float(receive.getTimestamp())

        def policy_step(count, host_before, q, W, wpos, wquat):
            """One 10 Hz policy step on the state read at this cycle; its target is used from the next cycle on
            (2 ms after this state; the anchor pose is this state, as in training)."""
            tick_started = time.monotonic()
            tracker.check(); gripper.check()
            gpos, _, gtime = gripper.latest()
            if host_before - gtime > limits["gripper_max_age_s"]:
                raise SafetyStop("Gripper position is stale")
            cubes, info = cube_state.poses(tracker.latest(), host_before, W, wrist_T_at,
                                           gripper_on_object(gpos, bool(st["closed"]), m))
            (rp, rq), (ip, iq) = cubes["receptive"], cubes["insertive"]
            joints = np.concatenate([q, gmap.sim_joints(gpos)])
            obs = history.push(O.frame_terms(st["last_action"], joints, wpos, wquat, ip, iq, rp, rq))
            action = act(obs)
            if not np.isfinite(action).all():
                raise SafetyStop("Non-finite policy action")
            target = action_to_target(wpos, wquat, action, scale)
            # Targets legitimately lead the arm by up to ~0.15 m (R220); only absurd targets stop here. The measured
            # wrist position is checked against the tight limits every cycle.
            if not in_workspace(target[0], limits["target_extra_margin_m"]):
                raise SafetyStop(f"Policy target {np.round(target[0], 3).tolist()} is far outside the workspace")
            close = bool(action[6] < 0)
            if execute and close != st["closed"]:
                gripper.command(close)
            st.update(target=target, last_action=action, closed=close)
            k = st["ticks"]
            log["tick_obs"][k], log["tick_action"][k], log["tick_cycle"][k] = obs, action, count
            log["tick_gripper_pos"][k] = gpos
            log["tick_cube_age"][k] = info["receptive"][0], info["insertive"][0]
            log["tick_attached"][k] = info["insertive"][1]
            log["tick_cubes"][k, 0, :3], log["tick_cubes"][k, 0, 3:] = rp, rq
            log["tick_cubes"][k, 1, :3], log["tick_cubes"][k, 1, 3:] = ip, iq
            log["tick_compute_s"][k] = time.monotonic() - tick_started
            st["ticks"] = k + 1
            if not execute and k % 5 == 0:
                print(f"t={count / 500:5.1f}s action {np.round(action, 2).tolist()} target-wrist "
                      f"{np.round((target[0] - wpos) * 1000, 1).tolist()} mm gripper {'close' if close else 'open'} "
                      f"cube ages {info['receptive'][0]:.2f}/{info['insertive'][0]:.2f}s", flush=True)

        while count < n_cycles:
            host_before, robot_t, q, qd, _ = ct.read_new_state(receive, last_robot_time,
                                                              max_interval_s=limits["max_state_interval_s"])
            last_robot_time = robot_t
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise SafetyStop("Non-finite robot state")
            W, wpos, wquat = wrist_matrix(q)
            log["q"][count], log["qd"][count], log["host_t"][count], log["robot_t"][count] = q, qd, host_before, robot_t
            if st["target"] is None:                       # first cycle: a target must exist before any torque
                policy_step(count, host_before, q, W, wpos, wquat)
            target_pos, target_quat = st["target"]
            log["target_pos"][count], log["target_quat"][count] = target_pos, target_quat
            if execute:
                if np.any(q < j_lo) or np.any(q > j_hi):
                    raise SafetyStop("Joint angle outside the trained range + margin")
                if not in_workspace(wpos):
                    raise SafetyStop("Wrist left the workspace limits")
                J = kin.compute_jacobian_calibrated(q)
                twist = J @ qd
                if np.linalg.norm(twist[:3]) > limits["tcp_speed_stop_m_s"]:
                    raise SafetyStop(f"Wrist speed {np.linalg.norm(twist[:3]):.2f} m/s above the stop limit")
                if np.linalg.norm(twist[3:]) > limits["tcp_angular_speed_stop_rad_s"]:
                    raise SafetyStop(f"Wrist angular speed {np.linalg.norm(twist[3:]):.2f} rad/s above the stop limit")
                osc.set_target(target_pos, target_quat)
                torque = osc.compute(wpos, wquat, twist, J)
                if not np.isfinite(torque).all():
                    raise SafetyStop("Non-finite torque")
                if time.monotonic() - host_before > ct.STATE_TIMEOUT_S:
                    raise SafetyStop("Robot state is more than 20 ms old before the torque command")
                torque_started = True
                if not control.directTorque(torque.tolist(), **zero):
                    raise RuntimeError("directTorque returned failure")
                log["torque"][count] = torque
            if count % POLICY_CYCLES == 0 and count > 0:   # after this cycle's torque: new target from the next cycle
                policy_step(count, host_before, q, W, wpos, wquat)
            count += 1
        summary["stop_reason"] = "max_episode_s"
    except KeyboardInterrupt:
        summary["stop_reason"] = "operator Ctrl-C"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        summary["stop_reason"] = failure
        print("STOP:", failure, flush=True)
    finally:
        if control is not None:
            if torque_started:   # same torque-to-hold handoff as the collector
                try:
                    control.directTorque([0.0] * 6, **zero)
                    control.servoJ(receive.getActualQ(), 0.5, 0.5, 0.1, 0.1, 300)
                    control.servoStop()
                except Exception as exc:
                    summary["events"].append(f"cleanup: {exc}")
            for call in (control.stopScript, control.disconnect):
                try:
                    call()
                except Exception as exc:
                    summary["events"].append(f"cleanup: {exc}")
        if receive is not None:
            receive.disconnect()
        os.sched_setscheduler(0, *initial_scheduler)
        if gc_enabled:
            gc.enable()
        ticks = st["ticks"]
        summary.update(cycles=count, policy_steps=ticks, failure=failure,
                       max_tick_compute_ms=float(log["tick_compute_s"][:ticks].max() * 1000) if ticks else None,
                       timing=ct.timing_summary(log["robot_t"][:count].tolist()) if count else None)
        np.savez_compressed(out_dir / "log.npz", **{k: v[:count] if v.shape[0] == n_cycles else v[:ticks]
                                                    for k, v in log.items()})
        print("saved", out_dir / "log.npz", json.dumps({k: summary[k] for k in ("stop_reason", "cycles", "policy_steps")}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--config", type=Path, default=HERE / "collection.state_policy.json")
    p.add_argument("--camera-transform", type=Path, required=True,
                   help="4x4 T with p_base = T @ p_camera (L515 color optical frame), .npy or .json key T_base_camera")
    p.add_argument("--camera-transform-frame", choices=F.FRAMES, required=True,
                   help="base_link (sim / our kinematics) or ur_base (UR controller Base: getActualTCPPose / pendant)")
    p.add_argument("--output", type=Path, required=True, help="New directory for summary.json and log.npz")
    p.add_argument("--seconds", type=float, default=30.0, help="track / check duration")
    p.add_argument("--allow-start-outside-training", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
