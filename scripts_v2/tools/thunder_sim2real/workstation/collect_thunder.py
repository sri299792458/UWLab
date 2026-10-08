"""Standalone Thunder recorder. Without --execute this only prints a motion plan.

The vendored UWLab OSC is unmodified. Hardware-specific parameters are installed
in memory. Recorded joint positions follow UWLab's pre-command convention.

Timing: the loop is locked to the robot's 500 Hz RTDE state stream (one fresh, consistent state per robot cycle)
instead of a host timer, and uses ur_rtde's recommended real-time priorities (examples/py/realtime_control_example.py).
UWLab's collector paces with initPeriod/waitPeriod at normal priority; the host and robot clocks drift, so it reads
some robot states twice and skips others (42-82 duplicate robot timestamps per 8 s Thunder recording).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from vendor import ur5e_kinematics as kin

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "thunder_calibration.json"
# ur_rtde 1.6.5 control script with its direct_torque friction-scale bug fixed (vendor/ur_rtde_1_6_5/PROVENANCE.json).
FIXED_CONTROL_SCRIPT = HERE / "vendor/ur_rtde_1_6_5/rtde_control_fixed.script"
SCRIPT_START_TIMEOUT_S = 5.0
DT = 1 / 500
REQUIRED_RTDE_VERSION = "1.6.5"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# ur_rtde realtime_control_example priorities: receive thread, control thread, this control loop.
RT_RECEIVE_PRIORITY, RT_CONTROL_PRIORITY, RT_APP_PRIORITY = 90, 85, 80
# Only these outputs are consumed here, including the torque-to-hold handoff.
# The SDK's empty/default recipe also streams many unused fields and registers.
RT_RECEIVE_VARIABLES = ("timestamp", "actual_q", "actual_qd")
STATE_TIMEOUT_S = 0.02
STATE_POLL_S = 5e-5
# Match records.validate_record: fixed-step replay cannot use skipped robot cycles.
STATE_INTERVAL_TOLERANCE_S = 0.0004
CONTROLLER_WARMUP_ITERATIONS = 16
STEP_MAX_STATE_INTERVAL_S = 0.006
STEP_MAX_MISSING_CYCLES = 5  # At most 10 ms missing across a comparison run.


class RecordingBuffer:
    """Fixed storage touched and locked before connecting; no growing per-sample array/list log."""
    COLUMNS = (slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 21), slice(21, 25),
               25, 26, 27, 28, 29, 30)

    def __init__(self, capacity):
        self.data = np.empty((capacity, 31), dtype=np.float64)
        self.data.fill(0)  # Fault in the writable pages before the control loop.
        self.count = 0
        self.locked = self.was_locked = False
        self.libc = ctypes.CDLL(None, use_errno=True)
        for name in ("mlock", "munlock"):
            function = getattr(self.libc, name)
            function.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            function.restype = ctypes.c_int

    def lock(self):
        if self.libc.mlock(self.data.ctypes.data, self.data.nbytes):
            raise RuntimeError(f"Cannot lock {self.data.nbytes} bytes of recording storage: "
                               f"{os.strerror(ctypes.get_errno())}. Check the memlock limit before running")
        self.locked = self.was_locked = True

    def unlock(self):
        if self.locked:
            if self.libc.munlock(self.data.ctypes.data, self.data.nbytes):
                raise RuntimeError(f"Cannot unlock recording storage: {os.strerror(ctypes.get_errno())}")
            self.locked = False

    def append(self, q, qd, torque, target_pos, target_quat, host, command, robot, after, returned):
        row = self.data[self.count]
        row[:6], row[6:12], row[12:18] = q, qd, torque
        row[18:21], row[21:25] = target_pos, target_quat
        row[25], row[26], row[27], row[28], row[29] = host, command, robot, after, returned
        row[30] = time.monotonic()
        self.count += 1

    def __len__(self):
        return self.count

    def column(self, index):
        return self.data[:self.count, self.COLUMNS[index]]


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


def set_fifo_priority(priority):
    """Require the requested FIFO priority; never silently run at normal priority."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        if os.sched_getscheduler(0) != os.SCHED_FIFO or os.sched_getparam(0).sched_priority != priority:
            raise OSError("Scheduler did not apply the requested priority")
        return f"SCHED_FIFO {priority}"
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"Real-time scheduling required: cannot set SCHED_FIFO {priority}. "
                           "Use the configured Thunder real-time launcher / rtprio limits.") from exc


def set_app_realtime_priority():
    return set_fifo_priority(RT_APP_PRIORITY)


def check_realtime_permissions():
    """Probe all requested priorities and restore scheduling before any robot connection."""
    previous = os.sched_getscheduler(0), os.sched_getparam(0)
    try:
        for priority in (RT_RECEIVE_PRIORITY, RT_CONTROL_PRIORITY, RT_APP_PRIORITY):
            set_fifo_priority(priority)
    finally:
        os.sched_setscheduler(0, *previous)


@contextmanager
def fifo_thread_creation(priority):
    """Make SDK workers inherit FIFO even when the SDK skips its PREEMPT_RT-only setup."""
    previous = os.sched_getscheduler(0), os.sched_getparam(0)
    try:
        set_fifo_priority(priority)
        yield
    finally:
        os.sched_setscheduler(0, *previous)


def thread_schedule_snapshot():
    threads = {}
    for task in Path('/proc/self/task').iterdir():
        tid = int(task.name)
        try:
            threads[tid] = {"policy": os.sched_getscheduler(tid),
                            "priority": os.sched_getparam(tid).sched_priority}
        except ProcessLookupError:
            pass
    return threads


def require_new_fifo_thread(before, priority, interface):
    """Check the SDK's actual worker scheduler, rather than just its requested argument."""
    workers = {tid: state for tid, state in thread_schedule_snapshot().items() if tid not in before}
    if not any(state == {"policy": os.SCHED_FIFO, "priority": priority} for state in workers.values()):
        raise RuntimeError(f"{interface} has no new SCHED_FIFO {priority} worker: {workers}")
    return workers


def read_new_state(receive, last_robot_time, *, require_continuity=True, max_interval_s=DT):
    """Wait for a robot state newer than last_robot_time and return a consistent snapshot.

    The robot timestamp is read before and after Q/Qd; if the robot published in between, the newer state is read
    again, so position and velocity always come from the same robot cycle and no robot cycle is recorded twice.
    By default, skipped cycles abort before another target is sent; a fresh state alone is not enough for replay.
    Step-response comparisons may explicitly allow bounded, whole-cycle intervals; fixed-step fitting remains strict.
    Receive-only timing audits may count all gaps with require_continuity=False.
    """
    wait_started = time.monotonic()
    deadline = wait_started + STATE_TIMEOUT_S
    crossed_reads = 0
    while True:
        robot_before = receive.getTimestamp()
        if last_robot_time is None or robot_before > last_robot_time:
            host_before = time.monotonic()
            q = np.asarray(receive.getActualQ())
            qd = np.asarray(receive.getActualQd())
            robot_after = receive.getTimestamp()
            if robot_after == robot_before:
                if require_continuity and last_robot_time is not None:
                    interval = robot_before - last_robot_time
                    cycles = round(interval / DT)
                    if (cycles < 1 or abs(interval - cycles * DT) > STATE_INTERVAL_TOLERANCE_S
                            or interval > max_interval_s + STATE_INTERVAL_TOLERANCE_S):
                        expectation = ("expected 2 ms (+/- 0.4 ms). Fixed-step collection cannot skip robot cycles"
                                       if max_interval_s == DT else
                                       f"comparison allows at most {max_interval_s * 1000:g} ms "
                                       "(+/- 0.4 ms) in whole robot cycles")
                        error = RuntimeError(f"Robot state interval {interval * 1000:.3f} ms; {expectation}")
                        error.timing_details = {"previous_robot_time_s": float(last_robot_time),
                                                "observed_robot_time_s": float(robot_before),
                                                "interval_s": float(interval),
                                                "host_wait_started_s": wait_started,
                                                "host_state_observed_s": host_before,
                                                "crossed_reads_during_wait": crossed_reads}
                        raise error
                return host_before, robot_before, q, qd, robot_after
            crossed_reads += 1
        if time.monotonic() > deadline:
            raise RuntimeError("No new robot state within 20 ms; RTDE state stream stalled")
        time.sleep(STATE_POLL_S)


def timing_summary(robot_times):
    intervals = np.diff(np.asarray(robot_times, dtype=float))
    return {"samples": len(robot_times), "robot_span_s": float(robot_times[-1] - robot_times[0]) if robot_times else 0.,
            "duplicate_intervals": int((intervals < 1e-4).sum()), "intervals_over_2p4ms": int((intervals > 0.0024).sum()),
            "max_interval_s": float(intervals.max()) if len(intervals) else 0.}


def friction_scales(config):
    """UR direct_torque friction-compensation scales: zeros (disabled, the fitted setting) unless the config gives
    explicit per-joint viscous_scale and coulomb_scale lists in [0, 1] under "direct_torque_params"."""
    params = config.get("direct_torque_params")
    if params is None:
        return {"viscous_scale": [0.0] * 6, "coulomb_scale": [0.0] * 6}
    if set(params) != {"viscous_scale", "coulomb_scale"}:
        raise ValueError("direct_torque_params needs exactly viscous_scale and coulomb_scale")
    out = {}
    for key in ("viscous_scale", "coulomb_scale"):
        value = np.asarray(params[key], dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"direct_torque_params.{key} must be 6 numbers in [0, 1]")
        out[key] = value.tolist()
    return out


def quat_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def torque_for_offset(osc, q, qd, center_pos, center_quat, offset):
    """Shared computation for controller warmup, collection and read-only timing audits."""
    pos, quat = kin.get_ee_pose(q)
    jacobian = kin.compute_jacobian_calibrated(q)
    target_pos = center_pos + offset[:3]
    target_quat = quat_multiply(kin.axis_angle_to_quat(offset[3:]), center_quat)
    osc.set_target(target_pos, target_quat)
    torque = osc.compute(pos, quat, jacobian @ qd, jacobian)
    return torque, target_pos, target_quat


def warmup_controller(osc, q, qd, center_pos, center_quat):
    """Warm computation only; no robot command is issued."""
    offset = np.full(6, 0.001)
    for _ in range(CONTROLLER_WARMUP_ITERATIONS):
        torque, _, _ = torque_for_offset(osc, q, qd, center_pos, center_quat, offset)
        if not np.isfinite(torque).all():
            raise RuntimeError("Non-finite torque during offline controller warmup")
    osc.set_target(center_pos, center_quat)


def collect(config, offsets, output, anchors=None, *, allow_small_gaps=False):
    """anchors (optional, one int per sample): 0 keep the current target anchor, 1 re-anchor to the measured flange pose of
    this sample (policy-style relative targets), 2 re-anchor to the initial center pose. Without anchors every target is
    center + offset, as in the sysid recordings."""
    # Importing this module and planning never open a robot connection.
    import torch
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    version = importlib.metadata.version("ur-rtde")
    if version != REQUIRED_RTDE_VERSION:
        raise RuntimeError(f"Expected Thunder collector ur-rtde {REQUIRED_RTDE_VERSION}, found {version}")
    if allow_small_gaps and config.get("step_test", {}).get("mode") not in ("held", "policy"):
        raise ValueError("Small gaps are allowed only for step-response comparisons, not sysid collection")
    check_realtime_permissions()
    initial_scheduler = os.sched_getscheduler(0), os.sched_getparam(0)
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
    # In 1.6.5 these explicit scales replace friction_comp=False. Never use
    # the SDK's nonzero defaults: the fitted controller disables friction compensation.
    # A config may request explicit robot-side compensation scales (step-response comparison only;
    # records.validate_record still rejects such records for fitting).
    direct_torque_params = friction_scales(config)
    compensation_requested = any(any(v) for v in direct_torque_params.values())
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=int)
        if anchors.shape != (len(offsets),) or not np.isin(anchors, (0, 1, 2)).all():
            raise ValueError("anchors must hold one code in {0, 1, 2} per sample")
    control_script = {"source": "ur_rtde compiled-in"}
    control = receive = None
    app_priority = "not requested"
    rt_threads = {}
    rows = None
    initial_q = None
    completed = False
    torque_started = False
    failure = None
    timing_failure = None
    gap_events = []
    missed_cycles = 0
    cleanup_errors = []
    gc_enabled_on_entry = gc.isenabled()
    gc_suspended = False
    try:
        # Torch imports and the growing recording can trigger a generation-2
        # collection taking tens of milliseconds. Collect before connecting,
        # then defer automatic cyclic GC only while torque control is active.
        if gc_enabled_on_entry:
            gc.collect()
        rows = RecordingBuffer(len(offsets))
        rows.lock()
        before_receive = thread_schedule_snapshot()
        with fifo_thread_creation(RT_RECEIVE_PRIORITY):
            receive = RTDEReceiveInterface(config["robot_ip"], 500, variables=list(RT_RECEIVE_VARIABLES),
                                           rt_priority=RT_RECEIVE_PRIORITY)
        rt_threads["receive"] = require_new_fifo_thread(before_receive, RT_RECEIVE_PRIORITY, "RTDEReceiveInterface")
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
        before_control = thread_schedule_snapshot()
        with fifo_thread_creation(RT_CONTROL_PRIORITY):
            control = RTDEControlInterface(
                config["robot_ip"], 500,
                RTDEControlInterface.FLAG_VERBOSE | RTDEControlInterface.FLAG_UPLOAD_SCRIPT,
                rt_priority=RT_CONTROL_PRIORITY,
            )
        rt_threads["control"] = require_new_fifo_thread(before_control, RT_CONTROL_PRIORITY, "RTDEControlInterface")
        if compensation_requested:
            # The stock 1.6.5 script silently drops nonzero friction scales; upload the two-line fix and wait for it.
            control.setCustomScriptFile(str(FIXED_CONTROL_SCRIPT))
            deadline = time.monotonic() + SCRIPT_START_TIMEOUT_S
            while not control.isProgramRunning():
                if time.monotonic() > deadline:
                    raise RuntimeError("Fixed control script did not start")
                time.sleep(0.01)
            control_script = {"source": "vendor fixed script", "path": str(FIXED_CONTROL_SCRIPT),
                              "sha256": hashlib.sha256(FIXED_CONTROL_SCRIPT.read_bytes()).hexdigest()}
        # setPayload goes through the running control script, so it also confirms a re-uploaded script responds.
        if not control.setPayload(config["payload_mass_kg"], config["payload_cog_m"]):
            raise RuntimeError("setPayload failed")
        gc.disable()
        gc_suspended = True
        app_priority = set_app_realtime_priority()
        warmup_controller(osc, initial_q, initial_qd, center_pos, center_quat)
        print(f"Timing mode: robot_state_locked; loop priority: {app_priority}", flush=True)
        # Anchor after all setup/warmup/output. The first recorded state must
        # also be fresh, rather than a cached frame already near its 2 ms deadline.
        last_robot_time = float(receive.getTimestamp())
        if not np.isfinite(last_robot_time):
            raise RuntimeError("Invalid robot timestamp before collection")
        anchor_pos, anchor_quat = center_pos, center_quat
        for index, offset in enumerate(offsets):
            # One fresh, consistent robot state per cycle; the command follows immediately.
            host_before, robot_before, q, qd, robot_after = read_new_state(
                receive, last_robot_time,
                max_interval_s=STEP_MAX_STATE_INTERVAL_S if allow_small_gaps else DT)
            missing = max(0, round((robot_before - last_robot_time) / DT) - 1)
            if missing:
                missed_cycles += missing
                event = {"sample_index": len(rows), "missing_cycles": missing,
                         "previous_robot_time_s": last_robot_time, "observed_robot_time_s": robot_before,
                         "interval_s": robot_before - last_robot_time}
                if missed_cycles > STEP_MAX_MISSING_CYCLES:
                    error = RuntimeError(f"Step-response missed-cycle budget exceeded: {missed_cycles * DT * 1000:g} ms "
                                         f"> {STEP_MAX_MISSING_CYCLES * DT * 1000:g} ms; no further test command sent")
                    error.timing_details = {**event, "missed_cycles": missed_cycles}
                    raise error
                gap_events.append(event)
            last_robot_time = robot_before
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise RuntimeError("Received non-finite robot state")
            if np.any(np.abs(q - initial_q) > config["joint_excursion_limit_rad"]):
                raise RuntimeError("Configured joint excursion exceeded")
            if anchors is not None and anchors[index] == 1:
                anchor_pos, anchor_quat = kin.get_ee_pose(q)
            elif anchors is not None and anchors[index] == 2:
                anchor_pos, anchor_quat = center_pos, center_quat
            torque, target_pos, target_quat = torque_for_offset(osc, q, qd, anchor_pos, anchor_quat, offset)
            if not np.isfinite(torque).all():
                raise RuntimeError("Non-finite torque command")
            command_time = time.monotonic()
            if command_time - host_before > STATE_TIMEOUT_S:
                raise RuntimeError("Robot state is more than 20 ms old before torque command; host loop stalled")
            torque_started = True
            if not control.directTorque(torque.tolist(), **direct_torque_params):
                raise RuntimeError("directTorque returned failure")
            command_return_time = time.monotonic()
            rows.append(q, qd, torque, target_pos, target_quat,
                        host_before, command_time, robot_before, robot_after, command_return_time)
        completed = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        timing_failure = getattr(exc, "timing_details", None)
        if timing_failure is not None and rows:
            timing_failure["recorded_samples"] = len(rows)
            timing_failure["last_record_end_s"] = float(rows.column(10)[-1])
        raise
    finally:
        if control is not None:
            # Same torque-to-hold handoff used by the pinned UWLab collector.
            if torque_started:
                try:
                    control.directTorque([0.0] * 6, **direct_torque_params)
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
        try:
            os.sched_setscheduler(0, *initial_scheduler)
        except OSError as exc:
            cleanup_errors.append(f"Restore application scheduler: {exc}")
        # Restore GC after torque-to-hold, disconnect and scheduler cleanup,
        # before allocating tensors or serializing the saved record.
        if gc_enabled_on_entry:
            gc.enable()
        else:
            gc.disable()
        if rows is not None:
            try:
                rows.unlock()
            except RuntimeError as exc:
                cleanup_errors.append(str(exc))
        if rows:
            tensor = lambda index: torch.tensor(rows.column(index), dtype=torch.float64)
            record = {
                "schema_version": 1, "robot": "thunder", "source_kind": "real_robot",
                "joint_names": JOINT_NAMES, "sample_phase": "pre_command",
                "pose_frame": "base_link", "quaternion_order": "wxyz",
                "dt": DT, "control_freq": 500, "completed": completed,
                "failure": failure, "cleanup_errors": cleanup_errors,
                "timing_failure": timing_failure,
                "state_gap_policy": {"mode": "step_response_bounded" if allow_small_gaps else "strict",
                                     "max_interval_s": STEP_MAX_STATE_INTERVAL_S if allow_small_gaps else DT,
                                     "max_missing_cycles": STEP_MAX_MISSING_CYCLES if allow_small_gaps else 0},
                "gap_events": gap_events,
                "joint_positions": tensor(0), "joint_velocities": tensor(1),
                "joint_torques": tensor(2), "initial_joint_pos": tensor(0)[0].clone(),
                "initial_joint_vel": tensor(1)[0].clone(),
                "waypoint_target_pos": tensor(3), "waypoint_target_quat": tensor(4),
                "waypoint_step_indices": torch.arange(len(rows)), "num_waypoints": len(rows),
                "host_sample_times_s": tensor(5), "host_command_times_s": tensor(6),
                "host_command_return_times_s": tensor(9),
                "host_record_end_times_s": tensor(10),
                "robot_sample_times_s": tensor(7), "robot_after_read_times_s": tensor(8),
                "osc_params": {k: config[k] for k in
                               ("motion_stiffness", "motion_damping_ratio", "torque_max")},
                "collection_config": config, "calibration": calibration,
                "calibration_sha256": hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
                "controller_provenance": json.loads((HERE / "vendor/PROVENANCE.json").read_text()),
                "ur_rtde_version": version,
                "direct_torque_params": direct_torque_params,
                "control_script": control_script,
                "target_mode": "center" if anchors is None else "anchored",
                **({} if anchors is None else {"target_anchors": torch.tensor(anchors[:len(rows)])}),
                "timing_mode": "robot_state_locked",
                "rt_priorities": {"receive": RT_RECEIVE_PRIORITY, "control": RT_CONTROL_PRIORITY, "loop": app_priority},
                "rt_threads": rt_threads,
                "rt_receive_variables": list(RT_RECEIVE_VARIABLES),
                "gc_control": {"automatic_gc_suspended": gc_suspended,
                               "enabled_on_entry": gc_enabled_on_entry},
                "recording_storage": {"kind": "preallocated_numpy", "capacity_samples": len(offsets),
                                      "bytes": rows.data.nbytes, "prefaulted": True,
                                      "memory_locked_during_collection": rows.was_locked},
                "startup": {"controller_warmup_iterations": CONTROLLER_WARMUP_ITERATIONS,
                            "first_sample_waited_for_new_frame": True},
                "timing_summary": timing_summary(rows.column(7).tolist()),
            }
            torch.save(record, output)
            print(f"Saved {len(rows)} samples to {output}; completed={completed}")
            print("Timing:", json.dumps({**record["timing_summary"], "loop_priority": app_priority}))
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
