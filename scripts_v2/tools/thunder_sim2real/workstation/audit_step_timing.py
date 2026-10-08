"""Receive-only RTDE timing audit with the full controller computation workload.

Opens only RTDEReceiveInterface. Sends no torque, motion, payload, or script commands.
It evaluates held targets in memory while observing the robot's existing state.
Its diagnostic outputs are not recordings eligible for dynamics fitting.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

import collect_thunder as ct
import step_test_thunder as st


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('held', 'policy'), default='held')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat the full computation sequence in one receive connection (1-10; no robot commands)')
    parser.add_argument('--verbose-receive', action='store_true',
                        help='Print SDK receive diagnostics, including its queued-packet discard messages (audit only)')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    config, offsets, anchors = st.make_plan(json.loads(args.config.read_text()), 'off', args.mode)
    sequence_samples = len(offsets)
    if args.repeats > 1:
        offsets = np.tile(offsets, (args.repeats, 1))
        if anchors is not None:
            anchors = np.tile(anchors, args.repeats)
    from rtde_receive import RTDEReceiveInterface
    ct.check_realtime_permissions()
    initial_scheduler = os.sched_getscheduler(0), os.sched_getparam(0)
    gc_initial = gc.isenabled()
    receive = None
    rows = None
    failure = None
    cleanup_errors = []
    workers = {}
    loop_priority = None
    print('READ-ONLY: RTDE state receive + controller math; zero robot-control commands', flush=True)
    try:
        if gc_initial:
            gc.collect()
        rows = ct.RecordingBuffer(len(offsets))
        rows.lock()
        before = ct.thread_schedule_snapshot()
        with ct.fifo_thread_creation(ct.RT_RECEIVE_PRIORITY):
            receive = RTDEReceiveInterface(config['robot_ip'], 500, variables=list(ct.RT_RECEIVE_VARIABLES),
                                           verbose=args.verbose_receive,
                                           rt_priority=ct.RT_RECEIVE_PRIORITY)
        workers = ct.require_new_fifo_thread(before, ct.RT_RECEIVE_PRIORITY, 'RTDEReceiveInterface')
        q0 = np.asarray(receive.getActualQ())
        qd0 = np.asarray(receive.getActualQd())
        if q0.shape != (6,) or qd0.shape != (6,) or not np.isfinite(q0).all() or not np.isfinite(qd0).all():
            raise RuntimeError('Invalid initial robot state')
        center_pos, center_quat = ct.kin.get_ee_pose(q0)
        osc = ct.kin.OperationalSpaceController(motion_stiffness=config['motion_stiffness'],
                                              motion_damping_ratio=config['motion_damping_ratio'],
                                              torque_max=config['torque_max'])
        gc.disable()
        loop_priority = ct.set_app_realtime_priority()
        ct.warmup_controller(osc, q0, qd0, center_pos, center_quat)
        last = float(receive.getTimestamp())
        if not np.isfinite(last):
            raise RuntimeError('Invalid initial robot timestamp')
        anchor_pos, anchor_quat = center_pos, center_quat
        for index, offset in enumerate(offsets):
            host, stamp, q, qd, after = ct.read_new_state(receive, last, require_continuity=False)
            last = stamp
            if anchors is not None and anchors[index] == 1:
                anchor_pos, anchor_quat = ct.kin.get_ee_pose(q)
            elif anchors is not None and anchors[index] == 2:
                anchor_pos, anchor_quat = center_pos, center_quat
            torque, target_pos, target_quat = ct.torque_for_offset(osc, q, qd, anchor_pos, anchor_quat, offset)
            if not np.isfinite(torque).all():
                raise RuntimeError('Non-finite diagnostic torque calculation')
            end = time.monotonic()
            rows.append(q, qd, torque, target_pos, target_quat, host, end, stamp, after, end)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if receive is not None:
            try:
                receive.disconnect()
            except Exception as exc:
                cleanup_errors.append(f'receive disconnect: {exc}')
        try:
            os.sched_setscheduler(0, *initial_scheduler)
        except OSError as exc:
            cleanup_errors.append(f'restore scheduler: {exc}')
        if gc_initial:
            gc.enable()
        else:
            gc.disable()
        if rows is not None:
            try:
                rows.unlock()
            except RuntimeError as exc:
                cleanup_errors.append(str(exc))
    times = rows.column(7).tolist() if rows else []
    compute = rows.column(6) - rows.column(5) if rows else np.array([])
    logging = rows.column(10) - rows.column(9) if rows else np.array([])
    timing = ct.timing_summary(times)
    continuous = bool(rows) and timing['duplicate_intervals'] == 0 and timing['intervals_over_2p4ms'] == 0
    summary = {'source_kind': 'receive_only_timing_audit', 'robot_control_commands_sent': 0,
               'mode': args.mode,
               'repeats': args.repeats, 'sequence_samples': sequence_samples,
               'robot_ip': config['robot_ip'], 'kernel': platform.release(),
               'requested_samples': len(offsets), 'completed': bool(rows) and len(rows) == len(offsets) and failure is None,
               'failure': failure, 'cleanup_errors': cleanup_errors,
               'continuous_500hz': continuous,
               'verbose_receive': args.verbose_receive,
               'rt_receive_workers': workers, 'loop_priority': loop_priority,
               'rt_receive_variables': list(ct.RT_RECEIVE_VARIABLES),
               'timing_summary': timing,
               'controller_compute_max_s': float(compute.max()) if len(compute) else None,
               'controller_compute_over_2ms': int(np.sum(compute > .002)),
               'recording_write_max_s': float(logging.max()) if len(logging) else None,
               'collector_sha256': hashlib.sha256(Path(ct.__file__).read_bytes()).hexdigest(),
               'limitation': 'This audits receive delivery and math under current load. It does not exercise RTDE torque submission or establish motion safety.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    if rows:
        np.savez_compressed(args.output.with_suffix('.npz'), source_kind='receive_only_timing_audit',
                            robot_sample_times_s=np.array(times),
                            host_sample_times_s=rows.column(5),
                            host_compute_end_times_s=rows.column(6),
                            host_record_end_times_s=rows.column(10),
                            robot_after_read_times_s=rows.column(8))
    print(json.dumps(summary, indent=2))
    print('Saved diagnostic:', args.output)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
