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
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError(args.output)
    config, offsets = st.make_plan(json.loads(args.config.read_text()), 'off')
    from rtde_receive import RTDEReceiveInterface
    ct.check_realtime_permissions()
    initial_scheduler = os.sched_getscheduler(0), os.sched_getparam(0)
    gc_initial = gc.isenabled()
    receive = None
    rows = []
    failure = None
    cleanup_errors = []
    workers = {}
    loop_priority = None
    print('READ-ONLY: RTDE state receive + controller math; zero robot-control commands', flush=True)
    try:
        if gc_initial:
            gc.collect()
        before = ct.thread_schedule_snapshot()
        with ct.fifo_thread_creation(ct.RT_RECEIVE_PRIORITY):
            receive = RTDEReceiveInterface(config['robot_ip'], 500, rt_priority=ct.RT_RECEIVE_PRIORITY)
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
        for offset in offsets:
            host, stamp, q, qd, after = ct.read_new_state(receive, last, require_continuity=False)
            last = stamp
            torque, target_pos, target_quat = ct.torque_for_offset(osc, q, qd, center_pos, center_quat, offset)
            if not np.isfinite(torque).all():
                raise RuntimeError('Non-finite diagnostic torque calculation')
            end = time.monotonic()
            rows.append((q.copy(), qd.copy(), torque.copy(), target_pos.copy(), target_quat.copy(),
                         host, end, stamp, after))
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if receive is not None:
            try:
                receive.disconnect()
            except Exception as exc:
                cleanup_errors.append(f'receive disconnect: {exc}')
        os.sched_setscheduler(0, *initial_scheduler)
        if gc_initial:
            gc.enable()
        else:
            gc.disable()
    times = [r[7] for r in rows]
    compute = np.array([r[6]-r[5] for r in rows])
    summary = {'source_kind': 'receive_only_timing_audit', 'robot_control_commands_sent': 0,
               'robot_ip': config['robot_ip'], 'kernel': platform.release(),
               'requested_samples': len(offsets), 'completed': len(rows) == len(offsets) and failure is None,
               'failure': failure, 'cleanup_errors': cleanup_errors,
               'rt_receive_workers': workers, 'loop_priority': loop_priority,
               'timing_summary': ct.timing_summary(times),
               'controller_compute_max_s': float(compute.max()) if len(compute) else None,
               'controller_compute_over_2ms': int(np.sum(compute > .002)),
               'collector_sha256': hashlib.sha256(Path(ct.__file__).read_bytes()).hexdigest(),
               'limitation': 'This audits receive delivery and math under current load. It does not exercise RTDE torque submission or establish motion safety.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    if rows:
        np.savez_compressed(args.output.with_suffix('.npz'), source_kind='receive_only_timing_audit',
                            robot_sample_times_s=np.array(times),
                            host_sample_times_s=np.array([r[5] for r in rows]),
                            host_compute_end_times_s=np.array([r[6] for r in rows]),
                            robot_after_read_times_s=np.array([r[8] for r in rows]))
    print(json.dumps(summary, indent=2))
    print('Saved diagnostic:', args.output)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
