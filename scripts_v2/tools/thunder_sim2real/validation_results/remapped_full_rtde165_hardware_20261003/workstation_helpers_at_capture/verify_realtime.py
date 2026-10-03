"""Verify effective FIFO scheduling and eight seconds of read-only RTDE sampling.

Never creates an RTDE control interface or sends movement/torque commands.
"""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'UWLab/scripts_v2/tools/thunder_sim2real/workstation'
sys.path.insert(0, str(SOURCE))
import collect_thunder as collector


def main():
    from rtde_receive import RTDEReceiveInterface
    version = importlib.metadata.version('ur-rtde')
    if version != collector.REQUIRED_RTDE_VERSION:
        raise RuntimeError(f'Expected ur-rtde {collector.REQUIRED_RTDE_VERSION}, found {version}')
    collector.check_realtime_permissions()
    previous_scheduler = os.sched_getscheduler(0), os.sched_getparam(0)
    config = json.loads((HERE/'collection.json').read_text())
    offsets = collector.make_plan(config)
    collector.install_calibration()
    osc = collector.kin.OperationalSpaceController(
        motion_stiffness=np.asarray(config['motion_stiffness']),
        motion_damping_ratio=np.asarray(config['motion_damping_ratio']),
        torque_max=np.asarray(config['torque_max']))
    receive = None
    timestamps = []
    compute_times = []
    print(f'RLIMIT_RTPRIO: {resource.getrlimit(resource.RLIMIT_RTPRIO)}; FIFO priorities 90/85/80 verified.', flush=True)
    try:
        before = collector.thread_schedule_snapshot()
        with collector.fifo_thread_creation(collector.RT_RECEIVE_PRIORITY):
            receive = RTDEReceiveInterface(config['robot_ip'], 500, rt_priority=collector.RT_RECEIVE_PRIORITY)
        workers = collector.require_new_fifo_thread(before, collector.RT_RECEIVE_PRIORITY, 'RTDEReceiveInterface')
        center_pos, center_quat = collector.kin.get_ee_pose(np.asarray(receive.getActualQ()))
        app_priority = collector.set_app_realtime_priority()
        print(f'Receive workers: {workers}; loop: {app_priority}. Read-only sampling for 8 seconds.', flush=True)
        for offset in offsets:
            _, robot_time, q, qd, _ = collector.read_new_state(receive, timestamps[-1] if timestamps else None)
            begun = time.monotonic()
            pos, quat = collector.kin.get_ee_pose(q)
            jacobian = collector.kin.compute_jacobian_calibrated(q)
            target_quat = collector.quat_multiply(collector.kin.axis_angle_to_quat(offset[3:]), center_quat)
            osc.set_target(center_pos+offset[:3], target_quat)
            torque = osc.compute(pos, quat, jacobian@qd, jacobian)
            if not np.isfinite(torque).all():
                raise RuntimeError('Non-finite local controller computation')
            compute_times.append(time.monotonic()-begun)
            timestamps.append(robot_time)
    finally:
        os.sched_setscheduler(0, *previous_scheduler)
        if receive is not None:
            receive.disconnect()
    report = {
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Read-only RTDE with full-test local controller computations. No control interface, torque or movement commands.',
        'ur_rtde_version': version,
        'operator_uid': os.getuid(),
        'rtprio_limits': list(resource.getrlimit(resource.RLIMIT_RTPRIO)),
        'fifo_priorities_probe_passed': [90, 85, 80],
        'receive_workers_effective': workers,
        'loop_scheduler_effective': app_priority,
        'kernel': os.uname().release,
        'preempt_rt_kernel': Path('/sys/kernel/realtime').exists() and Path('/sys/kernel/realtime').read_text().strip() == '1',
        'timing_summary': collector.timing_summary(timestamps),
        'local_computation_max_s': max(compute_times),
        'local_computation_p99_s': float(np.percentile(compute_times, 99)),
        'hardware_commands_sent': False,
        'control_worker_effective': 'Not constructed during this read-only check. Collector verifies FIFO 85 after connecting control and before its first torque command.',
    }
    summary = report['timing_summary']
    report['passed'] = summary['samples'] == 4000 and summary['duplicate_intervals'] == 0 and summary['intervals_over_2p4ms'] == 0
    (HERE/'realtime-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('Read-only timing check has gaps; inspect realtime-verification.json')


if __name__ == '__main__':
    main()
