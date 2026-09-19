"""Compare requested joint motion with UW's pinned calibration/default pose.

UW's result is ideal inverse kinematics, not a recording from UW hardware.
The Thunder requested paths and physics responses retain their separate labels.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'workstation'))
import collect_thunder as collector


def metrics(q, dt):
    q = np.unwrap(q, axis=0)
    degrees = np.degrees(q)
    return {'minimum_deg': degrees.min(axis=0).tolist(), 'maximum_deg': degrees.max(axis=0).tolist(),
        'range_deg': np.ptp(degrees, axis=0).tolist(),
        'peak_speed_deg_s': np.degrees(np.abs(np.gradient(q, dt, axis=0)).max(axis=0)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-preview', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Load the pristine pinned UW module; do not install Thunder calibration.
    source = HERE / 'workstation/vendor/ur5e_kinematics.py'
    spec = importlib.util.spec_from_file_location('uw_original_kinematics', source)
    kin = importlib.util.module_from_spec(spec); spec.loader.exec_module(kin)
    config = json.loads((HERE / 'workstation/collection.simulation_candidate.json').read_text())
    offsets = collector.generate_offsets(config)
    q0 = np.radians([0., -90., 90., -90., -90., 0.])
    p0, r0 = kin.get_ee_pose(q0)
    q = q0.copy(); path = []; failures = []
    for sample, offset in enumerate(offsets):
        target_q = collector.quat_multiply(kin.axis_angle_to_quat(offset[3:]), r0)
        for _ in range(50):
            p, quat = kin.get_ee_pose(q)
            error = kin.compute_pose_error(p, quat, p0 + offset[:3], target_q)
            if np.linalg.norm(error[:3]) < 1e-6 and np.linalg.norm(error[3:]) < 1e-5:
                break
            jacobian = kin.compute_jacobian_calibrated(q)
            delta = np.linalg.solve(jacobian.T @ jacobian + 1e-8 * np.eye(6), jacobian.T @ error)
            q += delta * min(1., .1 / max(np.abs(delta).max(), 1e-12))
        else:
            failures.append({'sample': sample, 'error': error.tolist()})
            break
        path.append(q.copy())
    assert not failures, failures
    path = np.array(path)
    np.savez_compressed(args.output / 'uw_default_requested.npz', joint_positions=path, dt=.002)
    output = {'joint_names': collector.JOINT_NAMES, 'UW_default_requested': metrics(path, .002),
        'UW_calibration_source': str(source), 'UW_calibration_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'UW_initial_joints_deg': np.degrees(q0).tolist(), 'samples': len(path),
        'scope': 'UW default pose and original pinned UW calibration under ideal tracking. This is not UW hardware data. Requested IK speeds are not actual robot speeds.'}
    for name, preview in [('outward', args.new_preview)]:
        data = np.load(preview / 'trajectories.npz')
        output[name + '_requested'] = metrics(data['full_joint_positions'][:, 0, :6], .002)
    (args.output / 'requested_comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    for name in ['UW_default_requested', 'outward_requested']:
        print(name, 'range_deg', np.round(output[name]['range_deg'], 2).tolist(),
              'peak_deg_s', np.round(output[name]['peak_speed_deg_s'], 2).tolist(), flush=True)


if __name__ == '__main__':
    main()
