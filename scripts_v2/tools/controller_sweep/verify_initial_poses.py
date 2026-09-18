"""Compare recorded native initial poses to the exported articulation tree."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument('--phases', nargs='+', default=['corrected_screen', 'corrected_holdout', 'grasp'])
args = p.parse_args()
root = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
out = root / '22_explicit_gain_sweep'
audit = json.loads((root / '15_table_reachability/model/export_audit.json').read_text())
mapping = json.loads((root / '21_sphere_reachability/config.json').read_text())


def poses_to_matrix(x):
    x = np.asarray(x)
    t = np.broadcast_to(np.eye(4), x.shape[:-1] + (4, 4)).copy()
    t[..., :3, 3] = x[..., :3]
    t[..., :3, :3] = Rotation.from_quat(x[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(x.shape[:-1] + (3, 3))
    return t


results = []
for phase in args.phases:
    files = sorted((out / phase).glob('worker_*/runtime_inputs.json'))
    assert files, phase
    for file in files:
        data = json.loads(file.read_text())
        q = np.asarray(data['initial_joint_positions'])
        n = len(q)
        intended = poses_to_matrix(data['intended_root_pose_world'])
        actual = poses_to_matrix(data['initial_root_pose_world'])
        transforms = {'base_link': intended}
        for joint in audit['articulation_tree']:
            left, right, sign = np.asarray(joint['T0']), np.asarray(joint['T1']), 1
            if joint['reverse']:
                left, right, sign = right, left, -1
            turn = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if not joint['fixed']:
                axis = np.eye(3)['XYZ'.index(joint['axis'])]
                angle = q[:, data['joint_names'].index(joint['name'])]
                turn[:, :3, :3] = Rotation.from_rotvec(sign * angle[:, None] * axis).as_matrix()
            transforms[joint['child']] = transforms[joint['parent']] @ left @ turn @ np.linalg.inv(right)
        body = poses_to_matrix(data['initial_body_poses_world'])
        position_errors, rotation_errors = [], []
        for i, name in enumerate(data['body_names']):
            expected = transforms[name]
            position_errors.append(np.linalg.norm(expected[:, :3, 3] - body[:, i, :3, 3], axis=1).max())
            relative = expected[:, :3, :3].transpose(0, 2, 1) @ body[:, i, :3, :3]
            rotation_errors.append(Rotation.from_matrix(relative).magnitude().max())
        result = dict(phase=phase, worker=file.parent.name, environments=n,
            root_transform_max_abs_error=float(np.abs(actual-intended).max()),
            body_position_max_error_m=float(max(position_errors)),
            body_rotation_max_error_rad=float(max(rotation_errors)))
        if 'initial_cube_poses_world' in data:
            cube = poses_to_matrix(data['initial_cube_poses_world'])
            pp = np.asarray([v['cube_pose_world'] for v in data['poses']])
            expected = poses_to_matrix(np.tile(pp, (n // len(pp), 1)))
            expected[:, :3, 3] += intended[:, :3, 3] - np.asarray(mapping['robot_base_position_world_m'])
            result['cube_transform_max_abs_error'] = float(np.abs(expected - cube).max())
            assert result['cube_transform_max_abs_error'] < 1e-5, result
        assert result['root_transform_max_abs_error'] < 1e-5, result
        assert result['body_position_max_error_m'] < 1e-5, result
        assert result['body_rotation_max_error_rad'] < 1e-4, result
        results.append(result)
        print(result, flush=True)
(out / 'initial_pose_validation.json').write_text(json.dumps(dict(
    method='Native initial root and all body poses compared to intended world root and independent exported-USD articulation-tree forward kinematics, including all 12 joints; cube poses compared to intended transformed saved poses.',
    results=results), indent=2) + '\n')
