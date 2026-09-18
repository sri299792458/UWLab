"""Verify every recorded row and the identity of freshly merged clearance banks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917')) / '49_clearance_and_placement'


def leaves(node, prefix=''):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, prefix + '/' + key)
    else:
        yield prefix, torch.stack(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    production = json.loads((args.root / 'production_runs.json').read_text())
    assert production['complete'] and len(production['merged']) == 4
    config = json.loads((args.root / 'atlas/config.json').read_text())
    report = dict(all_passed=False, total_count=0, families={},
                  method='All saved rows checked without filtering: finite tensors, unique states, unit quaternions, corrected robot root pose, and merge hashes. Native geometry validation is separate.')
    exclusions = {record['family']: record for path in args.root.glob('*_native_recheck_exclusions.json')
                  for record in [json.loads(path.read_text())]}
    for family, merged in production['merged'].items():
        path = Path(merged['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        exclusion = exclusions.get(family)
        expected_count = merged['count']
        if exclusion:
            assert merged['sha256'] == exclusion['original_sha256']
            assert digest == exclusion['final_sha256']
            expected_count = exclusion['after_count']
        else:
            assert digest == merged['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        tensors = dict(leaves(data))
        count = len(next(iter(tensors.values())))
        assert count >= production['target_per_family'] and count == expected_count
        assert all(len(v) == count and bool(torch.isfinite(v).all()) for v in tensors.values())
        flat = torch.cat([v.reshape(count, -1) for v in tensors.values()], dim=1)
        unique = len(torch.unique(flat, dim=0))
        assert unique == count, (family, 'Duplicate saved states', unique, count)
        errors = {k: float((v[:, 3:7].norm(dim=1)-1).abs().max())
                  for k, v in tensors.items() if k.endswith('/root_pose')}
        assert errors and max(errors.values()) < 1e-4
        root = tensors['/initial_state/articulation/robot/root_pose']
        expected = torch.tensor(config['robot_base_quaternion_world_wxyz'], dtype=root.dtype)
        dots = (root[:, 3:7] @ expected).abs() / (root[:, 3:7].norm(dim=1)*expected.norm())
        mount_error = float((1-dots).abs().max())
        position_error = float((root[:, :3]-torch.tensor(config['robot_base_position_world_m'])).abs().max())
        # Offline resets retain the upstream +/-10 mm root-position jitter.
        # The mount quaternion is fixed; position is checked against that range.
        assert mount_error < 1e-6 and position_error < .01005
        report['families'][family] = dict(path=str(path), sha256=digest, count=count, unique=unique,
            all_finite=True, quaternion_max_norm_error=errors, robot_mount_quaternion_dot_error=mount_error,
            robot_root_jitter_max_m=position_error,
            tensor_shapes={k: list(v.shape) for k, v in tensors.items()})
        report['total_count'] += count
    report['all_passed'] = True
    report['native_recheck_exclusions'] = exclusions
    (args.root / 'dataset_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(all_passed=True, total_count=report['total_count'],
        counts={k: v['count'] for k, v in report['families'].items()})))


if __name__ == '__main__':
    main()
