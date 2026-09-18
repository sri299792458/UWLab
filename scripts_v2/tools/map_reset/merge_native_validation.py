"""Combine disjoint native audit reports after checking final dataset identities."""
import argparse
import os
import hashlib
import json
from pathlib import Path

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917')) / '49_clearance_and_placement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.root / 'dataset_audit.json').read_text())
    assert audit['all_passed']
    merged = None
    parts = []
    for path in args.reports:
        report = json.loads(path.read_text())
        assert report['all_saved_states_valid'] and not report.get('validation_error')
        assert Path(report['dataset_dir']) == args.root / 'OmniReset'
        if merged is None:
            merged = {k: v for k, v in report.items() if k != 'families'}
            merged['families'] = {}
        else:
            for key in ('dataset_dir', 'seed', 'kp', 'kd', 'physics_dt', 'policy_dt', 'hold_probe'):
                assert report[key] == merged[key], key
        for family, result in report['families'].items():
            assert family not in merged['families'], family
            expected = audit['families'][family]
            assert result['sha256'] == expected['sha256'] and result['count'] == expected['count']
            assert hashlib.sha256(Path(expected['path']).read_bytes()).hexdigest() == result['sha256']
            assert result['saved_geometry_valid'] and result['hold_probe']['all_finite']
            assert result['robot_mount_quaternion_dot_error'] < 1e-6
            merged['families'][family] = result
        parts.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), families=list(report['families'])))
    assert set(merged['families']) == set(audit['families'])
    merged['validation_parts'] = parts
    merged['all_saved_states_valid'] = True
    (args.root / 'native_validation.json').write_text(json.dumps(merged, indent=2)+'\n')
    print(json.dumps(dict(all_saved_states_valid=True, counts={k: v['count'] for k, v in merged['families'].items()})))


if __name__ == '__main__':
    main()
