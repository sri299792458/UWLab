"""Combine disjoint native audit reports after checking their dataset identities."""
import os
import hashlib
import json
from pathlib import Path

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911') + '/26_map_reset_regeneration')
audit = json.loads((ROOT / 'dataset_audit.json').read_text())
assert audit['all_passed']
merged = None
parts = []
for name in ('general_partial', 'air', 'resting'):
    status = json.loads((ROOT / f'native_{name}_status.json').read_text())
    assert status['status'] == 'complete' and status['passed'], status
    path = ROOT / f'native_validation_{name}.json'
    report = json.loads(path.read_text())
    assert report['all_saved_states_valid'] and not report.get('validation_error')
    assert report['dataset_dir'] == str(ROOT / 'OmniReset')
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
        assert result['saved_geometry_valid'] and result['hold_probe']['all_finite']
        merged['families'][family] = result
    parts.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      families=list(report['families'])))
assert set(merged['families']) == set(audit['families'])
merged['validation_parts'] = parts
merged['all_saved_states_valid'] = True
(ROOT / 'native_validation.json').write_text(json.dumps(merged, indent=2)+'\n')
print(json.dumps(dict(all_saved_states_valid=True,
    counts={k: v['count'] for k, v in merged['families'].items()})))
