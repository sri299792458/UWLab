"""Check complete production banks and record data/source identity without filtering."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/26_map_reset_regeneration')
FAMILIES = ['ObjectAnywhereEEAnywhere', 'ObjectRestingEEGrasped',
            'ObjectAnywhereEEGrasped', 'ObjectPartiallyAssembledEEGrasped']


def leaves(node, prefix=''):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, prefix+'/'+k)
    else:
        yield prefix, node


def half_extents(q):
    w, x, y, z = q.unbind(-1)
    r = torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], dim=-1).reshape(-1, 3, 3)
    return .03 * r.abs().sum(dim=2)


def main():
    stages = [json.loads((ROOT / 'production' / (f+'.json')).read_text()) for f in FAMILIES]
    assert all(s['status'] == 'complete' for s in stages)
    report = dict(data_root=str(ROOT), families={}, all_passed=False)
    for stage in stages:
        path = Path(stage['path'])
        data = torch.load(path, map_location='cpu', weights_only=False)
        tensors = {k: torch.stack(v) for k, v in leaves(data)}
        count = len(next(iter(tensors.values())))
        assert count >= 10000 and count == stage['saved_count']
        assert all(len(v) == count and torch.isfinite(v).all() for v in tensors.values())
        unique = len(torch.unique(torch.cat([v.reshape(count, -1) for v in tensors.values()], dim=1), dim=0))
        assert unique == count
        quat_errors = {k: float((v[:, 3:7].norm(dim=1)-1).abs().max())
                       for k, v in tensors.items() if k.endswith('/root_pose')}
        assert max(quat_errors.values()) < 1e-4
        top = tensors['/initial_state/rigid_object/insertive_object/root_pose']
        bottom = tensors['/initial_state/rigid_object/receptive_object/root_pose']
        th, bh = half_extents(top[:, 3:]), half_extents(bottom[:, 3:])
        overlap = ((top[:, :2]-bottom[:, :2]).abs() <= th[:, :2]+bh[:, :2]).all(dim=1)
        table = torch.full((count,), .84235)
        support = torch.where(overlap, torch.maximum(table, bottom[:, 2]+bh[:, 2]), table)
        gap = top[:, 2]-th[:, 2]-support
        if stage['family'] == 'ObjectAnywhereEEGrasped':
            assert (gap > .01-1e-5).all()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == stage['cpu_sha256']
        report['families'][stage['family']] = dict(path=str(path), sha256=digest, count=count, unique=unique,
            all_finite=True, quaternion_max_norm_error=quat_errors,
            support_gap_m_quantiles=torch.quantile(gap, torch.tensor([0., .25, .5, .75, 1.])).tolist(),
            top_xyz_min=top[:, :3].amin(dim=0).tolist(), top_xyz_max=top[:, :3].amax(dim=0).tolist(),
            bottom_xyz_min=bottom[:, :3].amin(dim=0).tolist(), bottom_xyz_max=bottom[:, :3].amax(dim=0).tolist(),
            tensor_shapes={k: list(v.shape) for k, v in tensors.items()}, map_statistics=stage['map_statistics'])
    report['all_passed'] = True
    report['fresh_inputs'] = json.loads((ROOT / 'fresh_inputs.json').read_text())
    report['lookup'] = json.loads((ROOT / 'lookup_manifest.json').read_text())
    report['sphere_adapter_validation'] = json.loads((ROOT / 'sphere_adapter_validation.json').read_text())
    (ROOT / 'stage_runs.json').write_text(json.dumps(stages, indent=2)+'\n')
    (ROOT / 'dataset_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(all_passed=True, counts={k: v['count'] for k, v in report['families'].items()})))


if __name__ == '__main__':
    main()
