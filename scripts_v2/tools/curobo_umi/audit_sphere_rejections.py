"""Attribute the existing sphere-check comparison to individual link pairs.

CPU geometry audit only: reuses the saved configurations, sphere fit, collision
exclusions, and 1 mm radius padding. Does not change a model or restart mapping.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from reachability_geometry import ROOT, SourceCollisionChecker, batch_fk, pose_matrix


root = ROOT / '20_dense_reachability'
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['inflated', 'raw'], default='inflated')
args = parser.parse_args()
source = SourceCollisionChecker(ROOT / '15_table_reachability/model')
data = np.load(root / 'collision_validation_samples.npz')
q, hull_clear = data['q'], data['source_clear']
robot_config = yaml.safe_load((root / 'robot_open_conservative.yml').read_text())
config = robot_config['robot_cfg']['kinematics']
if args.model == 'raw':
    # The completed build report corresponds to the exported YAML. The separate
    # progress file contains an unfinished later refinement and is not used.
    completed = json.loads((source.model_dir / 'collision_model_audit.json').read_text())
    for name, old in config['collision_spheres'].items():
        raw = completed['coverage'][name]['raw_spheres']
        assert len(old) == len(raw)
        assert np.allclose([s['center'] for s in old], [s['center'] for s in raw])
        config['collision_spheres'][name] = copy.deepcopy(raw)
    (root / 'robot_open_raw_fit_candidate.yml').write_text(yaml.safe_dump(robot_config, sort_keys=False))
fk = batch_fk(source.audit, q)
spheres = {}
for shape in source.shapes:
    name = shape['name']
    values = config['collision_spheres'][name]
    centers = np.array([s['center'] for s in values])
    radii = np.array([s['radius'] for s in values]) + .001
    transform = fk[shape['body']]
    positions = np.einsum('bij,nj->bni', transform[:, :3, :3], centers) + transform[:, None, :3, 3]
    spheres[name] = (positions, radii)

base_pair = frozenset(('base_link_shape_0', 'upper_arm_link_shape_0'))
any_self = np.zeros(len(q), bool)
other_self = np.zeros(len(q), bool)
base_hits = np.zeros(len(q), bool)
pair_rows = []
for i, j in source.pairs:
    a, b = source.shapes[i]['name'], source.shapes[j]['name']
    pa, ra = spheres[a]
    pb, rb = spheres[b]
    overlap = (ra[None, :, None] + rb[None, None, :] -
               np.linalg.norm(pa[:, :, None] - pb[:, None, :], axis=-1)).max(axis=(1, 2))
    hit = overlap > 0
    any_self |= hit
    if frozenset((a, b)) == base_pair:
        base_hits = hit
    else:
        other_self |= hit
    if (hit & hull_clear).any():
        pair_rows.append(dict(a=a, b=b, hull_clear_configurations_rejected=int((hit & hull_clear).sum()),
                              all_configurations_rejected=int(hit.sum()),
                              maximum_padded_sphere_overlap_on_hull_clear_m=float(overlap[hull_clear].max())))

world = yaml.safe_load((source.model_dir / 'lab_scene.yml').read_text())['cuboid']
any_world = np.zeros(len(q), bool)
world_rows = []
for name, (positions, radii) in spheres.items():
    if name == 'base_link_shape_0':
        continue  # Same fixed-base/world handling as the original comparison.
    for obstacle_name, obstacle in world.items():
        transform = pose_matrix(obstacle['pose'][:3], obstacle['pose'][3:])
        local = (positions - transform[:3, 3]) @ transform[:3, :3]
        outside = np.abs(local) - np.array(obstacle['dims']) / 2
        distance = np.linalg.norm(np.maximum(outside, 0), axis=-1) + np.minimum(outside.max(axis=-1), 0)
        hit = (distance < radii[None]).any(axis=-1)
        any_world |= hit
        if (hit & hull_clear).any():
            world_rows.append(dict(robot_shape=name, obstacle=obstacle_name,
                                   hull_clear_configurations_rejected=int((hit & hull_clear).sum())))

previous = json.loads((root / 'collision_validation.json').read_text())
sphere_clear = ~any_self & ~any_world
report = dict(model=args.model, sample_count=len(q), hull_clear_count=int(hull_clear.sum()),
    total_spheres=sum(len(s) for s in config['collision_spheres'].values()),
    sphere_clear_count=int(sphere_clear.sum()),
    hull_clear_accepted=int((sphere_clear & hull_clear).sum()),
    hull_clear_rejected=int((~sphere_clear & hull_clear).sum()),
    hull_rejected_but_sphere_accepted=int((sphere_clear & ~hull_clear).sum()),
    self_rejected_count=int(any_self.sum()), world_rejected_count=int(any_world.sum()),
    agrees_with_original_gpu_self_count=int(any_self.sum()) == previous['sphere_self_collision_count'],
    agrees_with_original_gpu_world_count=int(any_world.sum()) == previous['sphere_world_collision_count'],
    base_upper_rejected_among_hull_clear=int((base_hits & hull_clear).sum()),
    other_self_rejected_among_hull_clear=int((other_self & hull_clear).sum()),
    world_rejected_among_hull_clear=int((any_world & hull_clear).sum()),
    clear_if_only_base_upper_pair_ignored=int((~other_self & ~any_world & hull_clear).sum()),
    still_rejected_if_only_base_upper_pair_ignored=int(((other_self | any_world) & hull_clear).sum()),
    self_pairs=sorted(pair_rows, key=lambda d: -d['hull_clear_configurations_rejected']),
    world_pairs=sorted(world_rows, key=lambda d: -d['hull_clear_configurations_rejected']),
    scope='Existing 2048 random configurations only; raw candidate is separate from the active model. No map or controller run, refitting, or exclusion changes.',
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'collision_validation_samples.npz', root/'robot_open_conservative.yml',
        source.model_dir/'collision_model_audit.json']},
    padding_note='The original comparison adds 1 mm to every radius: 2 mm combined padding for a sphere pair, 1 mm against world geometry.')
if args.model == 'inflated':
    assert report['agrees_with_original_gpu_self_count']
    assert report['agrees_with_original_gpu_world_count']
stem = 'sphere_pair_attribution' if args.model == 'inflated' else 'raw_sphere_pair_attribution'
(root / (stem+'.json')).write_text(json.dumps(report, indent=2) + '\n')
np.savez_compressed(root/(stem+'.npz'),q=q,sphere_clear=sphere_clear,hull_clear=hull_clear,
                    self_rejected=any_self,world_rejected=any_world)
print(json.dumps({k:v for k,v in report.items() if k not in ['self_pairs','world_pairs','input_hashes']}, indent=2))
