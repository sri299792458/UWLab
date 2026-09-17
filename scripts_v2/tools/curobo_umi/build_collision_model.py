"""Fit cuRobo spheres to exported Isaac hulls and conservatively cover their volume.

The fit uses cuRobo MorphIt. A separate grid cover bounds the maximum uncovered
distance: every grid cell that could intersect a source convex hull is enclosed
by a sphere after a documented radius inflation. This avoids interpreting a
sampled surface-coverage percentage as complete collision-volume coverage.
Self-collision exclusions are explicit mechanical adjacency, not random pruning.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull
import torch
import trimesh
import yaml

from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitType

ARM_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def volume_cover(mesh, spheres, pitch=.003):
    """Bound union-of-spheres coverage using an enclosing regular cell grid.

    Cells separated from the hull by a face plane are excluded. This may retain
    extra cells, but cannot remove a cell containing part of the hull. Distance
    to a sphere union is 1-Lipschitz, so center gap + half cell diagonal bounds
    the gap everywhere in the retained cell. Inflating every radius by the
    maximum bound covers the complete hull, up to numerical tolerance.
    """
    hull = ConvexHull(mesh.vertices)
    # Merge coplanar triangles to reduce the conservative box/plane test cost.
    planes = np.unique(np.round(hull.equations, decimals=9), axis=0)
    lo, hi = mesh.bounds
    count = np.maximum(1, np.ceil((hi-lo)/pitch).astype(int))
    widths = (hi-lo)/count
    axes = [lo[i]+widths[i]*(np.arange(count[i])+.5) for i in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1,3)
    half_diag = float(np.linalg.norm(widths/2))
    support = np.abs(planes[:,:3]) @ (widths/2)
    possible = []
    for block in np.array_split(centers, max(1,math.ceil(len(centers)/4096))):
        # Small positive numerical tolerance makes selection conservative.
        keep = np.all(block@planes[:,:3].T+planes[:,3] <= support+1e-7, axis=1)
        possible.append(block[keep])
    points = np.concatenate(possible)
    sphere_centers = np.asarray([s['center'] for s in spheres], dtype=float)
    radii = np.asarray([s['radius'] for s in spheres], dtype=float)
    max_gap = -np.inf
    for block in np.array_split(points, max(1,math.ceil(len(points)/4096))):
        gaps = np.linalg.norm(block[:,None]-sphere_centers[None],axis=2)-radii[None]
        max_gap = max(max_gap, float(gaps.min(axis=1).max()))
    # Keep the fit close to the source. Fill isolated gaps with extra spheres
    # instead of enlarging every sphere to cover the single worst missed corner.
    padding = .003
    covered = [dict(center=s['center'],radius=float(s['radius']+padding)) for s in spheres]
    gap = np.full(len(points),np.inf)
    for sphere in covered:
        gap=np.minimum(gap,np.linalg.norm(points-np.asarray(sphere['center']),axis=1)-sphere['radius'])
    additional=0
    while float(gap.max())+half_diag > -2e-7:
        center=points[int(np.argmax(gap))]
        interior=float(np.min(-(planes[:,:3]@center+planes[:,3])))
        radius=max(.007,interior+.003,half_diag+1e-6)
        covered.append(dict(center=center.tolist(),radius=radius))
        gap=np.minimum(gap,np.linalg.norm(points-center,axis=1)-radius)
        additional+=1
        if additional>2000:raise RuntimeError('Excessive gap spheres; refit source geometry')
    return covered, dict(grid_pitch_upper_m=pitch, cell_widths_m=widths.tolist(),
        retained_cells=len(points), max_center_gap_m=max_gap,
        half_cell_diagonal_m=half_diag, radius_inflation_m=padding,
        additional_gap_spheres=additional,final_max_cell_gap_bound_m=float(gap.max()+half_diag),
        coverage_version=2,
        full_source_convex_volume_covered=True,
        method='Hull-intersecting cell superset; nearest-sphere gap + half cell diagonal')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--cover-pitch',type=float,default=.003)
    args=p.parse_args()
    model=args.model_dir.resolve()
    audit=json.loads((model/'export_audit.json').read_text())
    torch.manual_seed(1709);np.random.seed(1709)
    results={};metrics={}
    fit_path=model/'sphere_fit_progress.json'
    if fit_path.exists():
        saved=json.loads(fit_path.read_text());results=saved['spheres'];metrics=saved['metrics']
    started=time.perf_counter()
    for c in audit['collision_shapes']:
        name=c['name']
        if name in results and metrics[name]['coverage_certificate'].get('coverage_version')==2:
            print('Reuse completed fit:',name,flush=True);continue
        mesh=trimesh.load(c['mesh'],force='mesh')
        sphere_count=max(16,min(100,math.ceil(mesh.area/.002)))
        if name=='camera_mount_and_body':sphere_count=max(sphere_count,64)
        if 'inner_finger' in name:sphere_count=max(sphere_count,32)
        if name in metrics:
            print('Refining existing sphere coverage:',name,flush=True)
            raw=metrics[name]['raw_spheres']
        else:
            print('Fitting',name,'with',sphere_count,'spheres',flush=True)
            fit=fit_spheres_to_mesh(mesh,num_spheres=sphere_count,iterations=args.iterations,
                fit_type=SphereFitType.MORPHIT,coverage_weight=2000.,protrusion_weight=10.,compute_metrics=True)
            centers=fit.centers.detach().cpu().numpy() if isinstance(fit.centers,torch.Tensor) else np.asarray(fit.centers)
            radii=fit.radii.detach().cpu().numpy() if isinstance(fit.radii,torch.Tensor) else np.asarray(fit.radii)
            raw=[dict(center=c.tolist(),radius=float(r)) for c,r in zip(centers,radii)]
        covered,certificate=volume_cover(mesh,raw,pitch=args.cover_pitch)
        results[name]=covered
        metrics[name]=dict(sphere_count=len(covered),raw_spheres=raw,coverage_certificate=certificate)
        fit_path.write_text(json.dumps(dict(spheres=results,metrics=metrics),indent=2)+'\n')
        print('Covered',name,'radius inflation mm',round(1000*certificate['radius_inflation_m'],3),flush=True)
    # Exclude only the same rigid body and bodies connected directly by a joint.
    # Do not use RobotBuilder.compute_collision_matrix: its automatic exclusions
    # include pairs colliding at the default pose and pairs absent in random samples.
    adjacent={frozenset((j['body0'],j['body1'])) for j in audit['source_joints']}
    shapes=audit['collision_shapes']
    ignores={c['name']:[] for c in shapes};pairs=[]
    for a,b in itertools.combinations(shapes,2):
        reason=None
        if a['body']==b['body']:reason='same rigid body'
        elif frozenset((a['body'],b['body'])) in adjacent:reason='direct mechanical joint adjacency'
        ignored=reason is not None
        if ignored:ignores[a['name']].append(b['name'])
        pairs.append(dict(a=a['name'],b=b['name'],body_a=a['body'],body_b=b['body'],ignored=ignored,reason=reason))
    for pair in pairs:
        if 'camera_mount_and_body' in (pair['a'],pair['b']):
            other=pair['b'] if pair['a']=='camera_mount_and_body' else pair['a']
            if other in ['wrist_1_link_shape_0','wrist_2_link_shape_0','forearm_link_shape_0','upper_arm_link_shape_0']:
                assert not pair['ignored'],f'Camera pair was incorrectly excluded: {pair}'
    fingers=[j['name'] for j in audit['articulation_tree'] if not j['fixed'] and j['name'] not in ARM_NAMES]
    names=ARM_NAMES+fingers
    kin=dict(urdf_path=audit['urdf'],asset_root_path=str(model),base_link='base_link',
        tool_frames=[audit['tool_frame']],mesh_link_names=list(results),
        collision_link_names=list(results),collision_spheres=results,
        collision_sphere_buffer=0.,self_collision_buffer={k:0. for k in results},
        self_collision_ignore=ignores,lock_joints=None,
        cspace=dict(joint_names=names,default_joint_position=[0.,-1.57,1.57,-1.57,-1.57,0.]+[0.]*len(fingers),
            cspace_distance_weight=[1.]*len(names),null_space_weight=[1.]*len(names),
            max_acceleration=12.,max_jerk=500.,position_limit_clip=0.))
    (model/'robot_full.yml').write_text(yaml.safe_dump({'robot_cfg':{'kinematics':kin}},sort_keys=False))
    kin['lock_joints']={name:0. for name in fingers}
    (model/'robot_open.yml').write_text(yaml.safe_dump({'robot_cfg':{'kinematics':kin}},sort_keys=False))
    (model/'self_collision_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
    report=dict(sphere_count=sum(map(len,results.values())),shape_count=len(results),
        elapsed_s=time.perf_counter()-started,seed=1709,fit_iterations=args.iterations,
        coverage=metrics,collision_pair_count=len(pairs),ignored_pair_count=sum(p['ignored'] for p in pairs),
        policy='Same body / directly connected mechanical bodies only. No default-pose or random-sample pruning.',
        open_finger_coordinates={name:0. for name in fingers})
    (model/'collision_model_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed',report['sphere_count'],'spheres,',len(pairs),'shape pairs,',report['elapsed_s'],'seconds',flush=True)


if __name__=='__main__':main()
