"""Validate the cuRobo sphere checks and independent source-hull classifier."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from reachability_geometry import ROOT, SourceCollisionChecker, batch_fk, exact_pair_gap, fcl, set_transform
from benchmark_dense_ik import prepare_config


def main() -> None:
    out=ROOT/'20_dense_reachability'
    source=SourceCollisionChecker(ROOT/'15_table_reachability/model',clearance=.001)
    config=prepare_config(source)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=config,scene_model=str(source.model_dir/'lab_scene.yml'),
        collision_activation_distance=0.,self_collision_activation_distance=0.))
    rng=np.random.default_rng(2718)
    q=rng.uniform(source.joint_limits[0],source.joint_limits[1],size=(2048,6))
    exact=source.evaluate(q,details=True)
    q_cuda=torch.as_tensor(q,device='cuda',dtype=torch.float32)[:,None]
    state=checker.get_kinematics(q_cuda)
    spheres=state.robot_spheres.clone()
    spheres[...,3]+=.001
    self_cost=checker.get_self_collision_distance(spheres).reshape(len(q),-1).amax(dim=1)
    base_indices=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link_shape_0')
    world_spheres=spheres.clone();world_spheres[:,:,base_indices,3]=-100.
    # The base is fixed: its non-mount world pairs were checked individually by
    # SourceCollisionChecker. It remains fully enabled for self collision.
    world_per_sphere=checker.get_collision_constraint(world_spheres).reshape(len(q),-1)
    world_cost=world_per_sphere.amax(dim=1)
    sphere_clear=((self_cost<=0)&(world_cost<=0)).cpu().numpy()
    false_free=np.flatnonzero(sphere_clear&~exact['clear'])
    conservative_rejections=np.flatnonzero(~sphere_clear&exact['clear'])
    print('SPHERE COMPARISON',int(sphere_clear.sum()),int(exact['clear'].sum()),len(false_free),
        'world colliding',int((world_cost>0).sum()),'self colliding',int((self_cost>0).sum()),flush=True)
    kparams=checker.kinematics.config.kinematics_config
    detailed_sphere_cases=[]
    for row in np.flatnonzero(exact['clear'])[:4]:
        local=[]
        for shape in source.shapes:
            indices=kparams.get_sphere_index_from_link_name(shape['name'])
            cost=float(world_per_sphere[row,indices].max())
            if cost>0:local.append(dict(shape=shape['name'],world_cost=cost))
        detailed_sphere_cases.append(dict(row=int(row),world=local,self_cost=float(self_cost[row])))
    print('SPHERE DETAILS',detailed_sphere_cases,flush=True)
    # Compare broad-phase filtering to an exhaustive check of all relevant
    # source-hull pairs, including geometrically distant pairs.
    brute=[]
    for row in range(24):
        poses=batch_fk(source.audit,q[row:row+1])
        for i,shape in enumerate(source.shapes):set_transform(source.objects[i],poses[shape['body']][0])
        valid=True
        for i,j in source.pairs:
            if exact_pair_gap(source.objects[i],source.objects[j])<source.clearance:
                valid=False;break
        if valid:
            for i,j in np.argwhere(source.world_mask):
                if exact_pair_gap(source.objects[i],source.world_objects[j])<source.clearance:
                    valid=False;break
        brute.append(valid)
    brute_disagreement=np.flatnonzero(np.asarray(brute)!=exact['clear'][:len(brute)])
    box=fcl.Box(.06,.06,.06)
    zero=fcl.CollisionObject(box)
    positive=fcl.CollisionObject(box,fcl.Transform(np.eye(3),np.array([.08,0,0])))
    negative=fcl.CollisionObject(box,fcl.Transform(np.eye(3),np.array([.05,0,0])))
    analytical=[exact_pair_gap(zero,positive,signed=True),exact_pair_gap(zero,negative,signed=True)]
    camera_pairs=[]
    camera_index=next(i for i,s in enumerate(source.shapes) if s['name']=='camera_mount_and_body')
    for other_name in ['wrist_1_link_shape_0','wrist_2_link_shape_0','forearm_link_shape_0','upper_arm_link_shape_0']:
        other_index=next(i for i,s in enumerate(source.shapes) if s['name']==other_name)
        pair=tuple(sorted((camera_index,other_index)))
        assert pair in map(tuple,source.pairs)
        camera_pairs.append(dict(a='camera_mount_and_body',b=other_name,checked=True))
    examples=[]
    for flag,label in [(sphere_clear,'clear_in_both'),(~exact['clear'],'source_collision'),
                       (~sphere_clear&exact['clear'],'sphere_only_rejection')]:
        for row in np.flatnonzero(flag)[:4]:
            examples.append(dict(label=label,row=int(row),q=q[row].tolist(),first_failure=exact['first_failure'][row]))
    report=dict(random_configurations=len(q),source_clear=int(exact['clear'].sum()),sphere_clear=int(sphere_clear.sum()),
        sphere_false_free_count=len(false_free),sphere_only_rejections=len(conservative_rejections),
        sphere_world_collision_count=int((world_cost>0).sum()),sphere_self_collision_count=int((self_cost>0).sum()),
        sphere_diagnostics=detailed_sphere_cases,
        brute_force_cases=len(brute),broad_phase_disagreement_count=len(brute_disagreement),
        analytical_box_gaps_m=analytical,clearance_m=source.clearance,
        base_world_handling='Fixed mounting pair exempt; other fixed-base/world pairs checked once with FCL; base retained in self checks',
        base_world_checks=source.invariant_world_audit,
        self_exclusions=source.exclusions,invariant_hand_checks=source.invariant_pair_audit,
        required_camera_pairs=camera_pairs,examples=examples,
        numerical_tolerances='1 mm minimum hull clearance; no claim of physical safety certification',
        passed=bool(not len(false_free) and not len(brute_disagreement) and np.allclose(analytical,[.02,-.01],atol=1e-8)))
    np.savez_compressed(out/'collision_validation_samples.npz',q=q,source_clear=exact['clear'],sphere_clear=sphere_clear)
    (out/'collision_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['base_world_checks','self_exclusions','invariant_hand_checks','examples']},indent=2),flush=True)
    assert report['passed'],report


if __name__=='__main__':main()
