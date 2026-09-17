"""Uniform full-table pose sampling: cuRobo IK + native GPU sphere acceptance.

Uses the user-accepted raw sphere fit and comparison's unchanged radius margin
and pair masks. No FCL acceptance, physics steps, approach sweep, or payload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import GoalToolPose, Pose
from reachability_geometry import ROOT, ARM_NAMES, batch_fk, pose_matrix

OUT=ROOT/'49_clearance_and_placement/atlas'

def atomic_json(path, data):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'20_dense_reachability'
    old=json.loads((previous/'atlas_config.json').read_text())
    candidate=previous/'robot_open_raw_fit_candidate.yml'
    model=OUT/'robot_open_spheres.yml'
    model.write_bytes(candidate.read_bytes())
    verification=json.loads((previous/'raw_sphere_gpu_verification.json').read_text())
    assert hashlib.sha256(model.read_bytes()).hexdigest()==verification['candidate_sha256']
    keep=['table_bounds_world_m','tabletop_z_m','tool_frame','tool_offset',
          'robot_base_position_world_m','robot_base_quaternion_world_wxyz',
          'open_hand_joint_values','joint_names','joint_limits_rad',
          'ik_position_tolerance_m','ik_rotation_tolerance_rad','ik_seeds','batch_size','random_seed']
    cfg={k:old[k] for k in keep}
    cfg.update(version=2,robot_model=str(model),scene_model=str(ROOT/'15_table_reachability/model/lab_scene.yml'),
        x_world_m=(old['broad']['x_world_m'][0]+np.arange(149)*.01).tolist(),
        y_world_m=(old['broad']['y_world_m'][0]+np.arange(73)*.01).tolist(),
        height_above_table_m=(np.arange(61)*.01).tolist(),orientations=old['broad']['orientations'],
        grid_spacing_m=.01,column_clearance_m=.05,column_box_names=[f'lab_{i}' for i in range(7,15)],
        radius_padding_m=.001,world_ignored_robot_link='base_link_shape_0',
        status_codes={'2':'no_converged_IK_found','3':'all_converged_IK_candidates_sphere_blocked','4':'sphere_clear_IK_found'},
        method='32-seed cuRobo pose IK followed by native cuRobo GPU self/world sphere collision checks.',
        collision_optimization_in_ik=False,
        known_comparison={'tested_configurations':2048,'sphere_clear_hull_blocked':1,
            'sphere_blocked_hull_clear':157,'known_missed_hull_overlap_m':.004960164919341728,
            'user_accepted_for_map':True,'source_report':str(previous/'raw_sphere_gpu_verification.json')},
        modeling_notes=[
            'Full tabletop XY sampled at exactly 10 mm; X stops 5 mm before its upper edge. Heights 0–600 mm in 10 mm increments. Identical 504 orientations at all positions.',
            'Empty fully open hand, all six finger coordinates locked at zero; no cubes or other payload.',
            'Original fitted 239 spheres, without the abandoned link-wide radius inflation.',
            'Self-pair padding remains 2 mm combined. World clearance is 50 mm for the eight vertical column boxes and 1 mm for other lab geometry.',
            'Same self-pair exclusions and fixed-base/world handling as the accepted comparison.',
            'World uses all 33 lab boxes. Fixed base sphere group is omitted from world queries because of its mounting interface; its pose is invariant.',
            'Collision tests determine acceptance after pose IK, and do not guide the IK optimization.',
            'No solution found describes this finite IK search; it does not prove geometric impossibility.',
            'Orientation fraction is a fraction of these samples, not an orientation-uniform probability.',
            'Endpoint map only: no approach-path, dynamics, gains, payload, or hardware validation.',
        ])
    scene=yaml.safe_load(Path(cfg['scene_model']).read_text())
    column_scene=OUT/'columns.yml'
    column_scene.write_text(yaml.safe_dump({'cuboid':{k:scene['cuboid'][k] for k in cfg['column_box_names']}}))
    cfg['column_scene_model']=str(column_scene)
    cfg['shape']=[len(cfg[k]) for k in ['height_above_table_m','orientations','y_world_m','x_world_m']]
    cfg['total_targets']=int(np.prod(cfg['shape']))
    paths=[model,column_scene,Path(cfg['scene_model']),ROOT/'15_table_reachability/model/export_audit.json',
        ROOT/'15_table_reachability/model/thunder_umi.urdf',previous/'raw_sphere_gpu_verification.json',
        Path(__file__),Path(__file__).with_name('reachability_geometry.py')]
    cfg['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    atomic_json(OUT/'config.json',cfg)
    print(json.dumps({k:cfg[k] for k in ['shape','total_targets','ik_seeds','radius_padding_m']},indent=2))

def main(args):
    cfg_path=OUT/'config.json';cfg=json.loads(cfg_path.read_text())
    for p,digest in cfg['input_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    sha=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    model=yaml.safe_load(Path(cfg['robot_model']).read_text())['robot_cfg']
    audit=json.loads((ROOT/'15_table_reachability/model/export_audit.json').read_text())
    ik=InverseKinematics(InverseKinematicsCfg.create(robot=model,
        self_collision_check=False,load_collision_spheres=False,num_seeds=cfg['ik_seeds'],
        seed_solver_num_seeds=cfg['ik_seeds'],max_batch_size=cfg['batch_size'],
        position_tolerance=cfg['ik_position_tolerance_m'],orientation_tolerance=cfg['ik_rotation_tolerance_rad'],
        random_seed=cfg['random_seed']+args.worker,use_cuda_graph=True))
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=model,scene_model=cfg['scene_model'],collision_activation_distance=0.,self_collision_activation_distance=0.))
    column_checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=model,scene_model=cfg['column_scene_model'],collision_activation_distance=0.,self_collision_activation_distance=0.))
    assert ik.kinematics.joint_names==checker.kinematics.joint_names==ARM_NAMES
    base_ids=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name(cfg['world_ignored_robot_link'])
    base_T_world=np.linalg.inv(pose_matrix(cfg['robot_base_position_world_m'],cfg['robot_base_quaternion_world_wxyz']))
    output=OUT/('pilot' if args.pilot else 'slices');output.mkdir(exist_ok=True)
    jobs=[(h,o) for h in range(cfg['shape'][0]) for o in range(cfg['shape'][1])]
    if args.pilot:
        jobs=[(h,o) for h,o in [(0,0),(0,251),(10,0),(30,0),(30,251),(60,0)]]
    else:jobs=jobs[args.worker::args.workers]
    xx,yy=np.meshgrid(cfg['x_world_m'],cfg['y_world_m'],indexing='xy');count=xx.size
    totals={str(k):0 for k in [2,3,4]};started=time.perf_counter();finished=0;ik_s=0.;collision_s=0.
    progress_path=output/f'worker_{args.worker}_progress.json'
    for hi,oi in jobs:
        stem=f'h{hi:02d}_o{oi:03d}';target=output/(stem+'.npz');meta=output/(stem+'.json')
        if target.exists() and meta.exists():
            prev=json.loads(meta.read_text());assert prev['config_sha256']==sha
            for key,value in prev['status_counts'].items():totals[key]+=value
            finished+=1
            continue
        before_slice=time.perf_counter()
        world_p=np.c_[xx.ravel(),yy.ravel(),np.full(count,cfg['tabletop_z_m']+cfg['height_above_table_m'][hi])]
        local_p=world_p@base_T_world[:3,:3].T+base_T_world[:3,3]
        orientation=cfg['orientations'][oi]
        world_r=Rotation.from_quat(np.asarray(orientation['quaternion_world_wxyz'])[[1,2,3,0]]).as_matrix()
        local_r=base_T_world[:3,:3]@world_r
        local_quat=Rotation.from_matrix(local_r).as_quat()[[3,0,1,2]]
        status=np.full(count,2,np.uint8);chosen=np.full((count,6),np.nan,np.float32)
        ik_count=np.zeros(count,np.uint8);clear_count=np.zeros(count,np.uint8)
        p_error=np.full(count,np.nan,np.float32);r_error=np.full(count,np.nan,np.float32)
        for start in range(0,count,cfg['batch_size']):
            end=min(start+cfg['batch_size'],count);n=end-start
            goal=GoalToolPose.from_poses({cfg['tool_frame']:Pose(
                position=torch.as_tensor(local_p[start:end],device='cuda',dtype=torch.float32),
                quaternion=torch.as_tensor(np.tile(local_quat,(n,1)),device='cuda',dtype=torch.float32))},num_goalset=1)
            before=time.perf_counter();result=ik.solve_pose(goal,return_seeds=cfg['ik_seeds']);torch.cuda.synchronize()
            ik_s+=time.perf_counter()-before
            names=result.js_solution.joint_names
            q=result.js_solution.position[..., [names.index(name) for name in ARM_NAMES]];success=result.success
            assert q.shape==(n,cfg['ik_seeds'],6),q.shape
            before=time.perf_counter()
            spheres=checker.get_kinematics(q).robot_spheres.clone();spheres[...,3]+=cfg['radius_padding_m']
            assert spheres.shape[-2:]==(239,4),spheres.shape
            self_bad=checker.get_self_collision_distance(spheres).reshape(n,cfg['ik_seeds'],-1).amax(dim=2)>0
            spheres[:,:,base_ids,3]=-100.
            world_bad=checker.get_collision_constraint(spheres).reshape(n,cfg['ik_seeds'],-1).amax(dim=2)>0
            spheres[...,3]+=cfg['column_clearance_m']-cfg['radius_padding_m']
            column_bad=column_checker.get_collision_constraint(spheres).reshape(n,cfg['ik_seeds'],-1).amax(dim=2)>0
            clear=success&~self_bad&~world_bad&~column_bad
            has=success.any(dim=1).cpu().numpy();good=clear.any(dim=1).cpu().numpy()
            first=clear.to(torch.int64).argmax(dim=1)
            selected=q[torch.arange(n,device=q.device),first].detach().cpu().numpy()
            ik_count[start:end]=success.sum(dim=1).cpu().numpy()
            clear_count[start:end]=clear.sum(dim=1).cpu().numpy()
            collision_s+=time.perf_counter()-before
            rows=np.arange(start,end);status[rows[has]]=3;status[rows[good]]=4
            chosen[rows[good]]=selected[good]
        good=status==4
        if good.any():
            fk=batch_fk(audit,chosen[good])[cfg['tool_frame']]
            p_error[good]=np.linalg.norm(fk[:,:3,3]-local_p[good],axis=1)
            r_error[good]=Rotation.from_matrix(np.transpose(fk[:,:3,:3],(0,2,1))@local_r).magnitude()
            assert p_error[good].max()<=cfg['ik_position_tolerance_m']+2e-6
            assert r_error[good].max()<=cfg['ik_rotation_tolerance_rad']+2e-6
        shape=xx.shape
        with target.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f,status=status.reshape(shape),chosen_q=chosen.reshape(*shape,6),
                ik_solution_count=ik_count.reshape(shape),clear_solution_count=clear_count.reshape(shape),
                position_error_m=p_error.reshape(shape),rotation_error_rad=r_error.reshape(shape))
        os.replace(target.with_suffix('.npz.tmp'),target)
        counts={str(k):int((status==k).sum()) for k in [2,3,4]}
        atomic_json(meta,dict(height_index=hi,orientation_index=oi,count=count,status_counts=counts,
            config_sha256=sha,elapsed_s=time.perf_counter()-before_slice,worker=args.worker,
            max_position_error_m=float(p_error[good].max()) if good.any() else None,
            max_rotation_error_rad=float(r_error[good].max()) if good.any() else None))
        for key,value in counts.items():totals[key]+=value
        finished+=1
        progress=dict(worker=args.worker,pid=os.getpid(),finished_slices=finished,total_slices=len(jobs),
            processed_targets=finished*count,status_counts=totals,elapsed_s=time.perf_counter()-started,
            ik_s=ik_s,collision_s=collision_s,complete=finished==len(jobs))
        atomic_json(progress_path,progress)
        if args.pilot or finished%20==0:print(json.dumps(progress),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=7);args=parser.parse_args()
    if args.prepare:prepare()
    else:main(args)
