"""Check sampled vertical approaches for stored dense-map configurations.

Build an outward 10 cm path from each selected endpoint and reverse it for an
approach. This establishes a local sampled geometric path, not reachability from
a home pose, continuous-time collision certification, or dynamic tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose
from reachability_geometry import ARM_NAMES, ROOT, SourceCollisionChecker, batch_fk, pose_matrix
from run_dense_atlas import atomic_json


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit-slices',type=int,default=0)
    parser.add_argument('--output-name',default='approach_slices')
    args=parser.parse_args()
    root=ROOT/'20_dense_reachability';cfg_path=root/'atlas_config.json'
    cfg=json.loads(cfg_path.read_text());config_sha=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    for path,digest in cfg['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    output=root/args.output_name;output.mkdir(exist_ok=True)
    checker=SourceCollisionChecker(ROOT/'15_table_reachability/model',clearance=cfg['minimum_source_hull_clearance_m'])
    ik=InverseKinematics(InverseKinematicsCfg.create(robot=cfg['robot_config'],self_collision_check=False,
        load_collision_spheres=False,num_seeds=16,seed_solver_num_seeds=16,max_batch_size=cfg['batch_size'],
        position_tolerance=cfg['ik_position_tolerance_m'],orientation_tolerance=cfg['ik_rotation_tolerance_rad'],
        random_seed=3709+args.worker,use_cuda_graph=True))
    inverse=np.linalg.inv(pose_matrix(cfg['robot_base_position_world_m'],cfg['robot_base_quaternion_world_wxyz']))
    grid=cfg['fine'];xs=np.asarray(grid['x_world_m']);ys=np.asarray(grid['y_world_m'])
    xx,yy=np.meshgrid(xs,ys,indexing='xy');count=xx.size
    jobs=[(hi,oi) for hi in range(len(grid['height_above_table_m'])) for oi in range(len(grid['orientations']))]
    jobs=[job for i,job in enumerate(jobs) if i%args.workers==args.worker]
    if args.limit_slices:jobs=jobs[:args.limit_slices]
    steps=10;maximum_joint_step=.20;interpolation_joint_step=.02;maximum_line_deviation=.002
    started=time.perf_counter();finished=0;totals={str(k):0 for k in range(7)}
    print('START APPROACH',args.worker,len(jobs),flush=True)
    for hi,oi in jobs:
        name=f'fine_h{hi:02d}_o{oi:03d}';target=output/(name+'.npz');meta=output/(name+'.json')
        endpoint_path=root/'endpoint_slices'/(name+'.npz')
        if target.exists() and meta.exists():
            previous=json.loads(meta.read_text());assert previous['config_sha256']==config_sha
            assert previous['endpoint_sha256']==hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
            for key,value in previous['status_counts'].items():totals[key]+=value
            finished+=1;continue
        assert endpoint_path.exists(),endpoint_path
        before_slice=time.perf_counter();endpoint=np.load(endpoint_path)
        endpoint_status=endpoint['status'].ravel();q_initial=endpoint['chosen_q'].reshape(-1,6)
        status=np.zeros(count,dtype=np.uint8);status[endpoint_status==4]=1
        path=np.full((count,steps+1,6),np.nan,dtype=np.float32);path[:,0]=q_initial
        reached=np.zeros(count,dtype=np.uint8)
        max_p_error=np.zeros(count,dtype=np.float32);max_r_error=np.zeros(count,dtype=np.float32)
        max_line_error=np.zeros(count,dtype=np.float32);max_joint_delta=np.zeros(count,dtype=np.float32)
        world_p=np.c_[xx.ravel(),yy.ravel(),np.full(count,cfg['tabletop_z_m']+grid['height_above_table_m'][hi])]
        world_r=Rotation.from_quat(np.asarray(grid['orientations'][oi]['quaternion_world_wxyz'])[[1,2,3,0]]).as_matrix()
        local_r=inverse[:3,:3]@world_r
        local_quat=Rotation.from_matrix(local_r).as_quat()[[3,0,1,2]]
        initial_p=world_p@inverse[:3,:3].T+inverse[:3,3]
        direction=inverse[:3,:3]@np.array([0.,0.,1.])
        exact_candidates=0;interpolated_states=0
        for step in range(1,steps+1):
            rows_all=np.flatnonzero(status==1)
            if not len(rows_all):break
            desired_p=initial_p+.01*step*direction
            previous_target=initial_p+.01*(step-1)*direction
            target_transforms=np.broadcast_to(np.eye(4),(len(rows_all),4,4)).copy()
            target_transforms[:,:3,:3]=local_r;target_transforms[:,:3,3]=desired_p[rows_all]
            hand_clear=checker.check_target_hand(target_transforms)
            status[rows_all[~hand_clear]]=2
            rows_all=rows_all[hand_clear]
            for start in range(0,len(rows_all),cfg['batch_size']):
                rows=rows_all[start:start+cfg['batch_size']]
                q_previous=path[rows,step-1].astype(np.float64)
                # Pad the entire request ourselves. The pinned cuRobo helper
                # pads some current-state fields but can leave jerk unpadded.
                # Every padded request is a duplicate and is discarded below.
                padded=np.pad(np.arange(len(rows)),(0,cfg['batch_size']-len(rows)),mode='edge')
                target_pose=GoalToolPose.from_poses({cfg['tool_frame']:Pose(
                    position=torch.as_tensor(desired_p[rows][padded],device='cuda',dtype=torch.float32),
                    quaternion=torch.as_tensor(np.tile(local_quat,(cfg['batch_size'],1)),device='cuda',dtype=torch.float32))},num_goalset=1)
                current=JointState.from_position(torch.as_tensor(q_previous[padded],device='cuda',dtype=torch.float32),joint_names=ARM_NAMES)
                result=ik.solve_pose(target_pose,current_state=current,return_seeds=16)
                names=result.js_solution.joint_names
                q=result.js_solution.position.detach().cpu().numpy()[:len(rows),...,[names.index(n) for n in ARM_NAMES]].astype(np.float64)
                success=result.success.detach().cpu().numpy()[:len(rows)]
                # Choose an equivalent angle representation near the previous
                # state only when it remains inside the actual joint limits.
                adjusted=q+2*np.pi*np.round((q_previous[:,None]-q)/(2*np.pi))
                in_bounds=(adjusted>=checker.joint_limits[0]-1e-7)&(adjusted<=checker.joint_limits[1]+1e-7)
                q=np.where(in_bounds,adjusted,q).astype(np.float32).astype(np.float64)
                delta=q-q_previous[:,None]
                jump=np.abs(delta).max(axis=-1)
                eligible=success&(jump<=maximum_joint_step)
                order=np.argsort(np.where(eligible,np.linalg.norm(delta,axis=-1),np.inf),axis=1)
                accepted=np.zeros(len(rows),dtype=bool)
                status[rows]=np.where(success.any(axis=1),5,3)
                status[rows[eligible.any(axis=1)]]=4
                for rank in range(16):
                    candidate_indices=order[:,rank]
                    active=np.flatnonzero(~accepted&eligible[np.arange(len(rows)),candidate_indices])
                    if not len(active):continue
                    candidates=q[active,candidate_indices[active]]
                    valid=checker.evaluate(candidates)['clear'];exact_candidates+=len(active)
                    survivors=np.flatnonzero(valid)
                    if not len(survivors):continue
                    chosen_rows=active[survivors];candidate=candidates[survivors]
                    previous=q_previous[chosen_rows]
                    # Validate every interior joint-interpolated state, with a
                    # maximum 0.02 rad change in any joint between checks.
                    n_segments=np.maximum(1,np.ceil(np.abs(candidate-previous).max(axis=1)/interpolation_joint_step).astype(int))
                    segment_clear=np.ones(len(candidate),dtype=bool)
                    line_error=np.zeros(len(candidate))
                    for pieces in np.unique(n_segments):
                        group=np.flatnonzero(n_segments==pieces)
                        if pieces<=1:continue
                        fractions=np.arange(1,pieces,dtype=float)/pieces
                        interpolated=previous[group,None]+fractions[None,:,None]*(candidate[group,None]-previous[group,None])
                        flat=interpolated.reshape(-1,6)
                        check=checker.evaluate(flat)['clear'].reshape(len(group),-1)
                        segment_clear[group]&=check.all(axis=1);interpolated_states+=len(flat)
                        actual=batch_fk(checker.audit,flat)[cfg['tool_frame']][:,:3,3].reshape(len(group),-1,3)
                        expected=previous_target[rows[chosen_rows[group]],None]+fractions[None,:,None]*.01*direction
                        line_error[group]=np.linalg.norm(actual-expected,axis=-1).max(axis=1)
                    segment_clear&=line_error<=maximum_line_deviation
                    successful=chosen_rows[segment_clear]
                    candidate=candidate[segment_clear]
                    accepted[successful]=True;status[rows[successful]]=1;reached[rows[successful]]=step
                    path[rows[successful],step]=candidate.astype(np.float32)
                    max_line_error[rows[successful]]=np.maximum(max_line_error[rows[successful]],line_error[segment_clear])
                    max_joint_delta[rows[successful]]=np.maximum(max_joint_delta[rows[successful]],
                        np.abs(candidate-q_previous[successful]).max(axis=1))
                    if len(successful):
                        fk=batch_fk(checker.audit,candidate)[cfg['tool_frame']]
                        pe=np.linalg.norm(fk[:,:3,3]-desired_p[rows[successful]],axis=1)
                        re=Rotation.from_matrix(np.transpose(fk[:,:3,:3],(0,2,1))@local_r).magnitude()
                        assert (pe<=cfg['ik_position_tolerance_m']+2e-6).all()
                        assert (re<=cfg['ik_rotation_tolerance_rad']+2e-6).all()
                        max_p_error[rows[successful]]=np.maximum(max_p_error[rows[successful]],pe)
                        max_r_error[rows[successful]]=np.maximum(max_r_error[rows[successful]],re)
            print('STEP',name,step,int((status==1).sum()),flush=True)
        assert ((status!=1)|(reached==steps)).all()
        with target.with_suffix('.npz.tmp').open('wb') as file:
            np.savez_compressed(file,status=status.reshape(len(ys),len(xs)),
                reached_withdrawal_steps=reached.reshape(len(ys),len(xs)),
                q_path=path.reshape(len(ys),len(xs),steps+1,6),
                max_position_error_m=max_p_error.reshape(len(ys),len(xs)),
                max_rotation_error_rad=max_r_error.reshape(len(ys),len(xs)),
                max_line_deviation_m=max_line_error.reshape(len(ys),len(xs)),
                max_joint_node_delta_rad=max_joint_delta.reshape(len(ys),len(xs)))
        os.replace(target.with_suffix('.npz.tmp'),target)
        counts={str(k):int((status==k).sum()) for k in range(7)}
        report=dict(height_index=hi,orientation_index=oi,count=count,status_counts=counts,
            elapsed_s=time.perf_counter()-before_slice,endpoint_sha256=hashlib.sha256(endpoint_path.read_bytes()).hexdigest(),
            config_sha256=config_sha,worker=args.worker,exact_candidate_checks=exact_candidates,
            interpolated_state_checks=interpolated_states,maximum_joint_node_step_rad=maximum_joint_step,
            joint_interpolation_step_rad=interpolation_joint_step,maximum_line_deviation_m=maximum_line_deviation,
            length_m=.10,cartesian_node_spacing_m=.01,
            status_codes={'0':'endpoint unavailable','1':'sampled approach verified for stored endpoint',
                '2':'hand target blocked along withdrawal','3':'no next IK found',
                '4':'candidate or joint segment rejected','5':'no sufficiently continuous IK branch found','6':'reserved'},
            scope='Local sampled path for chosen endpoint configuration, not proof that no other approach exists.',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_json(meta,report)
        for key,value in counts.items():totals[key]+=value
        finished+=1
        progress=dict(worker=args.worker,pid=os.getpid(),finished_slices=finished,total_slices=len(jobs),
            status_counts=totals,elapsed_s=time.perf_counter()-started,last_slice=name,complete=False)
        atomic_json(output/f'worker_{args.worker}_progress.json',progress)
        print('APPROACH SLICE',name,counts,'seconds',round(report['elapsed_s'],2),flush=True)
    atomic_json(output/f'worker_{args.worker}_progress.json',dict(worker=args.worker,pid=os.getpid(),
        finished_slices=finished,total_slices=len(jobs),status_counts=totals,elapsed_s=time.perf_counter()-started,complete=True))
    print('COMPLETE APPROACH',args.worker,flush=True)


if __name__=='__main__':main()
