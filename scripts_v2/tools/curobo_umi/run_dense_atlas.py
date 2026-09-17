"""Map every configured target using cuRobo IK and source-convex validation.

Workers own disjoint orientation/height slices and atomically save each slice.
All controller gains, simulation dynamics, and active assets are read-only.
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

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose
from reachability_geometry import ARM_NAMES, ROOT, SourceCollisionChecker, batch_fk, pose_matrix


def atomic_json(path: Path, data: dict) -> None:
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit-slices',type=int,default=0)
    parser.add_argument('--grid',choices=['both','fine','broad'],default='both')
    parser.add_argument('--output-name',default='endpoint_slices')
    args=parser.parse_args()
    root=ROOT/'20_dense_reachability'
    cfg_path=root/'atlas_config.json';cfg=json.loads(cfg_path.read_text())
    for path,digest in cfg['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    config_sha=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    output=root/args.output_name;output.mkdir(exist_ok=True)
    checker=SourceCollisionChecker(ROOT/'15_table_reachability/model',clearance=cfg['minimum_source_hull_clearance_m'])
    ik=InverseKinematics(InverseKinematicsCfg.create(robot=cfg['robot_config'],
        self_collision_check=False,load_collision_spheres=False,num_seeds=cfg['ik_seeds'],
        seed_solver_num_seeds=cfg['ik_seeds'],max_batch_size=cfg['batch_size'],
        position_tolerance=cfg['ik_position_tolerance_m'],orientation_tolerance=cfg['ik_rotation_tolerance_rad'],
        random_seed=cfg['random_seed']+args.worker,use_cuda_graph=True))
    assert ik.kinematics.joint_names==ARM_NAMES
    world_T_base=pose_matrix(cfg['robot_base_position_world_m'],cfg['robot_base_quaternion_world_wxyz'])
    base_T_world=np.linalg.inv(world_T_base)
    jobs=[]
    for name in ['fine','broad']:
        if args.grid not in ['both',name]:continue
        grid=cfg[name]
        for height_index,_ in enumerate(grid['height_above_table_m']):
            for orientation_index,_ in enumerate(grid['orientations']):
                jobs.append((name,height_index,orientation_index))
    jobs=[job for i,job in enumerate(jobs) if i%args.workers==args.worker]
    if args.limit_slices:jobs=jobs[:args.limit_slices]
    progress_path=output/f'worker_{args.worker}_progress.json'
    started=time.perf_counter();totals={str(k):0 for k in range(1,5)}
    finished=0;processed=0;ik_seconds=0.;geometry_seconds=0.
    print('START',args.worker,len(jobs),'slices',flush=True)
    for name,hi,oi in jobs:
        filename=f'{name}_h{hi:02d}_o{oi:03d}'
        target=output/(filename+'.npz');meta=output/(filename+'.json')
        if target.exists() and meta.exists():
            previous=json.loads(meta.read_text())
            assert previous['config_sha256']==config_sha
            for key,value in previous['status_counts'].items():totals[key]+=value
            processed+=previous['count'];finished+=1
            continue
        before_slice=time.perf_counter()
        grid=cfg[name];xs=np.asarray(grid['x_world_m']);ys=np.asarray(grid['y_world_m'])
        xx,yy=np.meshgrid(xs,ys,indexing='xy')
        count=xx.size
        world_p=np.c_[xx.ravel(),yy.ravel(),np.full(count,cfg['tabletop_z_m']+grid['height_above_table_m'][hi])]
        orientation=grid['orientations'][oi]
        world_r=Rotation.from_quat(np.asarray(orientation['quaternion_world_wxyz'])[[1,2,3,0]]).as_matrix()
        local_p=world_p@base_T_world[:3,:3].T+base_T_world[:3,3]
        local_r=base_T_world[:3,:3]@world_r
        local_quat=Rotation.from_matrix(local_r).as_quat()[[3,0,1,2]]
        transforms=np.broadcast_to(np.eye(4),(count,4,4)).copy()
        transforms[:,:3,:3]=local_r;transforms[:,:3,3]=local_p
        before=time.perf_counter()
        hand_clear=checker.check_target_hand(transforms)
        geometry_seconds+=time.perf_counter()-before
        status=np.where(hand_clear,2,1).astype(np.uint8)
        chosen=np.full((count,6),np.nan,dtype=np.float32)
        best=np.full((count,6),np.nan,dtype=np.float32)
        ik_counts=np.zeros(count,dtype=np.uint8)
        checked=np.zeros(count,dtype=np.uint8)
        p_error=np.full(count,np.nan,dtype=np.float32)
        r_error=np.full(count,np.nan,dtype=np.float32)
        failures={}
        indices=np.flatnonzero(hand_clear)
        for start in range(0,len(indices),cfg['batch_size']):
            rows=indices[start:start+cfg['batch_size']]
            target_pose=GoalToolPose.from_poses({cfg['tool_frame']:Pose(
                position=torch.as_tensor(local_p[rows],device='cuda',dtype=torch.float32),
                quaternion=torch.as_tensor(np.tile(local_quat,(len(rows),1)),device='cuda',dtype=torch.float32))},num_goalset=1)
            before=time.perf_counter()
            result=ik.solve_pose(target_pose,return_seeds=cfg['ik_seeds'])
            names=result.js_solution.joint_names
            q=result.js_solution.position.detach().cpu().numpy()[...,[names.index(n) for n in ARM_NAMES]]
            success=result.success.detach().cpu().numpy()
            ik_seconds+=time.perf_counter()-before
            assert q.shape==(len(rows),cfg['ik_seeds'],6),q.shape
            ik_counts[rows]=success.sum(axis=1)
            has_ik=success.any(axis=1)
            status[rows[has_ik]]=3
            first=success.argmax(axis=1)
            best[rows[has_ik]]=q[np.flatnonzero(has_ik),first[has_ik]]
            accepted=np.zeros(len(rows),dtype=bool)
            before=time.perf_counter()
            for branch in range(cfg['ik_seeds']):
                active=np.flatnonzero(success[:,branch]&~accepted)
                if not len(active):continue
                collision=checker.evaluate(q[active,branch],details=True)
                checked[rows[active]]+=1
                valid=collision['clear']
                ok=active[valid]
                accepted[ok]=True;status[rows[ok]]=4;chosen[rows[ok]]=q[ok,branch]
                for fail in collision['first_failure']:
                    if fail is not None:
                        key=fail['kind']+':'+fail['a']+' / '+fail['b']
                        failures[key]=failures.get(key,0)+1
            if accepted.any():
                fk=batch_fk(checker.audit,chosen[rows[accepted]])[cfg['tool_frame']]
                p_error[rows[accepted]]=np.linalg.norm(fk[:,:3,3]-local_p[rows[accepted]],axis=1)
                r_error[rows[accepted]]=Rotation.from_matrix(np.transpose(fk[:,:3,:3],(0,2,1))@local_r).magnitude()
                assert float(np.max(p_error[rows[accepted]])) <= cfg['ik_position_tolerance_m']+2e-6
                assert float(np.max(r_error[rows[accepted]])) <= cfg['ik_rotation_tolerance_rad']+2e-6
            geometry_seconds+=time.perf_counter()-before
        assert (status!=0).all()
        with target.with_suffix('.npz.tmp').open('wb') as file:
            np.savez_compressed(file,status=status.reshape(len(ys),len(xs)),
                chosen_q=chosen.reshape(len(ys),len(xs),6),best_ik_q=best.reshape(len(ys),len(xs),6),
                ik_solution_count=ik_counts.reshape(len(ys),len(xs)),checked_configurations=checked.reshape(len(ys),len(xs)),
                position_error_m=p_error.reshape(len(ys),len(xs)),rotation_error_rad=r_error.reshape(len(ys),len(xs)))
        os.replace(target.with_suffix('.npz.tmp'),target)
        counts={str(k):int((status==k).sum()) for k in range(1,5)}
        summary=dict(grid=name,height_index=hi,height_above_table_m=grid['height_above_table_m'][hi],
            orientation_index=oi,orientation=orientation,count=count,status_counts=counts,
            checked_configurations=int(checked.astype(np.int64).sum()),
            elapsed_s=time.perf_counter()-before_slice,worker=args.worker,
            maximum_position_error_m=float(np.nanmax(p_error)) if (status==4).any() else None,
            maximum_rotation_error_rad=float(np.nanmax(r_error)) if (status==4).any() else None,
            collision_rejection_counts=failures,config_sha256=config_sha,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_json(meta,summary)
        for key,value in counts.items():totals[key]+=value
        finished+=1;processed+=count
        progress=dict(worker=args.worker,pid=os.getpid(),finished_slices=finished,total_slices=len(jobs),
            processed_targets=processed,status_counts=totals,elapsed_s=time.perf_counter()-started,
            ik_s=ik_seconds,geometry_s=geometry_seconds,last_slice=filename,complete=False)
        atomic_json(progress_path,progress)
        print('SLICE',filename,'counts',counts,'seconds',round(summary['elapsed_s'],3),
            'progress',finished,'/',len(jobs),flush=True)
    progress=dict(worker=args.worker,pid=os.getpid(),finished_slices=finished,total_slices=len(jobs),
        processed_targets=processed,status_counts=totals,elapsed_s=time.perf_counter()-started,
        ik_s=ik_seconds,geometry_s=geometry_seconds,complete=True)
    atomic_json(progress_path,progress)
    print('COMPLETE',json.dumps(progress),flush=True)


if __name__=='__main__':main()
