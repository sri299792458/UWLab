"""Benchmark cuRobo candidates and verify source-hull collision classifications."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import GoalToolPose, Pose

from reachability_geometry import ARM_NAMES, ROOT, SourceCollisionChecker, batch_fk, pose_matrix


def prepare_config(checker: SourceCollisionChecker) -> dict:
    config = yaml.safe_load((checker.model_dir/'robot_open.yml').read_text())['robot_cfg']
    ignored = {s['name']: [] for s in checker.shapes}
    for pair in checker.exclusions:
        ignored[pair['a']].append(pair['b'])
    config['kinematics']['self_collision_ignore'] = ignored
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seeds', type=int, default=32)
    args = parser.parse_args()
    out = ROOT/'20_dense_reachability'
    checker = SourceCollisionChecker(ROOT/'15_table_reachability/model', clearance=.001)
    config = prepare_config(checker)
    (out/'robot_open_conservative.yml').write_text(yaml.safe_dump({'robot_cfg':config},sort_keys=False))
    (out/'collision_exclusions.json').write_text(json.dumps(dict(
        self=checker.exclusions, world=[dict(a='base_link_shape_0',b='lab_18',reason='fixed mounting interface')],
        invariant_open_hand_checks=checker.invariant_pair_audit),indent=2)+'\n')
    before=time.perf_counter()
    ik = InverseKinematics(InverseKinematicsCfg.create(robot=copy.deepcopy(config),
        self_collision_check=False, load_collision_spheres=False, num_seeds=args.seeds,
        seed_solver_num_seeds=args.seeds, position_tolerance=.001, orientation_tolerance=.01,
        random_seed=1709, max_batch_size=args.batch))
    print('CREATED',ik.kinematics.joint_names,'seconds',time.perf_counter()-before,flush=True)
    assert ik.kinematics.joint_names==ARM_NAMES, ik.kinematics.joint_names
    limits=ik.kinematics.get_joint_limits().position.cpu().numpy()
    assert np.allclose(limits,checker.joint_limits,atol=2e-6)
    rng=np.random.default_rng(1709)
    positions=rng.uniform([-.83483961,-.43230464,.87235],[.65016039,.28769536,1.32235],size=(args.batch,3))
    rot=Rotation.from_euler('ZYZ',np.c_[rng.uniform(-np.pi,np.pi,args.batch),np.full(args.batch,np.pi),np.zeros(args.batch)]).as_matrix()
    world_T_base=pose_matrix(checker.audit['lab_constants']['ROBOT_POS'],checker.audit['lab_constants']['ROBOT_ROT'])
    inverse=np.linalg.inv(world_T_base)
    local_p=positions@inverse[:3,:3].T+inverse[:3,3]
    local_q=Rotation.from_matrix(inverse[:3,:3]@rot).as_quat()[:,[3,0,1,2]]
    target=GoalToolPose.from_poses({checker.audit['tool_frame']:Pose(
        position=torch.as_tensor(local_p,device='cuda',dtype=torch.float32),
        quaternion=torch.as_tensor(local_q,device='cuda',dtype=torch.float32))},num_goalset=1)
    timings=[]
    for iteration in range(4):
        before=time.perf_counter()
        result=ik.solve_pose(target,return_seeds=args.seeds)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-before
        timings.append(elapsed)
        print('IK',iteration,elapsed,int(result.success.any(dim=1).sum()),flush=True)
    names=result.js_solution.joint_names
    print('RESULT SHAPE',result.js_solution.position.shape,'NAMES',names,flush=True)
    q=result.js_solution.position.cpu().numpy()[...,[names.index(n) for n in ARM_NAMES]]
    success=result.success.cpu().numpy()
    accepted=np.zeros(args.batch,dtype=bool)
    chosen=np.full((args.batch,6),np.nan)
    checked=0
    before=time.perf_counter()
    for branch in range(args.seeds):
        rows=np.flatnonzero(success[:,branch]&~accepted)
        if not len(rows):continue
        valid=checker.evaluate(q[rows,branch])['clear']
        checked+=len(rows)
        accepted[rows[valid]]=True;chosen[rows[valid]]=q[rows[valid],branch]
    hull_time=time.perf_counter()-before
    np.savez_compressed(out/'ik_benchmark_candidates.npz',positions=positions,quaternion_world=Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]],
        q=q,ik_success=success,accepted=accepted,chosen=chosen)
    # Verify independent source FK for accepted targets.
    fk=batch_fk(checker.audit,chosen[accepted])[checker.audit['tool_frame']]
    position_errors=np.linalg.norm(fk[:,:3,3]-local_p[accepted],axis=1)
    rotation_errors=Rotation.from_matrix(np.transpose(fk[:,:3,:3],(0,2,1))@(inverse[:3,:3]@rot[accepted])).magnitude()
    report=dict(batch=args.batch,seeds=args.seeds,joint_names=ik.kinematics.joint_names,
        timings_s=timings,ik_found=int(success.any(axis=1).sum()),source_hull_clear=int(accepted.sum()),
        exact_candidates_checked=checked,exact_validation_s=hull_time,
        max_position_error_m=float(position_errors.max()) if len(position_errors) else None,
        max_rotation_error_rad=float(rotation_errors.max()) if len(rotation_errors) else None)
    (out/'ik_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
