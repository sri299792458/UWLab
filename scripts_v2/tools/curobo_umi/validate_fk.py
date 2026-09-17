"""Compare cuRobo link poses with saved Isaac articulation witnesses and USD FK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

from export_model import ARM_NAMES, fk


def pose_matrix(pose):
    a=np.eye(4);p=np.asarray(pose)
    a[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();a[:3,3]=p[:3]
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--pose-manifest',type=Path,required=True);args=p.parse_args()
    model=args.model_dir.resolve();audit=json.loads((model/'export_audit.json').read_text())
    manifest=json.loads(args.pose_manifest.read_text())
    offset=audit['tool_offset']
    tool_transform=pose_matrix([*offset['pos'],*offset['quat']])
    links=sorted({j['parent'] for j in audit['articulation_tree']}|{j['child'] for j in audit['articulation_tree']})
    links+= [s['name'] for s in audit['collision_shapes']]+[audit['tool_frame']]
    config=dict(urdf_path=audit['urdf'],asset_root_path=str(model),base_link='base_link',
        tool_frames=links,load_collision_spheres=False)
    kin=Kinematics(KinematicsCfg.from_data_dict(config))
    rows=[]
    for s in manifest['samples']:
        rows.append(dict(zip(manifest['joint_names'],s['joint_positions'])))
    q=torch.tensor([[row[n] for n in kin.joint_names] for row in rows],device='cuda',dtype=torch.float32)
    result=kin.compute_kinematics(JointState.from_position(q,joint_names=kin.joint_names))
    errors=[]
    for i,s in enumerate(manifest['samples']):
        reference=dict(zip(manifest['body_names'],s['body_poses']))
        base=pose_matrix(reference['base_link'])
        reference[audit['tool_frame']]=pose_matrix(reference[audit['tool_parent']])@tool_transform
        for name,p in reference.items():
            cu=result.tool_poses.get_link_pose(name)
            pos=cu.position.detach().cpu().numpy().reshape(len(rows),3)[i]
            quat=cu.quaternion.detach().cpu().numpy().reshape(len(rows),4)[i]
            predicted=base@pose_matrix([*pos,*quat])
            target=p if isinstance(p,np.ndarray) and p.shape==(4,4) else pose_matrix(p)
            errors.append(dict(family=s['family'],row=s['row'],body=name,
                position_error_m=float(np.linalg.norm(predicted[:3,3]-target[:3,3])),
                rotation_error_rad=float((Rotation.from_matrix(predicted[:3,:3]).inv()*Rotation.from_matrix(target[:3,:3])).magnitude())))
    rng=np.random.default_rng(1709);random_rows=[]
    for _ in range(128):
        random_rows.append({n:float(rng.uniform(-np.pi,np.pi) if n in ARM_NAMES else rng.uniform(0,.5)) for n in kin.joint_names})
    q=torch.tensor([[r[n] for n in kin.joint_names] for r in random_rows],device='cuda',dtype=torch.float32)
    result=kin.compute_kinematics(JointState.from_position(q,joint_names=kin.joint_names))
    random_errors=[]
    for i,row in enumerate(random_rows):
        expected=fk(audit['articulation_tree'],row)
        expected[audit['tool_frame']]=expected[audit['tool_parent']]@tool_transform
        for name,T in expected.items():
            pose=result.tool_poses.get_link_pose(name)
            pos=pose.position.detach().cpu().numpy().reshape(len(random_rows),3)[i]
            quat=pose.quaternion.detach().cpu().numpy().reshape(len(random_rows),4)[i]
            predicted=pose_matrix([*pos,*quat])
            random_errors.append(dict(body=name,sample=i,position_error_m=float(np.linalg.norm(T[:3,3]-pos)),
                rotation_error_rad=float((Rotation.from_matrix(T[:3,:3]).inv()*Rotation.from_matrix(predicted[:3,:3])).magnitude())))
    report=dict(isaac_witness_count=len(rows),isaac_link_comparisons=len(errors),random_configuration_count=len(random_rows),
        joint_names=kin.joint_names,tool_frames=links,tool_offset=offset,isaac_errors=errors,random_errors=random_errors,
        max_isaac_position_error_m=max(e['position_error_m'] for e in errors),
        max_isaac_rotation_error_rad=max(e['rotation_error_rad'] for e in errors),
        max_random_position_error_m=max(e['position_error_m'] for e in random_errors),
        max_random_rotation_error_rad=max(e['rotation_error_rad'] for e in random_errors))
    report['passed']=max(report['max_isaac_position_error_m'],report['max_random_position_error_m'])<2e-5 and max(report['max_isaac_rotation_error_rad'],report['max_random_rotation_error_rad'])<2e-5
    (model/'curobo_fk_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['isaac_errors','random_errors','tool_frames']},indent=2),flush=True)
    if not report['passed']:raise RuntimeError('cuRobo FK does not match the Isaac/USD witnesses')


if __name__=='__main__':main()
