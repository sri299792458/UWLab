"""Select airborne grasp geometry for paired, nominal-dynamics contact trials."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911');OUT=ROOT/'22_explicit_gain_sweep'
source=ROOT/'11_original_gains_data/OmniReset/Resets/InsertiveAprilCube60__ReceptiveAprilCube60/resets_ObjectAnywhereEEGrasped.pt'
bank=torch.load(source,map_location='cpu',weights_only=False)['initial_state']
audit=json.loads((ROOT/'15_table_reachability/model/export_audit.json').read_text())
mapping=json.loads((ROOT/'21_sphere_reachability/config.json').read_text())
native=json.loads((OUT/'screen/worker_0/runtime_inputs.json').read_text())
names=native['joint_names'];allq=torch.stack(bank['articulation']['robot']['joint_position']).numpy()
oldroot=torch.stack(bank['articulation']['robot']['root_pose']).numpy()
oldcube=torch.stack(bank['rigid_object']['insertive_object']['root_pose']).numpy()
def T(p,q):
    out=np.eye(4);out[:3,3]=p;out[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();return out
base=T(mapping['robot_base_position_world_m'],mapping['robot_base_quaternion_world_wxyz'])
rng=np.random.default_rng(7392);rows=rng.permutation(len(allq))[:2048]
q=allq[rows];N=len(q)
transforms={'base_link':np.broadcast_to(np.eye(4),(N,4,4)).copy()}
for joint in audit['articulation_tree']:
    left,right,sign=np.asarray(joint['T0']),np.asarray(joint['T1']),1
    if joint['reverse']:left,right,sign=right,left,-1
    if joint['fixed']:
        transforms[joint['child']]=transforms[joint['parent']]@left@np.linalg.inv(right)
    else:
        axis=np.eye(3)['XYZ'.index(joint['axis'])]
        turn=np.broadcast_to(np.eye(4),(N,4,4)).copy()
        turn[:,:3,:3]=Rotation.from_rotvec(sign*q[:,names.index(joint['name']),None]*axis).as_matrix()
        transforms[joint['child']]=transforms[joint['parent']]@left@turn@np.linalg.inv(right)
cube=[];gaps=[];center_errors=[]
grasp=transforms['robotiq_base_link']@T(audit['tool_offset']['pos'],audit['tool_offset']['quat'])
for k,row in enumerate(rows):
    moved=base@np.linalg.inv(T(oldroot[row,:3],oldroot[row,3:]))@T(oldcube[row,:3],oldcube[row,3:])
    cube.append(np.r_[moved[:3,3],Rotation.from_matrix(moved[:3,:3]).as_quat()[[3,0,1,2]]])
    gaps.append(moved[2,3]-.03*np.abs(moved[2,:3]).sum()-mapping['tabletop_z_m'])
    center_errors.append(np.linalg.norm(moved[:3,3]-(base@grasp[k])[:3,3]))
cube=np.asarray(cube);limits=np.asarray(mapping['joint_limits_rad'])
valid=(q[:,:6]>limits[0]+.15).all(axis=1)&(q[:,:6]<limits[1]-.15).all(axis=1)
valid&=(np.asarray(gaps)>.08)&(np.asarray(center_errors)<.03)
valid&=(q[:,6]>.20)&(q[:,6]<.40)
model=yaml.safe_load((ROOT/'21_sphere_reachability/robot_open_spheres.yml').read_text())['robot_cfg']
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=model,scene_model=mapping['scene_model'],
    collision_activation_distance=0.,self_collision_activation_distance=0.))
spheres=checker.get_kinematics(torch.tensor(q[:,:6],device='cuda')[:,None]).robot_spheres.clone()
shape_by_name={s['name']:s for s in audit['collision_shapes']}
for name,raw in model['kinematics']['collision_spheres'].items():
    indices=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name(name)
    centers=np.array([s['center'] for s in raw]);radii=np.array([s['radius'] for s in raw])+.001
    transform=transforms[shape_by_name[name]['body']]
    centers=np.einsum('bij,sj->bsi',transform[:,:3,:3],centers)+transform[:,None,:3,3]
    spheres[:,0,indices,:3]=torch.as_tensor(centers,device='cuda',dtype=torch.float32)
    spheres[:,0,indices,3]=torch.as_tensor(radii,device='cuda',dtype=torch.float32)
self_bad=checker.get_self_collision_distance(spheres).reshape(N,-1).amax(dim=1)>0
baseids=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link_shape_0')
spheres[:,:,baseids,3]=-100.
world_bad=checker.get_collision_constraint(spheres).reshape(N,-1).amax(dim=1)>0
valid&=(~self_bad&~world_bad).cpu().numpy()
chosen=np.flatnonzero(valid)[:24];assert len(chosen)==24,int(valid.sum())
records=[dict(source_row=int(rows[i]),joint_positions=q[i].tolist(),cube_pose_world=cube[i].tolist(),
    cube_bottom_above_table_m=float(gaps[i]),cube_to_grasp_reference_m=float(center_errors[i])) for i in chosen]
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=7392,
    robot_root_pose=np.r_[mapping['robot_base_position_world_m'],mapping['robot_base_quaternion_world_wxyz']].tolist(),
    joint_names=names,poses=records,cube_masses_kg=[.02,.10,.20],candidates_considered=N,candidates_passing=int(valid.sum()),
    scope='Existing airborne-grasp bank supplies geometry only. Robot and cube are rigidly transformed together to the current nominal mounting pose; initial velocities will be zeroed. Saved table state and saved generation-controller behavior are not replayed. Starts require cube bottom >80 mm above current tabletop, cube center within 30 mm of grasp reference, mid-closure, joint-limit margin, and sphere clearance from the lab and non-excluded robot links. Full finger articulation is used to place the spheres. Dynamic validity and bilateral pad contact are tested afresh in Isaac Lab.')
(OUT/'grasp_poses.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(selected=len(records),candidates_passing=int(valid.sum()),minimum_cube_gap_m=min(r['cube_bottom_above_table_m'] for r in records)),indent=2))
