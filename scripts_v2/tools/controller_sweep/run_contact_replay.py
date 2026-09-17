"""Matched A/B replay adding native contact and full-state observations only."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--worker',type=int,default=0)
parser.add_argument('--workers',type=int,default=4)
parser.add_argument('--phase',choices=['screen','holdout','grasp'],default='screen')
parser.add_argument('--gain-ids',default='')
parser.add_argument('--gain-start',type=int,default=0)
parser.add_argument('--plan',default='/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep/plan.json')
parser.add_argument('--output-name',default='')
parser.add_argument('--pilot',action='store_true')
parser.add_argument('--count',type=int,default=0)
parser.add_argument('--knuckle-inertia',choices=['nominal','fallback'],default='nominal')
parser.add_argument('--robot-model',choices=['umi','stock'],default='umi')
parser.add_argument('--contact-mode',choices=['native','no_robot_obstacles'],default='native')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args,multi_gpu=False).app

import hashlib
import inspect
import json
import math
import os
import time
import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as mu
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_reset_cfg import UmiObjectAnywhereEEAnywhereCfg
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep')
plan_path=Path(args.plan);plan=json.loads(plan_path.read_text())
for path,digest in plan['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
if args.phase=='grasp':
    grasp_inputs=json.loads((ROOT/'grasp_poses.json').read_text())
    poses=[dict(**p,q=p['joint_positions'][:6],cube_mass_kg=mass) for p in grasp_inputs['poses'] for mass in grasp_inputs['cube_masses_kg']]
else:
    poses=[p for p in json.loads((ROOT/'poses.json').read_text())['poses'] if p['split']==args.phase]
if args.count:poses=poses[:args.count]
if args.pilot:gain_ids=[0,2,85]
elif args.gain_ids:gain_ids=[int(v) for v in args.gain_ids.split(',')]
else:gain_ids=list(range(args.gain_start,len(plan['gains'])))[args.worker::args.workers]
gains=[plan['gains'][i] for i in gain_ids];P=len(poses);G=len(gains);N=P*G
assert args.robot_model=='umi' and args.knuckle_inertia=='nominal'
out=ROOT.parent/'24_gain_failure_diagnosis'/(args.output_name or args.phase)/f'worker_{args.worker}'
out.mkdir(parents=True,exist_ok=True)
assert not (out/'summary.json').exists(),out
runner_source=Path(__file__).read_bytes()
(out/'run_gain_sweep.py').write_bytes(runner_source)

cfg=UmiObjectAnywhereEEAnywhereCfg();cfg.scene.num_envs=N;cfg.scene.env_spacing=3.0
if args.contact_mode=='no_robot_obstacles':
    from pxr import Usd,UsdPhysics
    original_spawn=cfg.scene.robot.spawn.func
    def spawn_with_lab_filter(prim_path,spawn_cfg,*positional,**keywords):
        prim=original_spawn(prim_path,spawn_cfg,*positional,**keywords)
        table_path=str(prim.GetPath().GetParentPath())+'/Table'
        for body in Usd.PrimRange(prim):
            if body.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().AddTarget(table_path)
        return prim
    cfg.scene.robot.spawn.func=spawn_with_lab_filter
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=False
if args.robot_model=='stock':
    assert args.phase!='grasp','UMI grasp geometry cannot be replayed as a stock-finger grasp.'
    assert args.knuckle_inertia=='nominal','The stock asset already supplies its native fallback properties.'
    cfg.scene.robot.spawn.usd_path='/home/kanth042/.cache/uwlab/assets/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd'
cfg.sim.device=args.device;cfg.seed=7391
assert plan['physics_hz']==120 and plan['decimation']==12
cfg.sim.dt=1/120.;cfg.decimation=12;cfg.sim.render_interval=12
disabled=[]
for name,term in list(vars(cfg.events).items()):
    if isinstance(term,EventTermCfg):setattr(cfg.events,name,None);disabled.append(name)
for name,term in list(vars(cfg.terminations).items()):
    if isinstance(term,TerminationTermCfg):setattr(cfg.terminations,name,None)
# Cubes are present only to satisfy the inherited scene API; neither participates
# in this empty-hand screen. All robot and lab properties remain inherited.
for name,position in [('insertive_object',(2.,2.,2.)),('receptive_object',(2.,2.,2.2))]:
    obj=getattr(cfg.scene,name);obj.init_state.pos=position;obj.spawn.rigid_props.kinematic_enabled=True
cfg.scene.robot.spawn.activate_contact_sensors=True
cfg.scene.robot_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,history_length=1)
if args.phase=='grasp':
    cfg.scene.insertive_object.spawn.rigid_props.kinematic_enabled=False
    cfg.scene.robot.spawn.activate_contact_sensors=True
    cfg.scene.insertive_object.spawn.activate_contact_sensors=True
    for side in ['left','right']:
        setattr(cfg.scene,side+'_pad_contact',ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/'+side+'_inner_finger',update_period=0.,history_length=1,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/InsertiveObject']))
env=gym.make('OmniReset-UMI-Defaults-ObjectAnywhereEEAnywhere-v0',cfg=cfg).unwrapped
env.reset();robot=env.scene['robot'];arm=env.action_manager.get_term('arm')
# Asset spawn applies the authored base-link transform as well as init_state.rot.
# The normal reset event overwrites that transform; this diagnostic disables events,
# so explicitly write the intended body pose before writing joints or cube poses.
mapping=json.loads((ROOT.parent/'21_sphere_reachability/config.json').read_text())
root_pose=torch.tensor(mapping['robot_base_position_world_m']+mapping['robot_base_quaternion_world_wxyz'],device=env.device).repeat(N,1)
root_pose[:,:3]+=env.scene.env_origins
robot.write_root_pose_to_sim(root_pose)
robot.write_root_velocity_to_sim(torch.zeros((N,6),device=env.device))
inertia_override=None
if args.knuckle_inertia=='fallback':
    view=robot.root_physx_view
    original_mass,original_inertia,original_com=(v.clone() for v in (view.get_masses(),view.get_inertias(),view.get_coms()))
    knuckles=[robot.body_names.index(n) for n in ['left_outer_knuckle','right_outer_knuckle']]
    requested=original_inertia.clone()
    requested[:,knuckles,:]=(torch.eye(3,device=requested.device,dtype=requested.dtype)*.004).flatten()
    # Touch only the selected links; a full-articulation setter can re-diagonalize
    # untouched tensors and change their stored principal-axis coordinates.
    for bid in knuckles:
        link_view=env.sim.physics_sim_view.create_rigid_body_view('/World/envs/env_*/Robot/'+robot.body_names[bid])
        assert link_view.count==N
        link_view.set_inertias(requested[:,bid].contiguous(),torch.arange(N,dtype=torch.int32,device='cpu'))
    readback=view.get_inertias()
    assert torch.equal(view.get_masses(),original_mass)
    # The orientation stored alongside COM locates the principal inertia axes;
    # it may change when replacing a tensor. The actual COM position must not.
    assert torch.equal(view.get_coms()[:,:,:3],original_com[:,:,:3])
    other=[i for i in range(len(robot.body_names)) if i not in knuckles]
    assert torch.equal(readback[:,other],original_inertia[:,other])
    assert torch.equal(view.get_coms()[:,other],original_com[:,other])
    assert torch.allclose(readback[:,knuckles],requested[:,knuckles],rtol=1e-6,atol=1e-10)
    inertia_override=dict(bodies=['left_outer_knuckle','right_outer_knuckle'],
        before_kg_m2=original_inertia[0,knuckles].cpu().tolist(),
        requested_kg_m2=requested[0,knuckles].cpu().tolist(),after_kg_m2=readback[0,knuckles].cpu().tolist(),
        masses_kg=original_mass[0,knuckles].cpu().tolist(),centers_of_mass=original_com[0,knuckles].cpu().tolist(),
        scope='Diagnostic fallback rotational-inertia reference on current UMI geometry; masses, COMs, all other body inertias, and source asset are unchanged.')
assert type(arm).__name__=='RelCartesianOSCAction'
assert not hasattr(arm.cfg,'implicit_damping')
assert abs(env.physics_dt-1/120)<1e-12 and abs(env.step_dt-.1)<1e-12
assert arm._input_clip is None
arm_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
ids=[robot.joint_names.index(n) for n in arm_names]
assert list(arm._joint_names)==arm_names
q=torch.zeros_like(robot.data.joint_pos)
if args.phase=='grasp':
    assert robot.joint_names==grasp_inputs['joint_names']
    q[:]=torch.tensor([p['joint_positions'] for p in poses],device=env.device).repeat(G,1)
else:q[:,ids]=torch.tensor([p['q'] for p in poses],device=env.device).repeat(G,1)
robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
if args.phase=='grasp':
    cube=env.scene['insertive_object']
    cube_pose=torch.tensor([p['cube_pose_world'] for p in poses],device=env.device).repeat(G,1)
    cube_pose[:,:3]+=env.scene.env_origins
    cube.write_root_pose_to_sim(cube_pose);cube.write_root_velocity_to_sim(torch.zeros((N,6),device=env.device))
    old_mass=cube.root_physx_view.get_masses();old_inertia=cube.root_physx_view.get_inertias()
    new_mass=torch.tensor([p['cube_mass_kg'] for p in poses],device=old_mass.device).repeat(G).reshape_as(old_mass)
    factor=(new_mass/old_mass).reshape(N,*([1]*(old_inertia.ndim-1)))
    indices=torch.arange(N,dtype=torch.int32,device='cpu')
    cube.root_physx_view.set_masses(new_mass,indices);cube.root_physx_view.set_inertias(old_inertia*factor,indices)
env.sim.forward();env.scene.update(0.);env.action_manager.reset()
assert torch.max(torch.abs(robot.data.root_pos_w-root_pose[:,:3]))<2e-6
assert torch.max(torch.abs((robot.data.root_quat_w*root_pose[:,3:]).sum(dim=1).abs()-1))<2e-6
arm._kp[:]=torch.tensor([g['kp'] for g in gains],device=env.device).repeat_interleave(P,dim=0)
arm._kd[:]=torch.tensor([g['kd'] for g in gains],device=env.device).repeat_interleave(P,dim=0)
assert torch.all(robot.data.joint_stiffness[:,ids]==0) and torch.all(robot.data.joint_damping[:,ids]==0)
p0,q0=(x.clone() for x in arm._get_ee_pose_root_frame())
if args.phase=='grasp':
    cube_p0,cube_q0=mu.subtract_frame_transforms(robot.data.root_pos_w,robot.data.root_quat_w,cube.data.root_pos_w,cube.data.root_quat_w)
    cube_in_wrist=mu.quat_apply(mu.quat_inv(q0),cube_p0-p0)
    cube_quat_in_wrist=mu.quat_mul(mu.quat_inv(q0),cube_q0)
initial_native_q=robot.data.joint_pos.detach().cpu().tolist()
assert torch.max(torch.abs(robot.data.joint_pos-q))<1e-6
masses=robot.root_physx_view.get_masses();inertias=robot.root_physx_view.get_inertias()
assert torch.allclose(masses,masses[:1].expand_as(masses))
assert torch.allclose(inertias,inertias[:1].expand_as(inertias))
snapshot=dict(robot_usd=cfg.scene.robot.spawn.usd_path,
    robot_model=args.robot_model,
    knuckle_inertia_mode=args.knuckle_inertia,knuckle_inertia_override=inertia_override,
    robot_usd_sha256=hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest(),
    controller_file=inspect.getfile(type(arm)),controller_sha256=hashlib.sha256(Path(inspect.getfile(type(arm))).read_bytes()).hexdigest(),
    physics_hz=120,decimation=12,action_hz=10,disabled_events=disabled,
    joint_names=robot.joint_names,body_names=robot.body_names,
    initial_root_pose_world=robot.data.root_state_w[:,:7].cpu().tolist(),
    intended_root_pose_world=root_pose.cpu().tolist(),
    initial_body_poses_world=robot.data.body_state_w[:,:,:7].cpu().tolist(),
    masses_kg=masses[0].cpu().tolist(),inertias_kg_m2=inertias[0].cpu().tolist(),
    joint_armature=robot.data.joint_armature[0].cpu().tolist(),
    joint_stiffness=robot.data.joint_stiffness[0].cpu().tolist(),joint_damping=robot.data.joint_damping[0].cpu().tolist(),
    gains=gains,poses=poses,initial_joint_positions=initial_native_q,
    arm_gravity_disabled=cfg.scene.robot.spawn.rigid_props.disable_gravity,
    solver_position_iterations=cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
    solver_velocity_iterations=cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
    note='Actual source controller invoked without modification. Fixed nominal robot dynamics. Diagnostic targets bypass policy-action clipping. Empty hand for screen/holdout; grasp phase uses a dynamic cube at each specified mass with bilateral pad/cube contact sensors.')
if args.phase=='grasp':
    snapshot['cube_masses_kg']=cube.root_physx_view.get_masses().flatten().cpu().tolist()
    snapshot['initial_cube_poses_world']=cube.data.root_state_w[:,:7].cpu().tolist()
    snapshot['cube_in_wrist_initial_m']=cube_in_wrist.cpu().tolist()
    snapshot['contact_measurement_scope']='Each inner-finger rigid body aggregates contact from its TPU pad and PETG adapter; forces are filtered to the insertive cube, not separated by collision shape.'
    snapshot['grasp_limits']=dict(position_drift_m=.015,rotation_drift_rad=math.radians(5),minimum_table_clearance_m=.02,minimum_each_finger_force_n=.02,minimum_both_fingers_contact_fraction=.8)
    from pxr import Usd, UsdPhysics
    stage=env.sim.stage
    snapshot['scene_physics_prims']=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'),Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            snapshot['scene_physics_prims'].append(dict(path=str(prim.GetPath()),
                collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if prim.HasAPI(UsdPhysics.CollisionAPI) else None,
                filtered_pairs=[str(x) for x in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()] if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else []))
    snapshot['grasp_input_sha256']=hashlib.sha256((ROOT/'grasp_poses.json').read_bytes()).hexdigest()
snapshot['contact_body_names']=env.scene['robot_contacts'].body_names
snapshot['contact_mode']=args.contact_mode
snapshot['joint_velocity_limits_rad_s']=robot.data.joint_vel_limits[0].cpu().tolist()
snapshot['actuator_classes']={name:type(actuator).__name__ for name,actuator in robot.actuators.items()}
snapshot['contact_intervention']='Robot self-collision disabled and each robot rigid body filtered against its environment Table body; cube/finger contacts preserved.' if args.contact_mode!='native' else None
snapshot['extra_recording_scope']='Native contact net forces per robot body, all joint states, body poses, cube poses. Added logging only.'
(out/'runtime_inputs.json').write_text(json.dumps(snapshot,indent=2)+'\n')

actions=torch.zeros((N,7),device=env.device);actions[:,-1]=-1. if args.phase=='grasp' else 1.
steps=1200
# Error(6), joint position(6), joint velocity(6), commanded torque(6), and
# abnormal flag are recorded at every physics step, including between actions.
values=torch.empty((steps,N,32 if args.phase=='grasp' else 25),device=env.device)
limits=robot.data.joint_vel_limits.clone()
full_q=torch.empty((steps,N,len(robot.joint_names)),device=env.device)
full_qd=torch.empty_like(full_q)
body_poses=torch.empty((steps,N,len(robot.body_names),7),device=env.device)
contact_forces=torch.empty((steps,N,len(env.scene['robot_contacts'].body_names),3),device=env.device)
cube_poses=torch.empty((steps,N,7),device=env.device) if args.phase=='grasp' else None
started=time.perf_counter();tindex=0;formula_error=None
for policy_step in range(100):
    t=policy_step/10.;phase=max(0.,min(6.,t-2.))
    window=math.sin(math.pi*phase/6.)**2 if 2.<=t<8. else 0.
    wave=torch.tensor([math.sin(2*math.pi*.5*phase+k*2*math.pi/3) for k in range(3)],device=env.device)
    dp=.02*window*wave;dr=.1*window*wave
    angle=dr.norm();axis=dr/angle.clamp_min(1e-9)
    dq=torch.cat((torch.cos(angle/2).reshape(1),axis*torch.sin(angle/2))).expand(N,4)
    desired_p=p0+dp;desired_q=mu.quat_mul(dq,q0)
    current_p,current_q=arm._get_ee_pose_root_frame()
    action_error=torch.cat((desired_p-current_p,mu.axis_angle_from_quat(mu.quat_mul(desired_q,mu.quat_inv(current_q)))),dim=1)
    actions[:,:6]=action_error/arm._scale
    env.action_manager.process_action(actions)
    assert torch.max(torch.abs(arm._ee_pos_des-desired_p))<2e-6
    for substep in range(12):
        env.action_manager.apply_action()
        tau=robot.data.joint_effort_target[:,ids].clone()
        if policy_step==25 and substep==0:
            cp,cq=arm._get_ee_pose_root_frame()
            e=torch.cat((arm._ee_pos_des-cp,mu.axis_angle_from_quat(mu.quat_mul(arm._ee_quat_des,mu.quat_inv(cq)))),dim=1)
            J=compute_jacobian_analytical(robot.data.joint_pos[:,ids],device=str(env.device),usd_path=cfg.scene.robot.spawn.usd_path)
            v=torch.bmm(J,robot.data.joint_vel[:,ids,None]).squeeze(-1)
            expected=torch.bmm(J.transpose(-1,-2),(arm._kp*e-arm._kd*v).unsqueeze(-1)).squeeze(-1)
            expected=torch.clamp(expected,-arm._torque_max,arm._torque_max)
            formula_error=float((expected-tau).abs().max());assert formula_error<1e-5
        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
        cp,cq=arm._get_ee_pose_root_frame()
        e=torch.cat((desired_p-cp,mu.axis_angle_from_quat(mu.quat_mul(desired_q,mu.quat_inv(cq)))),dim=1)
        abnormal=(robot.data.joint_vel.abs()>2*limits).any(dim=1)
        values[tindex,:,:6]=e;values[tindex,:,6:12]=robot.data.joint_pos[:,ids]
        values[tindex,:,12:18]=robot.data.joint_vel[:,ids];values[tindex,:,18:24]=tau
        values[tindex,:,24]=abnormal
        if args.phase=='grasp':
            cube_p,cube_q=mu.subtract_frame_transforms(robot.data.root_pos_w,robot.data.root_quat_w,cube.data.root_pos_w,cube.data.root_quat_w)
            expected_cube=cp+mu.quat_apply(cq,cube_in_wrist)
            bottom=cube.data.root_pos_w[:,2]-env.scene.env_origins[:,2]-.03*mu.matrix_from_quat(cube.data.root_quat_w)[:,2,:].abs().sum(dim=1)-.84235
            left=env.scene['left_pad_contact'].data.force_matrix_w
            right=env.scene['right_pad_contact'].data.force_matrix_w
            assert left is not None and right is not None
            values[tindex,:,25]=(cube_p-expected_cube).norm(dim=1)
            values[tindex,:,26]=bottom
            values[tindex,:,27]=left.reshape(N,-1,3).norm(dim=2).sum(dim=1)
            values[tindex,:,28]=right.reshape(N,-1,3).norm(dim=2).sum(dim=1)
            values[tindex,:,29]=robot.data.joint_pos[:,robot.joint_names.index('finger_joint')]
            values[tindex,:,30]=cube.data.root_lin_vel_w.norm(dim=1)
            expected_cube_q=mu.quat_mul(cq,cube_quat_in_wrist)
            values[tindex,:,31]=mu.axis_angle_from_quat(mu.quat_mul(cube_q,mu.quat_inv(expected_cube_q))).norm(dim=1)
        full_q[tindex]=robot.data.joint_pos
        full_qd[tindex]=robot.data.joint_vel
        body_poses[tindex]=robot.data.body_state_w[:,:,:7]
        contact_forces[tindex]=env.scene['robot_contacts'].data.net_forces_w
        if cube_poses is not None:cube_poses[tindex]=cube.data.root_state_w[:,:7]
        tindex+=1
    if policy_step%10==9:
        (out/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),policy_steps=policy_step+1,total_policy_steps=100,
            environments=N,gain_ids=gain_ids,elapsed_s=time.perf_counter()-started))+'\n')
        print('PROGRESS',policy_step+1,'/ 100','elapsed',round(time.perf_counter()-started,2),flush=True)
torch.cuda.synchronize();elapsed=time.perf_counter()-started
torch.save(dict(joint_positions=full_q.cpu(),joint_velocities=full_qd.cpu(),body_poses_world=body_poses.cpu(),
    contact_net_forces_world=contact_forces.cpu(),cube_poses_world=cube_poses.cpu() if cube_poses is not None else None,
    joint_names=robot.joint_names,body_names=robot.body_names,contact_body_names=env.scene['robot_contacts'].body_names,
    gain_ids=gain_ids,pose_count=P,physics_hz=120),out/'contact_trajectories.pt')
values=values.cpu()
torch.save(dict(values=values,gain_ids=gain_ids,pose_count=P,
    columns=['ex','ey','ez','erx','ery','erz']+[f'q{i}' for i in range(6)]+[f'v{i}' for i in range(6)]+[f'tau{i}' for i in range(6)]+['abnormal']+(['cube_relative_error_m','cube_bottom_above_table_m','left_pad_cube_force_n','right_pad_cube_force_n','finger_joint_rad','cube_speed_m_s','cube_relative_rotation_error_rad'] if args.phase=='grasp' else []),
    physics_hz=120,decimation=12),out/'trajectories.pt')
finite=torch.isfinite(values).all(dim=0).all(dim=1)
def rms(part,columns):return values[part,:,columns].double().square().sum(dim=-1).mean(dim=0).sqrt()
metrics=dict(motion_position_rms_m=rms(slice(240,960),slice(0,3)),motion_rotation_rms_rad=rms(slice(240,960),slice(3,6)),
    final_position_rms_m=rms(slice(1080,1200),slice(0,3)),final_rotation_rms_rad=rms(slice(1080,1200),slice(3,6)),
    final_wrist_speed_rms_rad_s=rms(slice(1080,1200),slice(15,18)),
    maximum_joint_speed_rad_s=values[:,:,12:18].abs().amax(dim=(0,2)),
    saturated_step_fraction=(values[:,:,18:24].abs()>=torch.tensor([150.,150.,150.,28.,28.,28.])*.999).any(dim=2).double().mean(dim=0),
    abnormal=values[:,:,24].bool().any(dim=0),finite=finite)
if args.phase=='grasp':
    bilateral=(values[240:,:,27]>.02)&(values[240:,:,28]>.02)
    metrics.update(cube_relative_error_max_m=values[240:,:,25].amax(dim=0),
        cube_rotation_error_max_rad=values[240:,:,31].amax(dim=0),
        cube_clearance_min_m=values[240:,:,26].amin(dim=0),
        bilateral_contact_fraction=bilateral.double().mean(dim=0))
    grasp_pass=finite&(metrics['cube_relative_error_max_m']<.015)&(metrics['cube_rotation_error_max_rad']<math.radians(5))&(metrics['cube_clearance_min_m']>.02)&(metrics['bilateral_contact_fraction']>=.8)
good=finite&~metrics['abnormal']
for key,limit in plan['ranking_limits'].items():good&=metrics[key]<=limit
rows=[]
for j,gain in enumerate(gains):
    s=slice(j*P,(j+1)*P);row=dict(**gain,count=P,passed=int(good[s].sum()),finite=int(finite[s].sum()),abnormal=int(metrics['abnormal'][s].sum()))
    if args.phase=='grasp':
        row['grasp_passed']=int(grasp_pass[s].sum());row['controller_and_grasp_passed']=int((good[s]&grasp_pass[s]).sum())
        row['grasp_passed_by_mass']={str(m):int(grasp_pass[s][torch.tensor([p['cube_mass_kg']==m for p in poses])].sum()) for m in grasp_inputs['cube_masses_kg']}
    row['per_pose']={k:[float(x) if np.isfinite(float(x)) else None for x in v[s]] for k,v in metrics.items()}
    for key,v in metrics.items():
        if key in ['abnormal','finite']:continue
        selected=v[s][finite[s]]
        row[key+'_median']=float(selected.double().median()) if len(selected) else None
        row[key+'_max']=float(selected.max()) if len(selected) else None
    rows.append(row)
summary=dict(phase=args.output_name or args.phase,pose_split=args.phase,pilot=args.pilot,worker=args.worker,physics_hz=120,decimation=12,action_hz=10,
    robot_model=args.robot_model,
    knuckle_inertia_mode=args.knuckle_inertia,
    environments=N,gain_ids=gain_ids,pose_count=P,simulation_seconds=10,wall_seconds=elapsed,
    controller_formula_max_error_nm=formula_error,results=rows,
    plan_path=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(runner_source).hexdigest())
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print('SWEEP_COMPLETE',[(r['id'],r['passed'],r['count']) for r in rows],flush=True)
env.close();app.close(skip_cleanup=True)
