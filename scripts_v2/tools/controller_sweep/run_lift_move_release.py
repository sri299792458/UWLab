"""Run hold, lift, lateral carry, and release with the unchanged explicit controller."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--worker',type=int,default=0)
parser.add_argument('--workers',type=int,default=4)
parser.add_argument('--phase',choices=['screen','holdout','grasp'],default='grasp')
parser.add_argument('--gain-ids',default='292')
parser.add_argument('--gains-json',default='')
parser.add_argument('--use-config-gains',action='store_true')
parser.add_argument('--gain-start',type=int,default=0)
parser.add_argument('--plan',default='/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep/plan_expanded.json')
parser.add_argument('--output-name',default='')
parser.add_argument('--pilot',action='store_true')
parser.add_argument('--count',type=int,default=0)
parser.add_argument('--knuckle-inertia',choices=['nominal','fallback'],default='nominal')
parser.add_argument('--robot-model',choices=['umi','stock'],default='umi')
parser.add_argument('--contact-mode',choices=['native'],default='native')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
assert args.phase=='grasp'
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
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg
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
gains=[plan['gains'][i] for i in gain_ids]
if args.gains_json:
    gains=json.loads(Path(args.gains_json).read_text())
    gain_ids=[g['id'] for g in gains]
P=len(poses);G=len(gains);N=P*G
assert args.robot_model=='umi' and args.knuckle_inertia=='nominal'
out=ROOT.parent/'25_lift_move_release'/(args.output_name or args.phase)/f'worker_{args.worker}'
out.mkdir(parents=True,exist_ok=True)
assert not (out/'summary.json').exists(),out
runner_source=Path(__file__).read_bytes()
(out/'run_gain_sweep.py').write_bytes(runner_source)

cfg=UmiCubeTrainCfg();cfg.scene.num_envs=N;cfg.scene.env_spacing=3.0
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
env=gym.make('OmniReset-UMI-Defaults-State-Train-v0',cfg=cfg).unwrapped
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
expected_kp=torch.tensor([g['kp'] for g in gains],device=env.device,dtype=arm._kp.dtype).repeat_interleave(P,dim=0)
expected_kd=torch.tensor([g['kd'] for g in gains],device=env.device,dtype=arm._kd.dtype).repeat_interleave(P,dim=0)
if args.use_config_gains:
    assert G==1
    assert torch.allclose(arm._kp,expected_kp,rtol=0,atol=1e-5)
    assert torch.allclose(arm._kd,expected_kd,rtol=0,atol=1e-5)
else:
    arm._kp[:]=expected_kp
    arm._kd[:]=expected_kd
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
    note='Actual training scene and source controller, fixed nominal dynamics. Saved grasp initial states. Scripted pose feedback passes through the normal action manager with raw arm actions clamped to [-1,1]. No policy or automatic resets.')
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
snapshot['uses_task_config_gains_without_runtime_override']=args.use_config_gains
snapshot['actual_runtime_kp']=arm._kp[0].cpu().tolist()
snapshot['actual_runtime_kd']=arm._kd[0].cpu().tolist()
snapshot['task_config_source']=inspect.getfile(UmiCubeTrainCfg)
snapshot['task_config_sha256']=hashlib.sha256(Path(inspect.getfile(UmiCubeTrainCfg)).read_bytes()).hexdigest()

# Use one concrete handling sequence. The hand orientation stays fixed.
# Scripted feedback is bounded to the training action scale before entering
# the unchanged action manager; targets are refreshed only at 10 Hz.
cube_initial_world=cube.data.root_pos_w.clone()
table_center=torch.tensor([.266535,-.060305],device=env.device)
direction_world=torch.zeros((N,3),device=env.device)
direction_world[:,:2]=table_center-(cube_initial_world-env.scene.env_origins)[:,:2]
direction_world/=direction_world.norm(dim=-1,keepdim=True).clamp_min(1e-9)
direction_root=mu.quat_apply_inverse(robot.data.root_quat_w,direction_world)
up_world=torch.zeros((N,3),device=env.device);up_world[:,2]=1
up_root=mu.quat_apply_inverse(robot.data.root_quat_w,up_world)
snapshot['protocol']=dict(hold_s=[0,2],lift_s=[2,4],lift_settle_s=[4,5],
    move_s=[5,7],move_settle_s=[7,8],open_s=[8,11],lift_m=.05,move_m=.05,
    direction_world=direction_world.cpu().tolist(),raw_arm_action_clip=[-1,1],
    scope='Lift and carry from saved airborne reset grasps, then release under gravity. No pickup or stacking policy is tested.')
(out/'runtime_inputs.json').write_text(json.dumps(snapshot,indent=2)+'\n')
steps=1320
values=torch.empty((steps,N,32),device=env.device)
targets=torch.empty((steps,N,7),device=env.device)
full_q=torch.empty((steps,N,len(robot.joint_names)),device=env.device)
full_qd=torch.empty_like(full_q)
body_poses=torch.empty((steps,N,len(robot.body_names),7),device=env.device)
contact_forces=torch.empty((steps,N,len(env.scene['robot_contacts'].body_names),3),device=env.device)
cube_poses=torch.empty((steps,N,7),device=env.device)
action_trace=torch.empty((110,N,7),device=env.device)
actions=torch.zeros((N,7),device=env.device)
limits=robot.data.joint_vel_limits.clone()
formula_error=0.;tindex=0;started=time.perf_counter()
def smooth_fraction(time,start,end):
    x=max(0.,min(1.,(time-start)/(end-start)))
    return x*x*(3-2*x)
for policy_step in range(110):
    t=policy_step/10.
    desired_p=p0+.05*smooth_fraction(t,2,4)*up_root+.05*smooth_fraction(t,5,7)*direction_root
    desired_q=q0
    cp,cq=arm._get_ee_pose_root_frame()
    outer_error=torch.cat((desired_p-cp,mu.axis_angle_from_quat(mu.quat_mul(desired_q,mu.quat_inv(cq)))),dim=1)
    actions[:,:6]=(outer_error/arm._scale).clamp(-1.,1.)
    actions[:,-1]=-1. if t<8 else 1.
    env.action_manager.process_action(actions);action_trace[policy_step]=actions
    for substep in range(12):
        env.action_manager.apply_action()
        tau=robot.data.joint_effort_target[:,ids].clone()
        if substep==0:
            cp,cq=arm._get_ee_pose_root_frame()
            actual_error=torch.cat((arm._ee_pos_des-cp,mu.axis_angle_from_quat(mu.quat_mul(arm._ee_quat_des,mu.quat_inv(cq)))),dim=1)
            J=compute_jacobian_analytical(robot.data.joint_pos[:,ids],device=str(env.device),usd_path=cfg.scene.robot.spawn.usd_path)
            v=torch.bmm(J,robot.data.joint_vel[:,ids,None]).squeeze(-1)
            expected=torch.bmm(J.transpose(-1,-2),(arm._kp*actual_error-arm._kd*v).unsqueeze(-1)).squeeze(-1)
            expected=torch.clamp(expected,-arm._torque_max,arm._torque_max)
            formula_error=max(formula_error,float((expected-tau).abs().max()))
            assert formula_error<1e-5
        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
        cp,cq=arm._get_ee_pose_root_frame()
        error=torch.cat((desired_p-cp,mu.axis_angle_from_quat(mu.quat_mul(desired_q,mu.quat_inv(cq)))),dim=1)
        values[tindex,:,:6]=error
        values[tindex,:,6:12]=robot.data.joint_pos[:,ids]
        values[tindex,:,12:18]=robot.data.joint_vel[:,ids]
        values[tindex,:,18:24]=tau
        values[tindex,:,24]=(robot.data.joint_vel.abs()>2*limits).any(dim=1)
        cube_p,cube_q=mu.subtract_frame_transforms(robot.data.root_pos_w,robot.data.root_quat_w,cube.data.root_pos_w,cube.data.root_quat_w)
        expected_cube=cp+mu.quat_apply(cq,cube_in_wrist)
        values[tindex,:,25]=(cube_p-expected_cube).norm(dim=1)
        values[tindex,:,26]=cube.data.root_pos_w[:,2]-env.scene.env_origins[:,2]-.03*mu.matrix_from_quat(cube.data.root_quat_w)[:,2,:].abs().sum(dim=1)-.84235
        for side,column in [('left',27),('right',28)]:
            force=env.scene[side+'_pad_contact'].data.force_matrix_w
            assert force is not None
            values[tindex,:,column]=force.reshape(N,-1,3).norm(dim=2).sum(dim=1)
        values[tindex,:,29]=robot.data.joint_pos[:,robot.joint_names.index('finger_joint')]
        values[tindex,:,30]=cube.data.root_lin_vel_w.norm(dim=1)
        expected_q=mu.quat_mul(cq,cube_quat_in_wrist)
        values[tindex,:,31]=mu.axis_angle_from_quat(mu.quat_mul(cube_q,mu.quat_inv(expected_q))).norm(dim=1)
        targets[tindex]=torch.cat((desired_p,desired_q),dim=1)
        full_q[tindex]=robot.data.joint_pos;full_qd[tindex]=robot.data.joint_vel
        body_poses[tindex]=robot.data.body_state_w[:,:,:7]
        contact_forces[tindex]=env.scene['robot_contacts'].data.net_forces_w
        cube_poses[tindex]=cube.data.root_state_w[:,:7]
        tindex+=1
    if policy_step%10==9:
        progress=dict(pid=os.getpid(),policy_steps=policy_step+1,total_policy_steps=110,
            environments=N,gain_ids=gain_ids,elapsed_s=time.perf_counter()-started)
        (out/'progress.json').write_text(json.dumps(progress)+'\n')
        print('PROGRESS',json.dumps(progress),flush=True)

torch.cuda.synchronize()
elapsed=time.perf_counter()-started
torch.save(dict(joint_positions=full_q.cpu(),joint_velocities=full_qd.cpu(),body_poses_world=body_poses.cpu(),
    contact_net_forces_world=contact_forces.cpu(),cube_poses_world=cube_poses.cpu(),
    joint_names=robot.joint_names,body_names=robot.body_names,contact_body_names=env.scene['robot_contacts'].body_names,
    gain_ids=gain_ids,pose_count=P,physics_hz=120),out/'contact_trajectories.pt')
torch.save(dict(values=values.cpu(),targets_root=targets.cpu(),actions=action_trace.cpu(),gain_ids=gain_ids,pose_count=P,
    columns=['ex','ey','ez','erx','ery','erz']+[f'q{i}' for i in range(6)]+[f'v{i}' for i in range(6)]+[f'tau{i}' for i in range(6)]+['abnormal','cube_relative_error_m','cube_bottom_above_table_m','left_pad_cube_force_n','right_pad_cube_force_n','finger_joint_rad','cube_speed_m_s','cube_relative_rotation_error_rad'],
    physics_hz=120,decimation=12),out/'trajectories.pt')

finite=torch.isfinite(values).all(dim=(0,2))
bilateral=(values[:,:,27]>.02)&(values[:,:,28]>.02)
cube_rise=cube_poses[599,:,2]-cube_poses[239,:,2]
cube_travel=((cube_poses[959,:,:3]-cube_poses[599,:,:3])*direction_world).sum(dim=1)
release_drop=cube_poses[959,:,2]-cube_poses[-1,:,2]
eligible=[i for i,n in enumerate(env.scene['robot_contacts'].body_names) if n not in ['left_inner_finger','right_inner_finger']]
obstacle_force=contact_forces[:960,:,eligible].norm(dim=-1).amax(dim=(0,2))
def rms(part,columns):
    return values[part,:,columns].double().square().sum(dim=-1).mean(dim=0).sqrt()
metrics=dict(finite=finite,abnormal=values[:,:,24].bool().any(dim=0),
    initial_hold_bilateral_fraction=bilateral[180:240].float().mean(dim=0),
    carried_bilateral_fraction=bilateral[240:960].float().mean(dim=0),
    cube_rise_m=cube_rise,cube_lateral_travel_m=cube_travel,
    carry_relative_position_max_m=values[240:960,:,25].amax(dim=0),
    carry_relative_rotation_max_rad=values[240:960,:,31].amax(dim=0),
    release_drop_m=release_drop,
    final_finger_cube_force_max_n=values[-60:,:,27:29].amax(dim=(0,2)),
    final_driver_angle_rad=values[-60:,:,29].mean(dim=0),
    lift_end_position_rms_m=rms(slice(540,600),slice(0,3)),
    move_end_position_rms_m=rms(slice(900,960),slice(0,3)),
    move_end_rotation_rms_rad=rms(slice(900,960),slice(3,6)),
    move_end_wrist_speed_rms_rad_s=rms(slice(900,960),slice(15,18)),
    final_wrist_speed_rms_rad_s=rms(slice(1200,1320),slice(15,18)),
    maximum_robot_obstacle_force_n=obstacle_force,
    maximum_torque_limit_fraction=(values[:,:,18:24].abs()/arm._torque_max).amax(dim=(0,2)))
held=finite&(metrics['initial_hold_bilateral_fraction']>=.8)
lifted=held&(cube_rise>=.04)&(metrics['carry_relative_position_max_m']<=.015)
carried=lifted&(cube_travel>=.04)&(metrics['carried_bilateral_fraction']>=.8)&(metrics['carry_relative_rotation_max_rad']<=math.radians(5))
released=carried&(release_drop>=.02)&(metrics['final_finger_cube_force_max_n']<.02)&(metrics['final_driver_angle_rad']<.1)
collision_free=released&(obstacle_force<.01)&~metrics['abnormal']
precise=collision_free&(metrics['lift_end_position_rms_m']<=.005)&(metrics['move_end_position_rms_m']<=.005)&(metrics['move_end_rotation_rms_rad']<=.035)&(metrics['move_end_wrist_speed_rms_rad_s']<=.1)
flags=dict(held=held,lifted=lifted,carried=carried,released=released,collision_free=collision_free,precise_and_settled=precise)
rows=[]
for i,gain in enumerate(gains):
    sl=slice(i*P,(i+1)*P)
    row=dict(gain=gain,counts={name:int(mask[sl].sum()) for name,mask in flags.items()},
        per_pose={key:value[sl].cpu().tolist() for key,value in metrics.items()},
        flags={key:value[sl].cpu().tolist() for key,value in flags.items()})
    rows.append(row)
    print('RESULT',json.dumps(dict(gain=gain,counts=row['counts'])),flush=True)
summary=dict(task='Saved-grasp lift, move, release in the UMI training scene',physics_hz=120,decimation=12,action_hz=10,
    environments=N,pose_count=P,gain_ids=gain_ids,simulation_seconds=11,wall_seconds=elapsed,
    controller_formula_max_error_nm=formula_error,results=rows,
    plan_path=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(runner_source).hexdigest(),
    criteria='Hold bilateral fraction >=0.8; lift and lateral carry >=40 mm for 50 mm commands; carry slip <=15 mm and <=5 deg; release fall >=20 mm, final cube/finger force <0.02 N and driver <0.1 rad. Collision-free additionally requires robot obstacle force <0.01 N and no abnormal speed. Precise/settled additionally requires endpoint position RMS <=5 mm, orientation RMS <=0.035 rad, and pre-release wrist speed RMS <=0.1 rad/s.')
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
env.close();app.close()
