"""Render recorded handling trajectories without advancing physics."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--poses',default='4,25')
parser.add_argument('--gain-index',type=int,default=0)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args,multi_gpu=False).app

import hashlib,json
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.managers import EventTermCfg,TerminationTermCfg
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg

folder=args.run/'worker_0'
snapshot=json.loads((folder/'runtime_inputs.json').read_text())
data=torch.load(folder/'contact_trajectories.pt',map_location='cpu',weights_only=True)
cfg=UmiCubeTrainCfg();cfg.scene.num_envs=1;cfg.sim.device=args.device
cfg.sim.dt=1/120;cfg.decimation=12;cfg.sim.render_interval=12
cfg.viewer.resolution=(960,720)
for name,term in list(vars(cfg.events).items()):
    if isinstance(term,EventTermCfg):setattr(cfg.events,name,None)
for name,term in list(vars(cfg.terminations).items()):
    if isinstance(term,TerminationTermCfg):setattr(cfg.terminations,name,None)
for name in ['insertive_object','receptive_object']:
    getattr(cfg.scene,name).init_state.pos=(2.,2.,2.)
env=gym.make('OmniReset-UMI-Defaults-State-Train-v0',cfg=cfg,render_mode='rgb_array').unwrapped
env.reset();robot=env.scene['robot'];cube=env.scene['insertive_object']
assert robot.joint_names==data['joint_names']
assert hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest()==snapshot['robot_usd_sha256']
root=torch.tensor(snapshot['intended_root_pose_world'][0],device=env.device).reshape(1,7)
root[:,:3]=torch.tensor(cfg.scene.robot.init_state.pos,device=env.device)
robot.write_root_pose_to_sim(root);robot.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
out=args.run/'videos';out.mkdir(exist_ok=True)
report=dict(source=str(folder/'contact_trajectories.pt'),physics_steps_during_render=0,fps=20,clips=[])
if (out/'manifest.json').exists():
    previous=json.loads((out/'manifest.json').read_text())
    assert previous['source']==report['source']
    report['clips']=previous['clips']
for pose_index in map(int,args.poses.split(',')):
    index=args.gain_index*data['pose_count']+pose_index
    offset=torch.tensor(snapshot['initial_root_pose_world'][index][:3])-root[0,:3].cpu()
    initial=data['cube_poses_world'][0,index,:3]-offset
    target=initial.numpy()+np.array([0.,0.,-.035])
    eye=target+np.array([.60,-.85,.45])
    env.sim.set_camera_view(eye,target)
    path=out/f"gain{data['gain_ids'][args.gain_index]}_posture{pose_index//3}_mass{int(round(snapshot['poses'][pose_index]['cube_mass_kg']*1000))}g.mp4"
    writer=imageio.get_writer(path,fps=20,codec='libx264',quality=8,macro_block_size=16)
    maximum_body_error=0.;count=0
    for step in range(0,len(data['joint_positions']),6):
        q=data['joint_positions'][step,index].to(env.device)[None]
        qd=data['joint_velocities'][step,index].to(env.device)[None]
        robot.write_joint_state_to_sim(q,qd)
        pose=data['cube_poses_world'][step,index].clone();pose[:3]-=offset
        cube.write_root_pose_to_sim(pose.to(env.device)[None])
        cube.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
        env.sim.forward();env.scene.update(0.)
        expected=data['body_poses_world'][step,index,:,:3]-offset
        maximum_body_error=max(maximum_body_error,float((robot.data.body_pos_w[0].cpu()-expected).norm(dim=-1).max()))
        for _ in range(12 if step==0 else 2):frame=env.render()
        if step==0:imageio.imwrite(path.with_suffix('.png'),frame)
        writer.append_data(frame);count+=1
    writer.close()
    clip=dict(path=str(path),gain_id=data['gain_ids'][args.gain_index],pose_index=pose_index,
        posture_index=pose_index//3,mass_kg=snapshot['poses'][pose_index]['cube_mass_kg'],
        frames=count,maximum_reconstructed_body_position_error_m=maximum_body_error,
        description='Playback of recorded joint and cube states, every sixth 120 Hz sample; no new dynamics.')
    report['clips']=[old for old in report['clips'] if old['path']!=clip['path']]+[clip]
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RENDERED',json.dumps(clip),flush=True)
env.close();app.close()
