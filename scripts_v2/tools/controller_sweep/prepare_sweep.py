"""Freeze the 120 Hz explicit-controller gain comparison and map-derived poses."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
OUT=ROOT/'22_explicit_gain_sweep';OUT.mkdir(exist_ok=True)
MAP=ROOT/'21_sphere_reachability'
cfg=json.loads((MAP/'config.json').read_text())
with np.load(MAP/'atlas.npz') as data:status=data['status']
rng=np.random.default_rng(7391)
screen_counts=[4,3,3,2,2,1,1];holdout_counts=[20,15,15,10,10,5,5]
pool=[]
for group,tilt in enumerate(range(0,181,30)):
    orientations=[o['index'] for o in cfg['orientations'] if o['tilt_deg']==tilt]
    for _ in range(2048):
        h=int(rng.integers(6,25));o=int(rng.choice(orientations))
        y=int(rng.integers(2,len(cfg['y_world_m'])-2));x=int(rng.integers(2,len(cfg['x_world_m'])-2))
        if status[h,o,y,x]!=4:continue
        pool.append(dict(group=group,height_index=h,orientation_index=o,y_index=y,x_index=x))
        if sum(p['group']==group for p in pool)>=256:break
cache={};q=[]
for p in pool:
    key=(p['height_index'],p['orientation_index'])
    if key not in cache:
        with np.load(MAP/'slices'/f'h{key[0]:02d}_o{key[1]:03d}.npz') as data:cache[key]=data['chosen_q']
    q.append(cache[key][p['y_index'],p['x_index']])
q=np.asarray(q)
model=yaml.safe_load((MAP/'robot_open_spheres.yml').read_text())['robot_cfg']
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
    robot_config=model,scene_model=cfg['scene_model'],collision_activation_distance=0.,self_collision_activation_distance=0.))
spheres=checker.get_kinematics(torch.tensor(q,device='cuda')[:,None]).robot_spheres.clone()
spheres[...,3]+=.001
self_bad=checker.get_self_collision_distance(spheres).reshape(len(q),-1).amax(dim=1)>0
# Empty-hand screening targets reserve additional space around the hand for
# the small Cartesian motion probe. This does not change map acceptance.
hand_links=[name for name in model['kinematics']['collision_link_names']
    if any(k in name for k in ['finger','knuckle','robotiq','camera','wrist_3'])]
for name in hand_links:
    ids=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name(name)
    spheres[:,:,ids,3]+=.049
ids=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link_shape_0')
spheres[:,:,ids,3]=-100.
world_bad=checker.get_collision_constraint(spheres).reshape(len(q),-1).amax(dim=1)>0
valid=(~self_bad&~world_bad).cpu().numpy()
limits=np.asarray(cfg['joint_limits_rad'])
valid&=(q>limits[0]+.2).all(axis=1)&(q<limits[1]-.2).all(axis=1)
valid&=(np.abs(np.sin(q[:,2]))>.2)&(np.abs(np.sin(q[:,4]))>.25)
screen=[];holdout=[]
for group in range(7):
    available=[i for i,p in enumerate(pool) if p['group']==group and valid[i]]
    needed=screen_counts[group]+holdout_counts[group]
    assert len(available)>=needed,(group,len(available),needed)
    selected=rng.choice(available,size=needed,replace=False)
    screen.extend(selected[:screen_counts[group]]);holdout.extend(selected[screen_counts[group]:])
records=[]
for split,indices in [('screen',screen),('holdout',holdout)]:
    for i in indices:
        p=pool[i];h,o,y,x=[p[k] for k in ['height_index','orientation_index','y_index','x_index']]
        records.append(dict(**p,split=split,q=q[i].tolist(),
            world_position_m=[cfg['x_world_m'][x],cfg['y_world_m'][y],cfg['tabletop_z_m']+cfg['height_above_table_m'][h]],
            orientation=cfg['orientations'][o]))
assert len({tuple(r[k] for k in ['height_index','orientation_index','y_index','x_index']) for r in records})==len(records)
(OUT/'poses.json').write_text(json.dumps(dict(seed=7391,poses=records,
    selection='Map-passing poses at 14–50 cm, at least 50 mm hand/world sphere clearance, 0.2 rad joint-limit margin, away from elbow/wrist singularities. Same 1 mm self-pair padding as accepted map.',
    screen_count=len(screen),holdout_count=len(holdout)),indent=2)+'\n')
gains=[]
def add(label,pt,dt,pr,dr):
    gains.append(dict(id=len(gains),label=label,kp=[pt]*3+[pr]*3,kd=[dt]*3+[dr]*3))
add('original_training',200,2*np.sqrt(200)*3,3,2*np.sqrt(3))
add('original_terminal',1000,2*np.sqrt(1000),50,2*np.sqrt(50))
add('prior_reduced_rotation_damping',200,2*np.sqrt(200)*3,3,2*np.sqrt(3)*.05)
add('prior_soft_rotation',200,2*np.sqrt(200)*3,.3,2*np.sqrt(3)*.05)
for pt,dt,pr,dr in itertools.product([100.,200.,400.],[20.,40.,2*np.sqrt(200)*3],[.3,1.,3.],[.1,.2,.4,.8]):
    add(f'pt{pt:g}_dt{dt:g}_pr{pr:g}_dr{dr:g}',pt,dt,pr,dr)
sources=[OUT/'poses.json',MAP/'robot_open_spheres.yml',MAP/'config.json',
    Path('/data/kanth042/converted_assets/thunder_d405_umi_rigid_asset/ur5e_robotiq_d405_umi_rigid_thunder.usd'),
    Path('/data/kanth042/converted_assets/thunder_d405_umi_rigid_asset/metadata.yaml')]
repo=Path('/data/kanth042/repos/UWLab-reset-from-defaults')
sources.extend([repo/'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/actions/task_space_actions.py',
    repo/'source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/kinematics.py'])
plan=dict(physics_hz=120,decimation=12,action_hz=10,controller='existing explicit Cartesian J-transpose PD',
    randomization=False,hand='open and empty for first screen',gains=gains,
    trajectory=dict(initial_hold_s=2,motion_s=6,final_hold_s=2,position_amplitude_m=.02,rotation_amplitude_rad=.1,frequency_hz=.5),
    ranking_limits=dict(motion_position_rms_m=.005,motion_rotation_rms_rad=.035,final_wrist_speed_rms_rad_s=.1,final_position_rms_m=.002,final_rotation_rms_rad=.02),
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(gain_settings=len(gains),screen_poses=len(screen),holdout_poses=len(holdout),available_clear=int(valid.sum())),indent=2))
