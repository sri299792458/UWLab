"""Audit recorded open-hand motion with the accepted cuRobo sphere model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

p=argparse.ArgumentParser();p.add_argument('--phase',default='holdout');args=p.parse_args()
root=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
out=root/'22_explicit_gain_sweep';model_root=root/'21_sphere_reachability'
model=yaml.safe_load((model_root/'robot_open_spheres.yml').read_text())['robot_cfg']
cfg=json.loads((model_root/'config.json').read_text())
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=model,
    scene_model=cfg['scene_model'],collision_activation_distance=0.,self_collision_activation_distance=0.))
base=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link_shape_0')
results=[]
for folder in sorted((out/args.phase).glob('worker_*')):
    data=torch.load(folder/'trajectories.pt',map_location='cpu',weights_only=True)
    values=data['values'];q=values[:,:,6:12].reshape(-1,6)
    finite=torch.isfinite(q).all(dim=1)
    self_bad=torch.ones(len(q),dtype=torch.bool);world_bad=self_bad.clone()
    rows=torch.nonzero(finite,as_tuple=True)[0]
    for start in range(0,len(rows),2048):
        take=rows[start:start+2048]
        sph=checker.get_kinematics(q[take].cuda()[:,None]).robot_spheres.clone();sph[...,3]+=.001
        self_bad[take]=(checker.get_self_collision_distance(sph).reshape(len(take),-1).amax(dim=1)>0).cpu()
        sph[:,:,base,3]=-100.
        world_bad[take]=(checker.get_collision_constraint(sph).reshape(len(take),-1).amax(dim=1)>0).cpu()
    self_bad=self_bad.reshape(values.shape[:2]);world_bad=world_bad.reshape(values.shape[:2])
    torch.save(dict(self_blocked=self_bad,world_blocked=world_bad),folder/'sphere_collision_flags.pt')
    P=data['pose_count']
    for i,gain_id in enumerate(data['gain_ids']):
        a=self_bad[:,i*P:(i+1)*P];b=world_bad[:,i*P:(i+1)*P]
        results.append(dict(gain_id=gain_id,pose_count=P,any_self_blocked=int(a.any(dim=0).sum()),
            any_world_blocked=int(b.any(dim=0).sum()),any_blocked=int((a|b).any(dim=0).sum()),
            initial_step_blocked=int((a[0]|b[0]).sum()),blocked_pose_indices=torch.nonzero((a|b).any(dim=0),as_tuple=True)[0].tolist()))
    print('AUDITED',folder.name,flush=True)
(out/f'{args.phase}_sphere_collision_audit.json').write_text(json.dumps(dict(phase=args.phase,
    scope='Every recorded 120 Hz joint configuration, with open-hand geometry and the accepted 1 mm-per-sphere margin; not continuous collision proof.',results=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
