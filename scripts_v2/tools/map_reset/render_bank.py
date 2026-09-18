"""Render exact saved bank rows without stepping physics; save reproducible row IDs."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--samples', type=int, default=3)
p.add_argument('--dataset-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args, multi_gpu=False).app

import json
import hashlib
import numpy as np
import torch
import gymnasium as gym
import imageio.v3 as iio
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg

OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
DATA = args.dataset_dir / 'Resets/InsertiveAprilCube60__ReceptiveAprilCube60'
FAMILIES = ['ObjectAnywhereEEAnywhere', 'ObjectRestingEEGrasped', 'ObjectAnywhereEEGrasped', 'ObjectPartiallyAssembledEEGrasped']
cfg = UmiCubeTrainCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
cfg.seed = 73
cfg.events.reset_from_reset_states = None
cfg.terminations.abnormal_robot = None
cfg.viewer.resolution = (960, 720)
cfg.viewer.eye = (1.2, -1.4, 1.65)
cfg.viewer.lookat = (.25, -.1, 1.05)
env = gym.make('OmniReset-UMI-Defaults-State-Train-v0', cfg=cfg, render_mode='rgb_array').unwrapped
env.reset()
robot = env.scene['robot']
report = {'seed':73, 'physics_steps_per_image':0, 'source_controller':'selected explicit gains: translation Kp500/Kd160, rotation Kp60/Kd0.1', 'samples':[], 'joint_names':robot.joint_names, 'body_names':robot.body_names}
rng = np.random.default_rng(73)

def take_state(d, row):
    if isinstance(d, dict):
        return {k: take_state(v,row) for k,v in d.items()}
    return torch.as_tensor(d[row], device=env.device).unsqueeze(0)

def capture(eye, target, path):
    env.sim.set_camera_view(eye, target)
    for _ in range(12):
        frame = env.render()
    iio.imwrite(path, frame)

for family in FAMILIES:
    path = DATA / f'resets_{family}.pt'
    bank = torch.load(path, map_location='cpu', weights_only=False)['initial_state']
    count = len(bank['articulation']['robot']['joint_position'])
    rows = rng.choice(count, size=args.samples, replace=False).tolist()
    for index,row in enumerate(rows):
        state = take_state(bank,row)
        env.scene.reset_to(state, is_relative=True)
        env.sim.forward()
        env.scene.update(0.0)
        top = state['rigid_object']['insertive_object']['root_pose'][0,:3].cpu().numpy()
        bottom = state['rigid_object']['receptive_object']['root_pose'][0,:3].cpu().numpy()
        target = top + np.array([0,0,.04])
        stem = f'{family}_row{row}'
        capture(target+np.array([.42,-.52,.29]), target, OUT / f'{stem}_close.png')
        capture(target+np.array([-.42,.46,.23]), target, OUT / f'{stem}_reverse.png')
        if index == 0:
            capture((1.2,-1.4,1.65),(.25,-.1,1.05), OUT/f'{stem}_overview.png')
        actual = env.scene.get_state(is_relative=True)
        err = max(float((actual['articulation']['robot']['joint_position']-state['articulation']['robot']['joint_position']).abs().max()),float((actual['rigid_object']['insertive_object']['root_pose']-state['rigid_object']['insertive_object']['root_pose']).abs().max()))
        entry = {'family':family, 'row':row, 'count':count, 'file':str(path), 'top_xyz':top.tolist(), 'bottom_xyz':bottom.tolist(), 'render_state_max_error':err, 'joint_positions':state['articulation']['robot']['joint_position'][0].cpu().tolist(), 'body_poses':robot.data.body_pose_w[0].cpu().tolist(), 'close_image':str(OUT/f'{stem}_close.png'), 'reverse_image':str(OUT/f'{stem}_reverse.png')}
        report['samples'].append(entry)
        (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print('CAPTURED',family,row,'error',err,flush=True)
    report.setdefault('sha256',{})[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',flush=True)
env.close()
app.close()
