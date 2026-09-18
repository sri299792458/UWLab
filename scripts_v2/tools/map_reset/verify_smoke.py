"""Verify a completed short PPO run against its actual saved config and metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--log', type=Path, required=True)
p.add_argument('--run-name', required=True)
p.add_argument('--dataset-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--require-same-device', action='store_true')
p.add_argument('--distributed-ranks', type=int, default=1)
args = p.parse_args()
repo = Path(__file__).resolve().parents[3]
log = args.log.read_text(errors='replace')
iterations = [int(n) for n in re.findall(r'Learning iteration\s+(\d+)/', log)]
assert iterations and max(iterations) >= 9
assert 'Traceback (most recent call last)' not in log and 'Error executing job' not in log
folders = sorted((repo / 'logs/rsl_rl/umi_map_reset_smoke').glob('*_'+args.run_name), key=lambda p: p.stat().st_mtime)
folders = [folder for folder in folders if any(folder.glob('events.out.tfevents.*'))]
assert folders, 'No TensorBoard output for the requested run name'
folder = folders[-1]
env = yaml.load((folder / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
agent = yaml.load((folder / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
arm = env['actions']['arm']
assert 'implicit_damping' not in arm
kp = [float(x) for x in arm['motion_stiffness']]
ratio = [float(x) for x in arm['motion_damping_ratio']]
kd = [2 * k**.5 * r for k, r in zip(kp, ratio)]
assert kp == [500., 500., 500., 60., 60., 60.] and kd == [160., 160., 160., .1, .1, .1]
assert env['events']['reset_from_reset_states']['params']['dataset_dir'] == str(args.dataset_dir)
assert env['terminations']['abnormal_robot']['func'].endswith(':abnormal_robot_state')
assert int(agent['max_iterations']) == 10 and int(agent['num_steps_per_env']) == 32
assert agent['resume'].lower() == 'false'
if args.require_same_device:
    assert agent['device'] == env['sim']['device'], (agent['device'], env['sim']['device'])
assert int(env['decimation']) == 12 and math.isclose(float(env['sim']['dt']), 1/120)
assert env['scene']['receptive_object']['spawn']['rigid_props']['kinematic_enabled'].lower() == 'true'
assert env['scene']['insertive_object']['spawn']['rigid_props']['kinematic_enabled'].lower() == 'false'
assert env['rewards']['progress_context']['func'].endswith(':ProgressContext')
assert env['rewards']['success_reward']['func'].endswith(':success_reward')
assert env['commands']['task_command']['class_type'].endswith(':TaskCommand')
expected_rewards = dict(action_magnitude=-1e-4, action_rate=-1e-3, joint_vel=-1e-2,
    abnormal_robot=-100., progress_context=.1, ee_asset_distance=.1, dense_success_reward=.1, success_reward=1.)
assert set(env['rewards']) == set(expected_rewards)
for name, weight in expected_rewards.items():
    assert float(env['rewards'][name]['weight']) == weight
assert float(env['rewards']['ee_asset_distance']['params']['std']) == 1.
assert float(env['rewards']['dense_success_reward']['params']['std']) == 1.
assert [float(x) for x in env['events']['reset_from_reset_states']['params']['probs']] == [.25]*4
for name in ('insertive_object', 'receptive_object'):
    params = env['events']['randomize_'+name+'_mass']['params']
    assert [float(x) for x in params['mass_distribution_params']] == [.02, .20]
devices = sorted(set(re.findall(r'Environment device\s*:\s*(cuda:\d+)', log)))
if args.distributed_ranks > 1:
    assert devices == [f'cuda:{rank}' for rank in range(args.distributed_ranks)], devices
policy, algorithm = agent['policy'], agent['algorithm']
assert [int(x) for x in policy['actor_hidden_dims']] == [512, 256, 128, 64]
assert policy['actor_hidden_dims'] == policy['critic_hidden_dims']
assert policy['noise_std_type'] == 'gsde' and policy['activation'] == 'elu'
assert policy['actor_obs_normalization'].lower() == policy['critic_obs_normalization'].lower() == 'true'
expected_ppo = dict(value_loss_coef=1., clip_param=.2, entropy_coef=.006,
    num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4,
    gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)
for key, value in expected_ppo.items():
    assert float(algorithm[key]) == value, (key, algorithm[key], value)
assert algorithm['use_clipped_value_loss'].lower() == 'true'
assert algorithm['normalize_advantage_per_mini_batch'].lower() == 'false'
assert algorithm['schedule'] == 'adaptive'
checkpoints = sorted(folder.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
assert checkpoints and int(checkpoints[-1].stem.split('_')[-1]) >= 9
checkpoint = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)

def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True

assert finite_tensors(checkpoint), 'Nonfinite tensor in checkpoint or optimizer state'
events = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
metrics = {}
for tag in events.Tags()['scalars']:
    values = events.Scalars(tag)
    assert all(math.isfinite(v.value) for v in values), tag
    metrics[tag] = dict(step=values[-1].step, value=values[-1].value, records=len(values))
assert any(k.startswith('Loss/') for k in metrics) and max(v['step'] for v in metrics.values()) >= 9
report = dict(passed=True, iterations=max(iterations)+1, num_envs=int(env['scene']['num_envs']),
    distributed_ranks=args.distributed_ranks, simulator_devices_observed=devices,
    success_definition='may_pose_alignment', lower_cube_kinematic=True,
    original_ppo_hyperparameters_verified=True, checkpoint_finite=True,
    checkpoint=dict(path=str(checkpoints[-1]), sha256=hashlib.sha256(checkpoints[-1].read_bytes()).hexdigest()),
    simulator_device=env['sim']['device'], policy_device=agent['device'],
    training_log_dir=str(folder), metrics=metrics, controller=dict(law='explicit_torque', kp=kp, kd=kd),
    log=str(args.log), log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
    dataset_dir=str(args.dataset_dir), dataset_sha256={f.stem.removeprefix('resets_'): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (args.dataset_dir / 'Resets/InsertiveAprilCube60__ReceptiveAprilCube60').glob('resets_*.pt')})
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(passed=True, iterations=report['iterations'], log_dir=str(folder),
    losses={k: v for k, v in metrics.items() if k.startswith('Loss/')})))
