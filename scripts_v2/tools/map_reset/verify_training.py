"""Verify four live ranks, current controller/data, finite W&B metrics, and checkpoint."""
import os
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import wandb

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared-run', type=Path, default=ROOT / '52_training/prepared_run.json')
args = parser.parse_args()
reference = json.loads(args.prepared_run.read_text())
output = Path(reference['output'])
metadata_path = output / 'run_metadata.json'
metadata = json.loads(metadata_path.read_text())
assert not (output / 'train.exit').exists(), 'Training has exited'
ranks = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        run_name = metadata['launch_command'][metadata['launch_command'].index('--run_name') + 1]
        if 'train.py' not in command or run_name not in command:
            continue
        values = dict(v.split('=', 1) for v in (directory / 'environ').read_text().split('\0') if '=' in v)
        if 'RANK' not in values or values.get('WANDB_RUN_ID') != metadata['wandb_run_id']:
            continue
        assert values['WORLD_SIZE'] == str(metadata['distributed_ranks'])
        assert values['CUDA_VISIBLE_DEVICES'] == metadata['launch_environment']['CUDA_VISIBLE_DEVICES']
        ranks.append(dict(pid=int(directory.name), rank=int(values['RANK']), local_rank=int(values['LOCAL_RANK']),
                          physical_gpu=metadata['physical_gpus'][int(values['LOCAL_RANK'])]))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert sorted(r['rank'] for r in ranks) == list(range(metadata['distributed_ranks'])), ranks
api = wandb.Api(timeout=30)
run_path = metadata['wandb_url'].removeprefix('https://wandb.ai/').replace('/runs/', '/')
run = api.run(run_path)
assert run.state == 'running', run.state
cfg = run.config
env, runner, policy = cfg['env_cfg'], cfg['runner_cfg'], cfg['policy_cfg']
assert int(env['scene']['num_envs']) == 16384
assert int(runner['max_iterations']) == 40000 and int(runner['num_steps_per_env']) == 32
assert runner['resume'] is False
assert policy['actor_hidden_dims'] == policy['critic_hidden_dims'] == [512, 256, 128, 64]
expected_ppo = dict(value_loss_coef=1., use_clipped_value_loss=True,
    normalize_advantage_per_mini_batch=False, clip_param=.2, entropy_coef=.006,
    num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4,
    schedule='adaptive', gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)
for key, value in expected_ppo.items():
    assert cfg['alg_cfg'][key] == value, (key, cfg['alg_cfg'][key], value)
assert policy['noise_std_type'] == 'gsde' and policy['activation'] == 'elu'
assert policy['actor_obs_normalization'] and policy['critic_obs_normalization']
assert env['terminations']['abnormal_robot'] is not None
arm = env['actions']['arm']
assert 'implicit_damping' not in arm
assert arm['motion_stiffness'] == [500., 500., 500., 60., 60., 60.]
kd = [2 * kp**.5 * ratio for kp, ratio in zip(arm['motion_stiffness'], arm['motion_damping_ratio'])]
assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(kd, [160., 160., 160., .1, .1, .1]))
assert env['events']['reset_from_reset_states']['params']['dataset_dir'] == metadata['dataset_dir']
assert int(env['decimation']) == 12 and math.isclose(float(env['sim']['dt']), 1/120)
assert env['scene']['receptive_object']['spawn']['rigid_props']['kinematic_enabled'] is True
assert env['scene']['insertive_object']['spawn']['rigid_props']['kinematic_enabled'] is False
assert 'CubeStackProgressContext' not in str(env['rewards']['progress_context']['func'])
assert 'ProgressContext' in str(env['rewards']['progress_context']['func'])
rows = run.history(samples=100, pandas=False)
steps = [row['_step'] for row in rows if '_step' in row]
assert steps and max(steps) >= 2, steps
loss_samples = [(k, v) for row in rows for k, v in row.items()
                if k.startswith('Loss/') and isinstance(v, (float, int))]
assert loss_samples and all(math.isfinite(v) for _, v in loss_samples), loss_samples
losses = dict(loss_samples)
assert any(k.startswith(('Train/', 'Episode_Reward/')) for row in rows for k in row)
folder = Path(cfg['log_dir'])
checkpoints = sorted(folder.glob('model_*.pt'), key=lambda p: p.stat().st_mtime)
assert checkpoints and checkpoints[-1].stat().st_size > 0
checkpoint = checkpoints[-1]
metadata.update(status='running_verified', training_log_dir=str(folder),
    live_rank_processes=sorted(ranks, key=lambda r: r['rank']),
    verified_utc=datetime.now(timezone.utc).isoformat(),
    verified_checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                             sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
keys = ['git_commit', 'git_branch', 'worktree', 'runtime_versions', 'tmux_session', 'local_log', 'local_metadata',
    'distributed_ranks', 'physical_gpus', 'num_envs_per_rank', 'total_environments', 'rollout_steps_per_environment',
    'samples_per_iteration', 'max_iterations', 'total_configured_samples', 'seed', 'fresh_policy',
    'supersedes_run_id', 'replacement_reason', 'code_snapshot', 'code_snapshot_sha256',
    'dataset_dir', 'dataset_counts', 'dataset_sha256', 'dataset_inputs', 'validation_files', 'validation_checks',
    'native_validation_summary', 'loaded_grasp_probe', 'controller', 'physics_hz', 'policy_hz',
    'cube_mass_range_kg', 'success_definition', 'reset_change_log', 'training_log_dir', 'verified_utc', 'verified_checkpoint']
run.config['provenance'] = {k: metadata[k] for k in keys}
run.notes = ('Fresh UMI cube policy using the regenerated atlas-seeded reset bank and selected explicit controller: '
    'translation Kp=500/Kd=160, rotation Kp=60/Kd=0.1 at 120 Hz. '
    'Original alignment reward, fixed lower cube and PPO settings; approved map/clearance/placement updates. '
    'Four GPUs, 16,384 environments per rank, 40,000 iterations. '
    'Source, data, and validation identities recorded in provenance.')
run.update()
api.flush()
readback = api.run('/'.join(run.path))
assert readback.config['provenance']['dataset_sha256'] == metadata['dataset_sha256']
assert readback.config['provenance']['code_snapshot_sha256'] == metadata['code_snapshot_sha256']
report = dict(verified_utc=metadata['verified_utc'], wandb_url=metadata['wandb_url'], wandb_state=readback.state,
    max_step=max(steps), latest_loss_metrics=losses, ranks=metadata['live_rank_processes'],
    budget_verified=True, controller_verified=True, data_verified=True, provenance_readback_verified=True,
    original_ppo_hyperparameters_verified=True,
    training_log_dir=str(folder), checkpoint=metadata['verified_checkpoint'])
(output / 'wandb_verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
