"""Freeze and optionally launch the validated fresh policy on four local GPUs."""
import os
import sys
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import wandb

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
DATA_ROOT = ROOT / '49_clearance_and_placement'
DATA = DATA_ROOT / 'OmniReset'
TRAINING_ROOT = ROOT / '52_training'
PYTHON = os.environ.get("THUNDER_PYTHON", sys.executable)
ENTITY = os.environ.get("WANDB_ENTITY", "")
PROJECT = 'uwlab-lab-cube-stack'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--supersedes-run-id')
    parser.add_argument('--reason', default='Corrected physical mounting and freshly regenerated 20 mm map/reset banks.')
    args = parser.parse_args()
    physical_gpus = [int(value) for value in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')]
    assert len(physical_gpus) == 4 and len(set(physical_gpus)) == 4, 'Select four distinct GPUs through CUDA_VISIBLE_DEVICES'
    audit = json.loads((DATA_ROOT / 'dataset_audit.json').read_text())
    native = json.loads((DATA_ROOT / 'native_validation.json').read_text())
    physics = json.loads((DATA_ROOT / 'training_physics.json').read_text())
    smoke = json.loads((TRAINING_ROOT / 'ppo_smoke_verification.json').read_text())
    assert audit['all_passed'] and len(audit['families']) == 4
    assert native['all_saved_states_valid'] and len(native['families']) == 4
    assert physics['success_definition'] == 'may_pose_alignment'
    expected_physics_checks = {
        'aligned_stack_accepted', 'alignment_reward_matches_may',
        'closed_hand_does_not_gate_alignment',
        'moving_aligned_pair_accepted', 'separate_cubes_rejected',
        'upper_cube_falls_under_gravity', 'lower_cube_fixed_under_gravity_and_force',
        'all_reset_families_finite', 'lower_cube_fixed_after_all_reset_families', 'loaded_lift_and_release',
    }
    assert set(physics['checks']) == expected_physics_checks and all(physics['checks'].values())
    assert physics['top']['dynamic'] and not physics['bottom']['dynamic']
    assert physics['bottom']['kinematic']
    assert physics['controller']['law'] == 'explicit_torque'
    assert physics['controller']['kp'] == [500., 500., 500., 60., 60., 60.]
    assert physics['controller']['kd'] == [160., 160., 160., .1, .1, .1]
    assert smoke['passed'] and smoke['iterations'] >= 10
    assert smoke['simulator_device'] == smoke['policy_device']
    assert smoke['distributed_ranks'] == 4 and smoke['num_envs'] == 16384
    assert smoke['success_definition'] == 'may_pose_alignment' and smoke['lower_cube_kinematic']
    assert smoke['original_ppo_hyperparameters_verified'] and smoke['checkpoint_finite']
    hashes = {f: v['sha256'] for f, v in audit['families'].items()}
    assert smoke['dataset_sha256'] == hashes
    assert physics['dataset_sha256'] == {'resets_'+f+'.pt': s for f, s in hashes.items()}
    for family, value in audit['families'].items():
        assert value['count'] >= 10000 and sha(value['path']) == value['sha256']
        assert native['families'][family]['sha256'] == value['sha256']
        assert native['families'][family]['count'] == value['count']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    session = 'umi_clearance_ppo4gpu_env16384_'+stamp
    output = TRAINING_ROOT / session
    output.mkdir(parents=True)
    run_id = wandb.util.generate_id()
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=4',
        'scripts/reinforcement_learning/rsl_rl/train.py', '--distributed', '--headless',
        '--task', 'OmniReset-UMI-Defaults-State-Train-v0', '--num_envs', '16384', '--max_iterations', '40000',
        '--logger', 'wandb', '--log_project_name', PROJECT, '--seed', '42',
        '--run_name', 'umi_mount20_baseline_4gpu_env16384', 'agent.experiment_name=umi_clearance_aprilcube60_mount20',
        'env.events.reset_from_reset_states.params.dataset_dir='+str(DATA)]
    environment = dict(CONDA_PREFIX=os.environ.get("CONDA_PREFIX", sys.prefix), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', WANDB_MODE='online', WANDB_ENTITY=ENTITY,
        WANDB_USERNAME=ENTITY, WANDB_RUN_ID=run_id, WANDB_RESUME='never',
        WANDB_TAGS='umi,aprilcube60,fresh-policy,clearance-resets,original-alignment-reward,fixed-lower-cube,explicit-controller',
        PYTHONPATH=':'.join(str(REPO / 'source' / s) for s in ('uwlab', 'uwlab_assets', 'uwlab_tasks', 'uwlab_rl')))
    launch = output / 'launch.sh'
    launch.write_text('#!/bin/bash\nset -u\ncd '+shlex.quote(str(REPO))+'\n'+
        '\n'.join('export '+k+'='+shlex.quote(v) for k, v in environment.items())+'\n'+
        shlex.join(command)+' > '+shlex.quote(str(output / 'train.log'))+' 2>&1\n'+
        'training_status=$?\nprintf "%s\\n" "$training_status" > '+shlex.quote(str(output / 'train.exit'))+
        '\nexit "$training_status"\n')
    launch.chmod(0o755)
    (output / 'tracked_changes.patch').write_text(git('diff', '--binary', 'HEAD')+'\n')
    files = sorted(set(git('ls-files', '-c', '-o', '--exclude-standard', '--',
        'source', 'scripts', 'scripts_v2', 'uwlab.sh', 'pyproject.toml',
        'umi_map_reset_changes.md', 'umi_default_pipeline_training.md').splitlines()))
    files = [f for f in files if (REPO / f).is_file()]
    with tarfile.open(output / 'code_snapshot.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(REPO / name, arcname=name, recursive=False)
    asset_dirs = [Path(os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets')) / d for d in (
        'thunder_d405_umi_rigid_asset', 'thunder_d405_umi_gripper_asset',
        'aprilcube_60mm_rounded/InsertiveAprilCube60', 'aprilcube_60mm_rounded/ReceptiveAprilCube60',
        'lab_vention_asset_v60')]
    assets = sorted({p for folder in asset_dirs for p in folder.rglob('*') if p.is_file()})
    validation_files = [DATA_ROOT / p for p in ('dataset_audit.json', 'native_validation.json',
        'training_physics.json', 'lookup/lookup_manifest.json', 'atlas/summary.json',
        'sphere_adapter_validation.json', 'fresh_inputs.json', 'production_runs.json',
        'source_integrity_validation.json')]
    validation_files += sorted(DATA_ROOT.glob('*_native_recheck_exclusions.json'))
    validation_files += sorted(DATA_ROOT.glob('*boundary_diagnosis*.json'))
    validation_files.append(TRAINING_ROOT / 'ppo_smoke_verification.json')
    versions = {}
    for name in ('torch', 'isaacsim', 'isaaclab', 'rsl-rl-lib', 'wandb'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'workspace source'
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
        git_status=git('status', '--short'), worktree=str(REPO), runtime_versions=versions,
        tmux_session=session, launch_command=command, launch_environment=environment,
        local_log=str(output / 'train.log'), local_metadata=str(output / 'run_metadata.json'),
        distributed_ranks=4, physical_gpus=physical_gpus, num_envs_per_rank=16384,
        total_environments=65536, rollout_steps_per_environment=32, samples_per_iteration=2097152,
        max_iterations=40000, total_configured_samples=83886080000, seed=42,
        task='OmniReset-UMI-Defaults-State-Train-v0', fresh_policy=True,
        supersedes_run_id=args.supersedes_run_id, replacement_reason=args.reason,
        wandb_run_id=run_id, wandb_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}',
        code_snapshot=str(output / 'code_snapshot.tar.gz'), code_snapshot_sha256=sha(output / 'code_snapshot.tar.gz'),
        source_sha256={name: sha(REPO / name) for name in files},
        asset_files=[dict(path=str(p), sha256=sha(p)) for p in assets],
        dataset_dir=str(DATA), dataset_counts={f: v['count'] for f, v in audit['families'].items()},
        dataset_sha256=hashes, generation_runs=json.loads((DATA_ROOT / 'production_runs.json').read_text()),
        dataset_inputs=json.loads((DATA_ROOT / 'fresh_inputs.json').read_text()),
        validation_files={str(p): sha(p) for p in validation_files}, validation_checks=physics['checks'],
        native_validation_summary={f: dict(count=v['count'], saved_geometry_valid=v['saved_geometry_valid'],
            maximum_reload_error=v['maximum_reload_error']) for f, v in native['families'].items()},
        loaded_grasp_probe={k: physics['loaded_grasp_probe'][k] for k in ('environments', 'lifted', 'released')},
        controller=physics['controller'], physics_hz=120, policy_hz=10, cube_mass_range_kg=[.02, .20],
        cube_motion_modes={'insertive_object': 'dynamic', 'receptive_object': 'kinematic'},
        reset_change_log=str(REPO / 'umi_map_reset_changes.md'),
        success_definition='Original May pose alignment using the selected cube metadata thresholds; reward terms and weights unchanged.',
        status='prepared')
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    reference = dict(output=str(output), tmux_session=session, wandb_run_id=run_id,
                     wandb_url=metadata['wandb_url'], status='prepared')
    if args.launch:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash '+shlex.quote(str(launch))], check=True)
        metadata.update(status='launched', launched_utc=datetime.now(timezone.utc).isoformat())
        (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        reference['status'] = 'launched'
    (TRAINING_ROOT / 'prepared_run.json').write_text(json.dumps(reference, indent=2)+'\n')
    print(json.dumps(reference, indent=2))


if __name__ == '__main__':
    main()
