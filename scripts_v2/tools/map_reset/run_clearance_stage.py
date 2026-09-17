"""Run an upstream recorder with map reset configuration; archive and canonicalize output."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import torch

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911') + '/49_clearance_and_placement')
PAIR = 'InsertiveAprilCube60__ReceptiveAprilCube60'


def clone_cpu(node):
    if isinstance(node, dict):
        return {k: clone_cpu(v) for k, v in node.items()}
    if isinstance(node, list):
        return [clone_cpu(v) for v in node]
    if isinstance(node, torch.Tensor):
        assert torch.isfinite(node).all()
        return node.detach().cpu().clone()
    raise TypeError(type(node))


def canonicalize(path, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    archive.write_bytes(raw)
    data = clone_cpu(torch.load(path, map_location='cpu', weights_only=False))
    temporary = path.with_suffix('.pt.tmp')
    torch.save(data, temporary)
    temporary.replace(path)
    return data, dict(raw_sha256=hashlib.sha256(raw).hexdigest(),
                     cpu_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--num-envs', type=int, default=8192)
    p.add_argument('--count', type=int, default=10000)
    p.add_argument('--output-subdir', default='production')
    p.add_argument('--dataset-dir', type=Path, default=ROOT / 'OmniReset')
    p.add_argument('--input-dir', type=Path, default=ROOT / 'OmniReset')
    args = p.parse_args()
    record_dir = ROOT / args.output_subdir
    record_dir.mkdir(parents=True, exist_ok=True)
    log_path = record_dir / (args.family + '.log')
    status_path = record_dir / (args.family + '.json')
    command = [sys.executable, 'scripts_v2/tools/record_reset_states.py', '--headless',
        '--device', f'cuda:{args.gpu}', '--task', 'OmniReset-UMI-Defaults-' + args.family + '-v0',
        '--num_envs', str(args.num_envs), '--num_reset_states', str(args.count), '--dataset_dir', str(args.dataset_dir)]
    for term in dict(
        ObjectAnywhereEEGrasped=['reset_end_effector_pose_from_grasp_dataset'],
        ObjectRestingEEGrasped=['reset_end_effector_pose_from_grasp_dataset', 'reset_insertive_object_pose_from_reset_states'],
        ObjectPartiallyAssembledEEGrasped=['reset_end_effector_pose_from_grasp_dataset', 'reset_insertive_object_pose_from_partial_assembly_dataset'],
    ).get(args.family, []):
        command.append(f'env.events.{term}.params.dataset_dir={args.input_dir}')
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONPATH=':'.join(str(REPO / 'source' / s) for s in ('uwlab', 'uwlab_assets', 'uwlab_tasks', 'uwlab_rl')))
    record = dict(family=args.family, command=command, physical_gpu=args.gpu, status='running',
                  dataset_dir=str(args.dataset_dir), log=str(log_path), started_at_utc=datetime.now(timezone.utc).isoformat())
    source_paths = [REPO / p for p in (
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/umi_reset_cfg.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/terminations.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_map_reset.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_sphere_geometry.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/actions/actions_cfg.py',
        'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/actions/task_space_actions.py',
        'source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/kinematics.py',
        'scripts_v2/tools/record_reset_states.py',
    )]
    record['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for path in source_paths:
        target = record_dir / 'source_snapshots' / args.family / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    inputs = [ROOT / 'lookup/lookup_manifest.json', ROOT / 'sphere_adapter_validation.json',
              ROOT / 'atlas/config.json',
              ROOT / 'atlas/robot_open_spheres.yml']
    if args.family != 'ObjectAnywhereEEAnywhere':
        inputs.append(args.input_dir / 'Grasps/InsertiveAprilCube60/grasps.pt')
    if args.family == 'ObjectPartiallyAssembledEEGrasped':
        inputs.append(args.input_dir / 'Resets' / PAIR / 'partial_assemblies.pt')
    if args.family == 'ObjectRestingEEGrasped':
        inputs.append(args.input_dir / 'Resets' / PAIR / 'resets_ObjectAnywhereEEAnywhere.pt')
    record['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    def save():
        tmp = status_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(status_path)
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = proc.pid
        save()
        code = proc.wait()
    try:
        if code:
            raise RuntimeError(f'Recorder exited with {code}')
        path = args.dataset_dir / 'Resets' / PAIR / f'resets_{args.family}.pt'
        data, hashes = canonicalize(path, record_dir / 'raw_gpu_recordings' / path.name)
        count = len(data['initial_state']['articulation']['robot']['joint_position'])
        assert count >= args.count, (count, args.count)
        record.update(hashes, saved_count=count, path=str(path))
        stats = [line.split('MAP_RESET_STATS ', 1)[1] for line in log_path.read_text(errors='replace').splitlines()
                 if 'MAP_RESET_STATS ' in line]
        assert stats, 'No map acceptance statistics in log'
        record['map_statistics'] = json.loads(stats[-1])
        assert record['map_statistics']['accepted'] == count
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        code = code or 1
    record.update(returncode=code, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(record), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
