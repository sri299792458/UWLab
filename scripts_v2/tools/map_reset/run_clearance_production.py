"""Record fresh reset banks in separate shards, then merge without filtering.

Uses the original recorder and sampling budgets. All shard records, raw files,
source snapshots and merge hashes are retained alongside the final CPU banks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917') + '/49_clearance_and_placement')
DATA = ROOT / 'OmniReset'
PAIR = 'InsertiveAprilCube60__ReceptiveAprilCube60'
GENERAL = 'ObjectAnywhereEEAnywhere'
AIR = 'ObjectAnywhereEEGrasped'
PARTIAL = 'ObjectPartiallyAssembledEEGrasped'
REST = 'ObjectRestingEEGrasped'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(nodes):
    first = nodes[0]
    if isinstance(first, dict):
        assert all(n.keys() == first.keys() for n in nodes)
        return {key: combine([n[key] for n in nodes]) for key in first}
    assert isinstance(first, list)
    result = [value for node in nodes for value in node]
    assert all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and torch.isfinite(v).all() for v in result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 6])
    args = parser.parse_args()
    assert len(set(args.gpus)) == len(args.gpus)
    pilot = json.loads((ROOT / 'pilot_validation.json').read_text())
    assert pilot['all_saved_states_valid']
    assert len(pilot['families']) == 4
    records, processes, merged = {}, {}, {}
    env = os.environ.copy()
    env.update(CONDA_PREFIX=os.environ.get("CONDA_PREFIX", sys.prefix), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def save():
        value = dict(shards=records, merged=merged,
                     complete=len(merged) == 4, target_per_family=10000)
        temp = ROOT / 'production_runs.json.tmp'
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(ROOT / 'production_runs.json')

    def launch(family, shard, gpu, count, num_envs=4096):
        key = family + '_shard' + str(shard)
        shard_data = ROOT / 'production_shards' / key / 'OmniReset'
        command = [sys.executable, 'scripts_v2/tools/map_reset/run_clearance_stage.py',
            '--family', family, '--gpu', str(gpu), '--num-envs', str(num_envs),
            '--count', str(count), '--dataset-dir', str(shard_data),
            '--input-dir', str(DATA), '--output-subdir', 'production_stage/' + key]
        with (ROOT / ('production_' + key + '_supervisor.log')).open('w') as stream:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        processes[key] = proc
        records[key] = dict(family=family, shard=shard, command=command, physical_gpu=gpu,
            pid=proc.pid, process_group=proc.pid, status='running',
            dataset_dir=str(shard_data), stage_record=str(ROOT / 'production_stage' / key / (family + '.json')))
        save()

    def merge(family):
        sources = [r for r in records.values() if r['family'] == family]
        expected_shards = {GENERAL: 1, AIR: 3, PARTIAL: 3, REST: 4}[family]
        if len(sources) != expected_shards or not all(r['status'] == 'complete' for r in sources):
            return
        paths = [Path(r['dataset_dir']) / 'Resets' / PAIR / ('resets_' + family + '.pt') for r in sources]
        banks = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
        data = combine(banks)
        count = len(data['initial_state']['articulation']['robot']['joint_position'])
        assert count >= 10000, (family, count)
        target = DATA / 'Resets' / PAIR / ('resets_' + family + '.pt')
        assert not target.exists(), f'Refusing to replace an existing final bank: {target}'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pt.tmp')
        torch.save(data, temporary)
        temporary.replace(target)
        merged[family] = dict(path=str(target), count=count, sha256=sha(target),
            sources=[dict(path=str(p), sha256=sha(p)) for p in paths],
            operation='Concatenate every saved row in shard order; no filtering, deduplication or truncation.')
        save()
        print('MERGED ' + family + ' ' + str(count), flush=True)

    try:
        pending = [(GENERAL, 0, 10000)]
        pending += [(family, shard, 3334) for shard in range(3) for family in (AIR, PARTIAL)]
        resting_queued = False
        while True:
            for key, proc in list(processes.items()):
                code = proc.poll()
                if code is not None and records[key]['status'] == 'running':
                    records[key].update(status='complete' if code == 0 else 'failed', returncode=code)
                    save()
                    if code:
                        raise RuntimeError(f'{key} failed with {code}')
            for family in (GENERAL, AIR, PARTIAL, REST):
                if family not in merged:
                    merge(family)
            if GENERAL in merged and not resting_queued:
                pending += [(REST, shard, 2500) for shard in range(4)]
                resting_queued = True
            for gpu in args.gpus:
                busy = any(r['physical_gpu'] == gpu and r['status'] == 'running' for r in records.values())
                if pending and not busy:
                    family, shard, count = pending.pop(0)
                    launch(family, shard, gpu, count, num_envs=2048 if gpu == 2 else 4096)
            if len(merged) == 4:
                break
            time.sleep(3)
    except BaseException:
        for proc in processes.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        raise
    print('ALL_FOUR_FRESH_BANKS_RECORDED', flush=True)


if __name__ == '__main__':
    main()
