"""Consolidate identical independent resting-grasp batches after 10,000 successes.

Only already accepted recorder rows are used. Capture a readable snapshot before
stopping a live writer, retain source files and identities, then concatenate rows.
"""
from datetime import datetime, timezone
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import signal
import time
import torch
from run_stage import ROOT, PAIR, clone_cpu

FAMILY = 'ObjectRestingEEGrasped'
MAIN = ROOT / 'production' / (FAMILY+'.json')
EXTRA = ROOT / 'shards/resting_gpu4' / (FAMILY+'.json')
OUT = ROOT / 'resting_consolidation.json'


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def count(data):
    return len(data['initial_state']['articulation']['robot']['joint_position'])


def concatenate(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return {k: concatenate(a[k], b[k]) for k in a}
    return a+b


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def stop_writer(record):
    pid = record['pid']
    if not alive(pid):
        return
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert b'scripts_v2/tools/record_reset_states.py' in args
    assert ('OmniReset-UMI-Defaults-'+FAMILY+'-v0').encode() in args
    assert record['dataset_dir'].encode() in args
    parent = int(Path(f'/proc/{pid}/stat').read_text().split()[3])
    os.kill(pid, signal.SIGINT)
    started = time.monotonic()
    while alive(pid):
        if time.monotonic()-started > 30:
            os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    # Its wrapper writes an exit record; wait before replacing that record with
    # the consolidated dataset result so no two writers race on the JSON file.
    while alive(parent):
        command = Path(f'/proc/{parent}/cmdline').read_bytes()
        if b'map_reset/run_stage.py' not in command:
            break
        time.sleep(1)


def main():
    while True:
        main = json.loads(MAIN.read_text())
        extra = json.loads(EXTRA.read_text())
        if main['status'] == 'complete':
            if extra['status'] == 'running':
                stop_writer(extra)
                extra = json.loads(EXTRA.read_text())
                extra['status'] = 'stopped_no_rows_used'
                save(EXTRA, extra)
            save(OUT, dict(status='not_needed', reason='Main recorder completed its full target first.'))
            return
        assert main['status'] == 'running', main
        assert extra['status'] in ('running', 'complete'), extra
        if extra['status'] != 'complete':
            time.sleep(15)
            continue
        assert main['source_sha256'] == extra['source_sha256']
        assert main['input_sha256'] == extra['input_sha256']
        path = Path(main['dataset_dir']) / 'Resets' / PAIR / f'resets_{FAMILY}.pt'
        try:
            raw = path.read_bytes()
            first = clone_cpu(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False))
        except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError):
            time.sleep(2)
            continue
        save(OUT, dict(status='waiting_for_combined_target', main_snapshot_count=count(first),
                       extra_completed_count=extra['saved_count']))
        if count(first)+extra['saved_count'] < 10000:
            time.sleep(15)
            continue
        second = torch.load(extra['path'], map_location='cpu', weights_only=False)
        assert count(second) == extra['saved_count']
        archive = ROOT / 'production/raw_gpu_recordings' / f'{FAMILY}_main_snapshot.pt'
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(raw)
        parts = [dict(kind='accepted_live_snapshot', count=count(first), path=str(archive),
                      sha256=hashlib.sha256(raw).hexdigest(), recorder=copy.deepcopy(main)),
                 dict(kind='completed_independent_batch', count=count(second), path=extra['path'],
                      sha256=extra['cpu_sha256'], recorder=extra)]
        stop_writer(main)
        exit_record = json.loads(MAIN.read_text())
        combined = concatenate(first, second)
        tmp = path.with_suffix('.pt.tmp')
        torch.save(combined, tmp)
        tmp.replace(path)
        log_stats = [json.loads(line.split('MAP_RESET_STATS ', 1)[1])
            for line in Path(main['log']).read_text(errors='replace').splitlines() if 'MAP_RESET_STATS ' in line]
        stats = {k: log_stats[-1][k]+extra['map_statistics'][k] for k in log_stats[-1]}
        stats['worker_accepted_reported'] = stats['accepted']
        stats['accepted'] = count(combined)
        main.update(status='complete', path=str(path), saved_count=count(combined),
            cpu_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), map_statistics=stats,
            completion_method='concatenated_accepted_batches', shards=parts,
            main_recorder_exit_record=exit_record, returncode=0,
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        save(MAIN, main)
        save(OUT, dict(status='complete', count=count(combined), parts=[p['count'] for p in parts],
            method='Concatenated all captured accepted rows; no additional filtering or resampling.',
            path=str(path), sha256=main['cpu_sha256']))
        print(json.dumps(json.loads(OUT.read_text())), flush=True)
        return


if __name__ == '__main__':
    main()
