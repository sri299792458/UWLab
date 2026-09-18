"""Exercise all four original reset recipes with the completed new atlas."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917') + '/49_clearance_and_placement')
DATA = ROOT / 'pilot/OmniReset'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=4, default=[0, 1, 2, 6])
    args = parser.parse_args()
    assert json.loads((ROOT / 'atlas/summary.json').read_text())['complete']
    assert json.loads((ROOT / 'lookup/packing_progress.json').read_text())['complete']
    assert json.loads((ROOT / 'placement_math_validation.json').read_text())['all_passed']
    for relative in ('Grasps/InsertiveAprilCube60/grasps.pt',
                     'Resets/InsertiveAprilCube60__ReceptiveAprilCube60/partial_assemblies.pt'):
        target = DATA / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'OmniReset' / relative, target)
    records, processes = {}, {}
    env = os.environ.copy()
    env.update(CONDA_PREFIX=os.environ.get("CONDA_PREFIX", sys.prefix), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def save():
        (ROOT / 'pilot_runs.json').write_text(json.dumps(records, indent=2) + '\n')

    def launch(family, gpu):
        command = [sys.executable, 'scripts_v2/tools/map_reset/run_clearance_stage.py',
            '--family', family, '--gpu', str(gpu), '--num-envs', '512', '--count', '64',
            '--dataset-dir', str(DATA), '--input-dir', str(DATA),
            '--output-subdir', 'pilot_stage/' + family]
        log = ROOT / ('pilot_' + family + '_supervisor.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        processes[family] = proc
        records[family] = dict(command=command, physical_gpu=gpu, pid=proc.pid,
                               process_group=proc.pid, status='running')
        save()

    launch('ObjectAnywhereEEAnywhere', args.gpus[0])
    launch('ObjectAnywhereEEGrasped', args.gpus[1])
    launch('ObjectPartiallyAssembledEEGrasped', args.gpus[2])
    try:
        while True:
            for family, proc in list(processes.items()):
                code = proc.poll()
                if code is not None and records[family]['status'] == 'running':
                    records[family].update(status='complete' if code == 0 else 'failed', returncode=code)
                    save()
                    if code:
                        raise RuntimeError(f'{family} failed with {code}')
                    print(f'{family} complete', flush=True)
            if records['ObjectAnywhereEEAnywhere']['status'] == 'complete' and 'ObjectRestingEEGrasped' not in records:
                launch('ObjectRestingEEGrasped', args.gpus[3])
            if len(records) == 4 and all(r['status'] == 'complete' for r in records.values()):
                break
            time.sleep(3)
    except BaseException:
        for proc in processes.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        raise
    print('ALL_FOUR_NEW_MAP_PILOTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
