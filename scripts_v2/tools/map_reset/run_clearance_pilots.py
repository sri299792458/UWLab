"""Exercise all four original reset recipes with the completed new atlas."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/49_clearance_and_placement')
DATA = ROOT / 'pilot/OmniReset'


def main():
    assert json.loads((ROOT / 'atlas/summary.json').read_text())['complete']
    assert json.loads((ROOT / 'lookup/packing_progress.json').read_text())['complete']
    assert json.loads((ROOT / 'position_only_fixture/summary.json').read_text())['all_six_faces_present']
    for relative in ('Grasps/InsertiveAprilCube60/grasps.pt',
                     'Resets/InsertiveAprilCube60__ReceptiveAprilCube60/partial_assemblies.pt'):
        target = DATA / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'OmniReset' / relative, target)
    records, processes = {}, {}
    env = os.environ.copy()
    env.update(CONDA_PREFIX='/data/kanth042/envs/uwlab-isaac51', OMP_NUM_THREADS='1',
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

    launch('ObjectAnywhereEEAnywhere', 0)
    launch('ObjectAnywhereEEGrasped', 3)
    launch('ObjectPartiallyAssembledEEGrasped', 5)
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
                launch('ObjectRestingEEGrasped', 1)
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
