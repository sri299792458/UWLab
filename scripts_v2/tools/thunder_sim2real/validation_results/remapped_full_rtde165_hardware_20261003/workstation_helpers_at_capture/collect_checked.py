"""Run the updated collector after checking its reviewed inputs and live readiness."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from robot_readiness import check_gripper_open

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'UWLab/scripts_v2/tools/thunder_sim2real'
sys.path.insert(0, str(SOURCE / 'workstation'))
import collect_thunder as collector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timing-check', action='store_true', help='Use the explicit 10% amplitude timing-check candidate')
    args = parser.parse_args()
    config_path = HERE / ('collection.timing_check.json' if args.timing_check else 'collection.json')
    config = json.loads(config_path.read_text())
    candidate_name = ('collection.remapped_swe_timing_check.json' if args.timing_check
                      else 'collection.remapped_full_amplitude_swe_candidate.json')
    candidate = json.loads((SOURCE / 'workstation' / candidate_name).read_text())
    hardware_fields = {'robot_ip', 'polyscope_version', 'payload_mass_kg', 'payload_cog_m'}
    for key, expected in candidate.items():
        if key not in hardware_fields and config.get(key) != expected:
            raise RuntimeError(f'Local collection setting differs from the reviewed candidate: {key}')
    offsets = collector.make_plan(config)
    sdk_version = importlib.metadata.version('ur-rtde')
    if sdk_version != collector.REQUIRED_RTDE_VERSION:
        raise RuntimeError(f'Collector requires ur-rtde {collector.REQUIRED_RTDE_VERSION}; found {sdk_version}')
    print(f'ur-rtde {sdk_version}; robot-state-locked timing; candidate: {candidate_name}')
    print(f'{len(offsets)} samples at 500 Hz; duration {len(offsets)*collector.DT:g} seconds.')
    print(f'Excitation: {config["f0_hz"]:g}–{config["f1_hz"]:g} Hz; amplitudes (m/rad): {config["amplitudes_m_rad"]}')
    print(f'Start degrees: {np.rad2deg(config["start_joint_positions_rad"]).tolist()}')
    if not args.execute:
        print('Plan only. No robot connection.')
        return
    check_gripper_open(config['robot_ip'])
    import rtde_receive
    robot = rtde_receive.RTDEReceiveInterface(config['robot_ip'], 10.,
        ['actual_q', 'actual_qd', 'robot_mode', 'safety_mode', 'payload', 'payload_cog'])
    try:
        time.sleep(.25)
        if robot.getRobotMode() != 7 or robot.getSafetyMode() != 1:
            raise RuntimeError('Robot must be powered and in Normal safety mode')
        payload = np.array([robot.getPayload(), *robot.getPayloadCog()])
        expected = np.array([config['payload_mass_kg'], *config['payload_cog_m']])
        if not np.isfinite(payload).all() or np.max(np.abs(payload-expected)) > .001:
            raise RuntimeError('Live payload differs from the configured collection payload')
    finally:
        robot.disconnect()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = HERE / 'data' / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / 'collection.json').write_text(json.dumps(config, indent=2)+'\n')
    output = run / 'thunder-fit.pt'
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, value):
            for stream in self.streams: stream.write(value); stream.flush()
        def flush(self):
            for stream in self.streams: stream.flush()
    import contextlib
    import subprocess
    source_hashes = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (SOURCE / 'workstation').rglob('*') if p.is_file() and p.suffix in ('.py', '.json')}
    (run / 'provenance.json').write_text(json.dumps({
        'git_commit': subprocess.check_output(['git', '-C', str(HERE.parent/'UWLab'), 'rev-parse', 'HEAD'], text=True).strip(),
        'git_worktree_dirty': bool(subprocess.check_output(['git', '-C', str(HERE.parent/'UWLab'), 'status', '--porcelain'], text=True).strip()),
        'source_sha256': source_hashes, 'bundle_release': 'thunder-lab-20260917-mount-v2',
        'ur_rtde_version': sdk_version, 'candidate_file': candidate_name,
        'workstation_helpers_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (HERE/'collect_checked.py', HERE/'run.sh', HERE/'run_realtime.sh',
                                                HERE/'verify_realtime.py', HERE/'robot_readiness.py')}
        }, indent=2)+'\n')
    with (run/'collector.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f'Run directory: {run}', flush=True)
            try:
                collector.collect(config, offsets, output)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
    result = subprocess.run([sys.executable, str(SOURCE/'records.py'), 'validate', str(output)],
                            text=True, capture_output=True)
    (run/'validation.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr)
    result.check_returncode()
    print(f'Completed and validated recording: {output}')


if __name__ == '__main__':
    main()
