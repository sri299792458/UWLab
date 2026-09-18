"""Check diagnostic provenance and independent kinematics, without running Isaac."""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

torch.set_num_threads(1)
ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
OUT = ROOT / '24_gain_failure_diagnosis'
REPO = Path('/data/kanth042/repos/UWLab-reset-from-defaults')
SCRIPTS = Path(__file__).parent
SOURCE = REPO / 'source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/kinematics.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(path, download_dir=None):
    assert Path(path).is_file(), path
    return path


plan = json.loads((ROOT / '22_explicit_gain_sweep/plan_expanded.json').read_text())
for path, expected in plan['input_hashes'].items():
    assert digest(path) == expected, path
comparison = json.loads((OUT / 'comparison.json').read_text())
for replay in comparison['original_replay_validation']:
    assert replay['all_initial_states_properties_gains_and_source_hashes_equal']
    assert replay['original_passed'] == replay['replay_passed']
    for key in ['arm_position_max_abs_trajectory_delta_rad',
                'arm_velocity_max_abs_trajectory_delta_rad_s', 'target_error_max_abs_delta']:
        assert replay[key] == 0, (replay['replay'], key)

# Load exactly the same pure numerical definitions, without starting Kit.
tree = ast.parse(SOURCE.read_text())
tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
scope = dict(torch=torch, yaml=yaml, functools=functools, os=os,
             tempfile=tempfile, retrieve_file_path=local_path)
exec(compile(tree, str(SOURCE), 'exec'), scope)

audit = json.loads((ROOT / '15_table_reachability/model/export_audit.json').read_text())
arm_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
fk_scope = dict(np=np, Rotation=Rotation, audit=audit, arm_names=arm_names,
                arm_tree=[j for j in audit['articulation_tree'] if j['name'] in arm_names])
reference_source = SCRIPTS / 'check_reference_paths.py'
tree = ast.parse(reference_source.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fk']
exec(compile(tree, str(reference_source), 'exec'), fk_scope)
fk = fk_scope['fk']
q = []
for phase in ['corrected_holdout', 'grasp']:
    snap = json.loads((ROOT / '22_explicit_gain_sweep' / phase / 'worker_0/runtime_inputs.json').read_text())
    q.extend(p['q'] for p in (snap['poses'] if phase != 'grasp' else snap['poses'][::3]))
q = np.asarray(q)
transforms, jacobian = fk(q)
analytic = scope['compute_jacobian_analytical'](
    torch.tensor(q, dtype=torch.float32), device='cpu', usd_path=snap['robot_usd']).numpy()
jacobian_delta = float(np.max(np.abs(jacobian - analytic)))
assert jacobian_delta < 1e-5, jacobian_delta

epsilon = 1e-6
finite_difference = np.zeros_like(jacobian)
for joint in range(6):
    delta = np.zeros_like(q)
    delta[:, joint] = epsilon
    plus = fk(q + delta)[0]['wrist_3_link']
    minus = fk(q - delta)[0]['wrist_3_link']
    finite_difference[:, :3, joint] = (plus[:, :3, 3] - minus[:, :3, 3]) / (2 * epsilon)
    finite_difference[:, 3:, joint] = Rotation.from_matrix(
        plus[:, :3, :3] @ minus[:, :3, :3].transpose(0, 2, 1)).as_rotvec() / (2 * epsilon)
finite_difference_delta = float(np.max(np.abs(jacobian - finite_difference)))
assert finite_difference_delta < 1e-7, finite_difference_delta

runs = []
for suffix in ['', '_without_robot_obstacles']:
    for name in ['holdout_A', 'holdout_B', 'grasp_A', 'grasp_B']:
        folder = OUT / (name + suffix) / 'worker_0'
        snap = json.loads((folder / 'runtime_inputs.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        assert digest(snap['robot_usd']) == snap['robot_usd_sha256']
        assert digest(snap['controller_file']) == snap['controller_sha256']
        assert digest(summary['plan_path']) == summary['plan_sha256']
        assert digest(folder / 'run_gain_sweep.py') == summary['script_sha256']
        assert summary['controller_formula_max_error_nm'] == 0
        assert summary['physics_hz'] == 120 and summary['decimation'] == 12
        trajectory = torch.load(folder / 'trajectories.pt', map_location='cpu', weights_only=True)
        contacts = torch.load(folder / 'contact_trajectories.pt', map_location='cpu', weights_only=True)
        assert torch.isfinite(trajectory['values']).all()
        for key, value in contacts.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), (name, key)
        record = dict(run=name + suffix, archived_source_hash_valid=True,
                      source_model_controller_and_inputs_unchanged=True, all_tensors_finite=True,
                      explicit_torque_formula_error_nm=0)
        if name.startswith('grasp'):
            mismatch = []
            for side, column in [('left', 27), ('right', 28)]:
                index = contacts['contact_body_names'].index(side + '_inner_finger')
                net = contacts['contact_net_forces_world'][:, :, index].norm(dim=-1)
                mismatch.append(float((net - trajectory['values'][:, :, column]).abs().max()))
            record['finger_net_force_vs_cube_filtered_norm_max_difference_n'] = max(mismatch)
            assert max(mismatch) < 2e-5, (name, mismatch)
        runs.append(record)

scripts = ['analyze_failures.py', 'check_initial_contacts.py', 'run_contact_replay.py',
           'analyze_contact_replays.py', 'summarize_failure_diagnosis.py',
           'check_reference_paths.py', 'validate_failure_diagnosis.py']
for script in scripts:
    compile((SCRIPTS / script).read_text(), str(SCRIPTS / script), 'exec')
result = dict(input_hashes_match=True, ordinary_replays_exactly_match_originals=True,
              independent_kinematic_configurations=len(q),
              tree_vs_active_analytical_jacobian_max_abs_error=jacobian_delta,
              tree_jacobian_vs_central_difference_max_abs_error=finite_difference_delta,
              runs=runs, syntax_checked_scripts=scripts,
              source_sha256={script: digest(SCRIPTS / script) for script in scripts})
(OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
