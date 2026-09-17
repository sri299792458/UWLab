"""Compare runtime placement math to independent SciPy/corner calculations.

Use the previous complete atlas as a fixture while the new atlas is rebuilding.
This validates the algorithms, not the new map's final coverage or reset bank.
"""
import os
import ast
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911'))
OUT = ROOT / '49_clearance_and_placement'
spec = importlib.util.spec_from_file_location('placement_isaac_math',
    str(Path(os.environ.get("ISAACLAB_PATH", "/data/kanth042/repos/IsaacLab")) / "source/isaaclab/isaaclab/utils/math.py"))
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)
source = REPO / 'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_map_reset.py'
tree = ast.parse(source.read_text())
selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ('AtlasSeeds', '_map_acceptance')]
namespace = dict(json=json, np=np, torch=torch, yaml=yaml, Path=Path, math_utils=math_utils,
                 ATLAS=ROOT / '21_sphere_reachability', LOOKUP=ROOT / '26_map_reset_regeneration')
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
atlas = namespace['AtlasSeeds'](SimpleNamespace(device='cuda'))

rng = np.random.default_rng(238)
bounds = atlas.cfg['table_bounds_world_m']
lo, hi = np.array(bounds['min'][:2]), np.array(bounds['max'][:2])
rot = Rotation.random(512, random_state=rng)
p = np.column_stack([rng.uniform(lo[0]-.1, hi[0]+.1, 512),
                     rng.uniform(lo[1]-.1, hi[1]+.1, 512), np.full(512, 1.)])
corners = np.array(list(itertools.product([-.03, .03], repeat=3)))
world_corners = np.einsum('bij,vj->bvi', rot.as_matrix(), corners) + p[:, None]
expected = ((world_corners[:, :, :2].min(axis=1) >= lo+.05) &
            (world_corners[:, :, :2].max(axis=1) <= hi-.05)).all(axis=1)
def t(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device='cuda')
actual = atlas.cube_edge_clear(t(p), t(rot.as_quat()[:, [3, 0, 1, 2]])).cpu().numpy()
assert np.array_equal(actual, expected)

# Explicit near-boundary cases distinguish center clearance from surface clearance.
fixture_p, fixture_q, fixture_expected = [], [], []
for yaw in (0., 45., 90.):
    r = Rotation.from_euler('z', yaw, degrees=True)
    half = abs(r.apply(corners)[:, 0]).max()
    for extra in (-.0001, .0001):
        fixture_p.append([lo[0] + .05 + half + extra, -.1, 1.])
        fixture_q.append(r.as_quat()[[3, 0, 1, 2]])
        fixture_expected.append(extra > 0)
assert atlas.cube_edge_clear(t(fixture_p), t(fixture_q)).tolist() == fixture_expected

# Position coverage must match the union over all saved map orientations.
with np.load(ROOT / '21_sphere_reachability/atlas.npz') as saved:
    occupied = (saved['status'] == 4).any(axis=1)
axes = [a.cpu().numpy() for a in atlas.axes]
positions = np.column_stack([rng.uniform(lo[0]-.1, hi[0]+.1, 512),
    rng.uniform(lo[1]-.1, hi[1]+.1, 512), atlas.cfg['tabletop_z_m'] + rng.uniform(-.1, .7, 512)])
coordinates = [positions[:, 2] - atlas.cfg['tabletop_z_m'], positions[:, 1], positions[:, 0]]
inside = np.ones(512, dtype=bool)
idx = []
for axis, value in zip(axes, coordinates):
    inside &= (value >= axis[0]) & (value <= axis[-1])
    idx.append(abs(value[:, None] - axis).argmin(axis=1))
expected_coverage = inside & occupied[idx[0], idx[1], idx[2]]
actual_coverage = atlas.covers_positions(t(positions)).cpu().numpy()
assert np.array_equal(actual_coverage, expected_coverage)

# Exactly the stock angle draws must survive position selection. Some sampled
# pose orientations are deliberately uncovered, proving the old veto is gone.
ranges = t([[0., 0.], [np.pi/4, 3*np.pi/4], [np.pi/2, 3*np.pi/2]])
torch.manual_seed(81)
expected_angles = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (512, 3), device='cuda')
expected_q = math_utils.quat_from_euler_xyz(expected_angles[:, 0], expected_angles[:, 1], expected_angles[:, 2])
torch.manual_seed(81)
hand_p, hand_q = atlas.sample_free_hand_poses(512, ranges)
assert torch.equal(hand_q, expected_q)
tool_p, tool_q = math_utils.combine_frame_transforms(hand_p, hand_q,
    atlas.tool_p.expand(512, -1), atlas.tool_q.expand(512, -1))
assert atlas.covers_positions(tool_p).all()
previously_vetoed = int((~atlas.covers_tool_poses(tool_p, tool_q)).sum())
assert previously_vetoed > 0

# Remove the initial-pose gate, but retain rejection of final colliding geometry.
robot = SimpleNamespace(body_names=['body'], data=SimpleNamespace(
    body_link_pos_w=t(np.zeros((2, 1, 3))), body_link_quat_w=t([[[1, 0, 0, 0]], [[1, 0, 0, 0]]])))
cube = SimpleNamespace(data=SimpleNamespace(root_pos_w=t([[0, -.1, 1], [0, -.1, 1]]),
                                            root_quat_w=t([[1, 0, 0, 0], [1, 0, 0, 0]])))
scene = type('Scene', (dict,), {})({'robot': robot, 'insertive_object': cube, 'receptive_object': cube})
scene.env_origins = t(np.zeros((2, 3)))
env = SimpleNamespace(scene=scene, device='cuda', cfg=SimpleNamespace(events=SimpleNamespace()),
    _umi_map_ik_valid=torch.zeros(2, dtype=torch.bool, device='cuda'), _umi_atlas_seeds=atlas,
    _umi_map_stats={k: 0 for k in ('upstream_accepted', 'geometry_checked', 'self_rejected',
                                 'world_rejected', 'placement_rejected', 'accepted')},
    _umi_sphere_geometry=SimpleNamespace(evaluate=lambda *args: (
        torch.tensor([True, True], device='cuda'), torch.tensor([True, False], device='cuda'))))
result = namespace['_map_acceptance'](env, torch.ones(2, dtype=torch.bool, device='cuda'))
assert result.tolist() == [True, False]
report = dict(random_rotated_footprint_cases=512, surface_boundary_cases=6,
              position_coverage_cases=512, original_hand_angle_samples_preserved=512,
              samples_no_longer_vetoed_by_missing_orientation=previously_vetoed,
              pose_error_gate_removed=True, final_collision_gate_retained=True,
              all_passed=True, fixture_atlas=str(ROOT / '21_sphere_reachability'),
              scope='Position-only algorithm validation; not a new-map coverage result.')
(OUT / 'placement_math_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
