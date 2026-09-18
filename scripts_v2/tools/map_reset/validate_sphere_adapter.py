"""Compare the native-pose sphere adapter against cuRobo on open-hand poses."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts_v2/tools/curobo_umi'))
from reachability_geometry import batch_fk

spec = importlib.util.spec_from_file_location('umi_sphere_geometry', REPO /
    'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_sphere_geometry.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
root = geometry.ROOT
cfg = json.loads((geometry.ATLAS / 'config.json').read_text())
audit = json.loads((root / '15_table_reachability/model/export_audit.json').read_text())
body_names = ['base_link'] + [j['child'] for j in audit['articulation_tree']]
body_names = list(dict.fromkeys(body_names))
adapter = geometry.UmiSphereGeometry(body_names, 'cuda')
model = yaml.safe_load(Path(cfg['robot_model']).read_text())['robot_cfg']
checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=model,
    scene_model=cfg['scene_model'], collision_activation_distance=0., self_collision_activation_distance=0.))

rng = np.random.default_rng(42)
q = rng.uniform(-np.pi, np.pi, (1024, 6)).astype(np.float32)
with np.load(geometry.ATLAS / 'atlas.npz') as atlas:
    # Sample from the packed lookup, including several orientations and heights.
    dense = np.load(root / '26_map_reset_regeneration/atlas_joint_seeds.npy', mmap_mode='r')
    clear = np.argwhere(atlas['status'] == 4)
    chosen = clear[rng.choice(len(clear), 1024, replace=False)]
    good = dense[chosen[:, 0], chosen[:, 2], chosen[:, 3], chosen[:, 1]]
    q = np.concatenate([q, good])
poses = batch_fk(audit, q)
base_r = Rotation.from_quat(np.array(cfg['robot_base_quaternion_world_wxyz'])[[1, 2, 3, 0]]).as_matrix()
base_p = np.array(cfg['robot_base_position_world_m'])
p = np.stack([poses[n][:, :3, 3] @ base_r.T + base_p for n in body_names], axis=1)
r = np.stack([base_r @ poses[n][:, :3, :3] for n in body_names], axis=1)
actual_self, actual_world = adapter.evaluate(torch.tensor(p, device='cuda', dtype=torch.float32),
    torch.tensor(r, device='cuda', dtype=torch.float32), torch.zeros((len(q), 3), device='cuda'))
spheres = checker.get_kinematics(torch.tensor(q, device='cuda')[:, None]).robot_spheres.clone()
spheres[..., 3] += cfg['radius_padding_m']
expected_self = checker.get_self_collision_distance(spheres).reshape(len(q), -1).amax(dim=1) <= 0
base = checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name(cfg['world_ignored_robot_link'])
spheres[:, :, base, 3] = -100.
expected_world = checker.get_collision_constraint(spheres).reshape(len(q), -1).amax(dim=1) <= 0
result = dict(configurations=len(q), self_disagreements=int((actual_self != expected_self).sum()),
              world_disagreements=int((actual_world != expected_world).sum()),
              expected_self_clear=int(expected_self.sum()), expected_world_clear=int(expected_world.sum()),
              scope='Open-hand numerical equivalence to accepted cuRobo model; original pair exclusions retained.')
(root / '26_map_reset_regeneration/sphere_adapter_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
assert result['self_disagreements'] == result['world_disagreements'] == 0
