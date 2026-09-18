"""The accepted atlas spheres evaluated at supplied rigid-body transforms.

No cuRobo runtime dependency is needed inside Isaac Sim. Sphere radii, link-pair
exclusions, and all 33 oriented lab boxes come directly from the accepted model.
Native body poses allow the same spheres to follow moving fingers and base jitter.
"""
import os
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
ATLAS = ROOT / '49_clearance_and_placement/atlas'


class UmiSphereGeometry:
    def __init__(self, body_names, device):
        self.device = device
        cfg = json.loads((ATLAS / 'config.json').read_text())
        model = yaml.safe_load(Path(cfg['robot_model']).read_text())['robot_cfg']['kinematics']
        audit = json.loads((ROOT / '15_table_reachability/model/export_audit.json').read_text())
        shape_body = {s['name']: s['body'] for s in audit['collision_shapes']}
        centers, radii, bodies, links = [], [], [], []
        for link in model['collision_link_names']:
            for sphere in model['collision_spheres'][link]:
                centers.append(sphere['center'])
                radii.append(sphere['radius'] + cfg['radius_padding_m'])
                bodies.append(body_names.index(shape_body[link]))
                links.append(link)
        self.centers = torch.tensor(centers, device=device, dtype=torch.float32)
        self.radii = torch.tensor(radii, device=device, dtype=torch.float32)
        self.body_ids = torch.tensor(bodies, device=device)
        ignore = {frozenset((a, b)) for a, bs in model['self_collision_ignore'].items() for b in bs}
        pairs = [(i, j) for i in range(len(links)) for j in range(i+1, len(links))
                 if links[i] != links[j] and frozenset((links[i], links[j])) not in ignore]
        self.pair_a, self.pair_b = torch.tensor(pairs, device=device).T
        self.pair_radius = self.radii[self.pair_a] + self.radii[self.pair_b]
        self.world_ids = torch.tensor([i for i, l in enumerate(links)
                                      if l != cfg['world_ignored_robot_link']], device=device)
        scene = yaml.safe_load(Path(cfg['scene_model']).read_text())['cuboid']
        base_r = Rotation.from_quat(np.array(cfg['robot_base_quaternion_world_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        base_p = np.array(cfg['robot_base_position_world_m'])
        box_p, box_r, half = [], [], []
        for box in scene.values():
            box_p.append(base_r @ box['pose'][:3] + base_p)
            box_r.append(base_r @ Rotation.from_quat(np.array(box['pose'][3:])[[1, 2, 3, 0]]).as_matrix())
            half.append(np.array(box['dims']) / 2)
        self.box_p = torch.tensor(np.array(box_p), device=device, dtype=torch.float32)
        self.box_r = torch.tensor(np.array(box_r), device=device, dtype=torch.float32)
        self.box_half = torch.tensor(np.array(half), device=device, dtype=torch.float32)
        self.world_padding = torch.tensor([
            cfg['column_clearance_m'] - cfg['radius_padding_m']
            if name in cfg['column_box_names'] else 0.0 for name in scene
        ], device=device, dtype=torch.float32)

    def evaluate(self, body_positions, body_rotations, env_origins):
        """Return self/world clearance flags for actual link poses, in batches."""
        all_self, all_world = [], []
        for start in range(0, len(body_positions), 128):
            sl = slice(start, start+128)
            centers = (body_rotations[sl][:, self.body_ids] @ self.centers[None, :, :, None]).squeeze(-1)
            centers += body_positions[sl][:, self.body_ids] - env_origins[sl, None, :]
            gap = (centers[:, self.pair_a] - centers[:, self.pair_b]).norm(dim=-1) - self.pair_radius
            all_self.append((gap >= 0).all(dim=1))
            delta = centers[:, self.world_ids, None, :] - self.box_p
            local = torch.einsum('bski,kij->bskj', delta, self.box_r)
            d = local.abs() - self.box_half
            signed = d.clamp_min(0).norm(dim=-1) + d.amax(dim=-1).clamp_max(0)
            required = self.radii[self.world_ids][None, :, None] + self.world_padding
            all_world.append((signed >= required).all(dim=(1, 2)))
        return torch.cat(all_self), torch.cat(all_world)
