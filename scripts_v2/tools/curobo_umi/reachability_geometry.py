"""Source-convex collision checks for the frozen open-hand reachability model.

cuRobo supplies IK candidates. FCL checks the original exported convex hulls so
sphere protrusion does not silently decide the final geometric classification.
This module neither operates hardware nor changes the active Isaac asset.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
sys.path.insert(0, str(ROOT / '15_table_reachability/deps'))
import fcl

ARM_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def pose_matrix(position: list, quaternion_wxyz: list) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
    value[:3, 3] = position
    return value


def batch_fk(audit: dict, q: np.ndarray) -> dict[str, np.ndarray]:
    """Compute source-joint FK for [B,6] arm angles and all-zero finger angles."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f'Expected arm angles [B,6], got {q.shape}')
    count = len(q)
    transforms = {'base_link': np.broadcast_to(np.eye(4), (count, 4, 4)).copy()}
    for joint in audit['articulation_tree']:
        left, right, sign = np.asarray(joint['T0']), np.asarray(joint['T1']), 1
        if joint['reverse']:
            left, right, sign = right, left, -1
        constant = left @ np.linalg.inv(right)
        if joint['fixed'] or joint['name'] not in ARM_NAMES:
            transforms[joint['child']] = transforms[joint['parent']] @ constant
        else:
            axis = np.eye(3)['XYZ'.index(joint['axis'])]
            vectors = sign * q[:, ARM_NAMES.index(joint['name']), None] * axis
            turn = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            turn[:, :3, :3] = Rotation.from_rotvec(vectors).as_matrix()
            transforms[joint['child']] = transforms[joint['parent']] @ left @ turn @ np.linalg.inv(right)
    transforms[audit['tool_frame']] = transforms['robotiq_base_link'] @ pose_matrix(
        audit['tool_offset']['pos'], audit['tool_offset']['quat'])
    return transforms


def convex_geometry(path: str) -> tuple:
    mesh = trimesh.load(path, force='mesh', process=False)
    faces = np.c_[np.full(len(mesh.faces), 3), mesh.faces].ravel().astype(np.int32)
    geometry = fcl.Convex(np.asarray(mesh.vertices), len(mesh.faces), faces)
    return geometry, np.asarray(mesh.bounds)


def set_transform(obj: object, transform: np.ndarray) -> None:
    obj.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))


def exact_pair_gap(a: object, b: object, signed: bool = False) -> float:
    """Return FCL hull distance; intersecting geometry returns -1 if unsigned."""
    if not signed:
        if fcl.collide(a, b, fcl.CollisionRequest(), fcl.CollisionResult()):
            return -1.0
    return float(fcl.distance(a, b, fcl.DistanceRequest(
        enable_nearest_points=False, enable_signed_distance=signed), fcl.DistanceResult()))


class SourceCollisionChecker:
    """Check exported hulls against each other and the 33 lab collision boxes.

    Exclusions: same body, direct mechanical adjacency, and fixed robot base
    versus its mounting plate lab_18. Frozen open-hand internal pairs are
    independently measured at construction; only clear invariant pairs can
    subsequently be skipped. No moving arm/camera pair is pruned by sampling.
    """

    def __init__(self, model_dir: Path, clearance: float = 0.001):
        self.model_dir = Path(model_dir)
        self.audit = json.loads((self.model_dir / 'export_audit.json').read_text())
        self.clearance = clearance
        self.shapes = self.audit['collision_shapes']
        self.objects, self.bounds = [], []
        for shape in self.shapes:
            geometry, bounds = convex_geometry(shape['mesh'])
            self.objects.append(fcl.CollisionObject(geometry))
            self.bounds.append(bounds)
        self.bounds = np.asarray(self.bounds)
        self.local_centers = self.bounds.mean(axis=1)
        self.local_extents = (self.bounds[:, 1] - self.bounds[:, 0]) / 2
        scene = yaml.safe_load((self.model_dir / 'lab_scene.yml').read_text())
        self.world_names, self.world_objects, self.world_centers, self.world_extents = [], [], [], []
        for name, cuboid in scene['cuboid'].items():
            transform = pose_matrix(cuboid['pose'][:3], cuboid['pose'][3:])
            self.world_names.append(name)
            self.world_objects.append(fcl.CollisionObject(fcl.Box(*cuboid['dims']),
                fcl.Transform(transform[:3, :3], transform[:3, 3])))
            self.world_centers.append(transform[:3, 3])
            self.world_extents.append(np.abs(transform[:3, :3]) @ (np.asarray(cuboid['dims']) / 2))
        self.world_centers, self.world_extents = np.asarray(self.world_centers), np.asarray(self.world_extents)
        adjacency = {frozenset((j['body0'], j['body1'])) for j in self.audit['source_joints']}
        self.pairs, self.exclusions = [], []
        self.fixed_hand = {'wrist_3_link', 'robotiq_base_link'} | {
            s['body'] for s in self.shapes if s['body'].startswith(('left_', 'right_'))}
        home = np.array([[0., -1.57, 1.57, -1.57, -1.57, 0.]])
        poses = batch_fk(self.audit, home)
        for i, shape in enumerate(self.shapes):
            set_transform(self.objects[i], poses[shape['body']][0])
        self.invariant_pair_audit = []
        for i, j in itertools.combinations(range(len(self.shapes)), 2):
            first, second = self.shapes[i], self.shapes[j]
            reason = None
            if first['body'] == second['body']:
                reason = 'same rigid body'
            elif frozenset((first['body'], second['body'])) in adjacency:
                reason = 'direct mechanical joint adjacency'
            if reason:
                self.exclusions.append(dict(a=first['name'], b=second['name'], reason=reason))
                continue
            if first['body'] in self.fixed_hand and second['body'] in self.fixed_hand:
                distance = exact_pair_gap(self.objects[i], self.objects[j], signed=True)
                self.invariant_pair_audit.append(dict(a=first['name'], b=second['name'], gap_m=distance))
                # Clear pairs stay clear for all arm configurations when the
                # complete gripper is locked to this exact opening.
                if distance >= clearance:
                    self.exclusions.append(dict(a=first['name'], b=second['name'],
                        reason='frozen open-hand pair independently checked clear', gap_m=distance))
                    continue
            self.pairs.append((i, j))
        self.pairs = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
        self.world_mask = np.ones((len(self.shapes), len(self.world_names)), dtype=bool)
        self.world_mask[[i for i, s in enumerate(self.shapes) if s['body'] == 'base_link'],
                        self.world_names.index('lab_18')] = False
        self.invariant_world_audit = []
        for i, shape in enumerate(self.shapes):
            if shape['body'] != 'base_link':
                continue
            for j, name in enumerate(self.world_names):
                if not self.world_mask[i, j]:
                    continue
                distance = exact_pair_gap(self.objects[i], self.world_objects[j], signed=True)
                self.invariant_world_audit.append(dict(a=shape['name'], b=name, gap_m=distance))
                if distance < clearance:
                    raise ValueError(f'Fixed base intersects unexcluded world component {name}: {distance}')
                self.world_mask[i, j] = False
        self.joint_limits = np.array([[next(j for j in self.audit['articulation_tree']
            if j['name'] == name)[side] for name in ARM_NAMES] for side in ['lower', 'upper']])
        self.exact_calls = 0
        inverse_tool = np.linalg.inv(poses[self.audit['tool_frame']][0])
        self.hand_indices = [i for i,s in enumerate(self.shapes) if s['body'] in self.fixed_hand]
        self.tool_to_hand = np.stack([inverse_tool @ poses[self.shapes[i]['body']][0] for i in self.hand_indices])

    def check_target_hand(self, tool_transforms: np.ndarray) -> np.ndarray:
        """Check the frozen hand at exact target poses against lab geometry.

        All IK branches share these hand transforms. Keeping this independent
        target check also avoids accepting a slightly displaced IK solution
        for a nominal target whose hand itself intersects the scene.
        """
        tool_transforms = np.asarray(tool_transforms)
        if tool_transforms.ndim != 3 or tool_transforms.shape[1:] != (4,4):
            raise ValueError(f'Expected transforms [B,4,4], got {tool_transforms.shape}')
        transforms = tool_transforms[:,None] @ self.tool_to_hand[None]
        ids = self.hand_indices
        centers = np.einsum('bnij,nj->bni', transforms[:,:,:3,:3], self.local_centers[ids]) + transforms[:,:,:3,3]
        extents = np.einsum('bnij,nj->bni', np.abs(transforms[:,:,:3,:3]), self.local_extents[ids])
        near = (np.abs(centers[:,:,None] - self.world_centers[None,None]) <=
            extents[:,:,None] + self.world_extents[None,None] + self.clearance).all(axis=-1)
        clear = np.ones(len(tool_transforms),dtype=bool)
        for row in range(len(tool_transforms)):
            for i,j in np.argwhere(near[row]):
                original = ids[i]
                set_transform(self.objects[original],transforms[row,i])
                self.exact_calls += 1
                if exact_pair_gap(self.objects[original],self.world_objects[j]) < self.clearance:
                    clear[row] = False
                    break
        return clear

    def evaluate(self, q: np.ndarray, details: bool = False) -> dict:
        """Check [B,6] candidates; return clear flag and first failing pair."""
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 6:
            raise ValueError(f'Expected arm angles [B,6], got {q.shape}')
        count = len(q)
        finite = np.isfinite(q).all(axis=1)
        within = finite & (q >= self.joint_limits[0] - 1e-6).all(axis=1) & (q <= self.joint_limits[1] + 1e-6).all(axis=1)
        safe_q = np.where(finite[:, None], q, 0)
        all_poses = batch_fk(self.audit, safe_q)
        transforms = np.stack([all_poses[s['body']] for s in self.shapes], axis=1)
        centers = np.einsum('bnij,nj->bni', transforms[:, :, :3, :3], self.local_centers) + transforms[:, :, :3, 3]
        extents = np.einsum('bnij,nj->bni', np.abs(transforms[:, :, :3, :3]), self.local_extents)
        world_near = (np.abs(centers[:, :, None] - self.world_centers[None, None]) <=
            extents[:, :, None] + self.world_extents[None, None] + self.clearance).all(axis=-1)
        world_near &= self.world_mask
        a, b = self.pairs.T
        self_near = (np.abs(centers[:, a] - centers[:, b]) <= extents[:, a] + extents[:, b] + self.clearance).all(axis=-1)
        clear = within.copy()
        reason = np.full(count, 'clear', dtype=object)
        reason[~within] = 'joint_limit_or_nonfinite'
        failure = []
        for row in range(count):
            if not within[row]:
                failure.append(None)
                continue
            near_world = np.argwhere(world_near[row])
            near_self = np.flatnonzero(self_near[row])
            needed = set(near_world[:, 0].tolist()) | set(self.pairs[near_self].ravel().tolist())
            for i in needed:
                set_transform(self.objects[i], transforms[row, i])
            failed = None
            for k in near_self:
                i, j = self.pairs[k]
                distance = exact_pair_gap(self.objects[i], self.objects[j])
                self.exact_calls += 1
                if distance < self.clearance:
                    clear[row], reason[row] = False, 'self_collision_or_clearance'
                    failed = dict(kind='self', a=self.shapes[i]['name'], b=self.shapes[j]['name'], gap_m=distance)
                    break
            if clear[row]:
                for i, j in near_world:
                    distance = exact_pair_gap(self.objects[i], self.world_objects[j])
                    self.exact_calls += 1
                    if distance < self.clearance:
                        clear[row], reason[row] = False, 'world_collision_or_clearance'
                        failed = dict(kind='world', a=self.shapes[i]['name'], b=self.world_names[j], gap_m=distance)
                        break
            failure.append(failed)
        result = dict(clear=clear, reason=reason)
        if details:
            result.update(first_failure=failure, tool_transform=all_poses[self.audit['tool_frame']])
        return result


if __name__ == '__main__':
    checker = SourceCollisionChecker(ROOT / '15_table_reachability/model')
    print(json.dumps(dict(invariant_pairs=checker.invariant_pair_audit,
        excluded_pairs=checker.exclusions, checked_pair_count=len(checker.pairs)), indent=2), flush=True)
    rng = np.random.default_rng(1709)
    q = rng.uniform(checker.joint_limits[0], checker.joint_limits[1], size=(1024, 6))
    before = time.perf_counter()
    result = checker.evaluate(q, details=True)
    report = dict(seconds=time.perf_counter()-before, count=len(q), clear=int(result['clear'].sum()),
        failure_counts={key:int((result['reason']==key).sum()) for key in set(result['reason'])},
        exact_pair_queries=checker.exact_calls, first_failures=result['first_failure'][:12],
        invariant_pairs=checker.invariant_pair_audit)
    (ROOT/'20_dense_reachability/source_collision_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)
