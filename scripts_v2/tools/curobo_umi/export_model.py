"""Export the selected Isaac USD articulation and lab boxes for offline cuRobo use.

Joint frames and convex collision geometry come from the selected asset, including
instance proxies. No hardware interfaces or controller code are loaded. The URDF
retains all independent articulation coordinates; closed-chain constraint joints
excluded by PhysX are recorded, and the fingers can be locked for a map query.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
import yaml

ARM_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def matrix(pos, quat):
    q = np.asarray([quat.GetReal(), *quat.GetImaginary()], dtype=float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = np.asarray(pos)
    return out


def points_relative(prim, body, cache):
    if prim.GetTypeName() == 'Xform':
        return np.concatenate([points_relative(p, body, cache)
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            if p.GetTypeName() == 'Mesh'])
    if prim.GetTypeName() == 'Mesh':
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    elif prim.GetTypeName() == 'Cube':
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        points = np.asarray([[x, y, z] for x in [-.5, .5]
                             for y in [-.5, .5] for z in [-.5, .5]]) * size
    else:
        raise ValueError(f'Unsupported collision primitive: {prim.GetPath()} {prim.GetTypeName()}')
    transform, _ = cache.ComputeRelativeTransform(prim, body)
    return (np.c_[points, np.ones(len(points))] @ np.asarray(transform))[:, :3]


def extract_joints(stage):
    records = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        if not joint.GetJointEnabledAttr().Get():
            continue
        lower = prim.GetAttribute('physics:lowerLimit').Get()
        upper = prim.GetAttribute('physics:upperLimit').Get()
        records.append(dict(name=prim.GetName(), body0=b0[0].name, body1=b1[0].name,
            fixed=prim.IsA(UsdPhysics.FixedJoint),
            excluded=joint.GetExcludeFromArticulationAttr().Get(),
            axis=prim.GetAttribute('physics:axis').Get() or 'Z',
            T0=matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()).tolist(),
            T1=matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()).tolist(),
            lower=None if lower is None or not np.isfinite(lower) else np.deg2rad(lower),
            upper=None if upper is None or not np.isfinite(upper) else np.deg2rad(upper)))
    graph = {}
    for j in records:
        if not j['excluded']:
            for a, b, reverse in [(j['body0'], j['body1'], False), (j['body1'], j['body0'], True)]:
                graph.setdefault(a, []).append((b, j, reverse))
    seen = {'base_link'}
    queue = deque(['base_link'])
    tree = []
    while queue:
        parent = queue.popleft()
        for child, j, reverse in graph.get(parent, []):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            tree.append(dict(**j, parent=parent, child=child, reverse=reverse))
    return records, tree, seen


def fk(tree, values):
    poses = {'base_link': np.eye(4)}
    for j in tree:
        a, b = np.asarray(j['T0']), np.asarray(j['T1'])
        sign = 1
        if j['reverse']:
            a, b, sign = b, a, -1
        turn = np.eye(4)
        if not j['fixed']:
            axis = np.eye(3)['XYZ'.index(j['axis'])]
            turn[:3, :3] = Rotation.from_rotvec(axis * sign * values.get(j['name'], 0)).as_matrix()
        poses[j['child']] = poses[j['parent']] @ a @ turn @ np.linalg.inv(b)
    return poses


def origin_xml(parent, transform):
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')
    ET.SubElement(parent, 'origin', xyz=' '.join(map(str, transform[:3, 3])),
                  rpy=' '.join(map(str, rpy)))


def fixed_xml(root, name, parent, child, transform):
    joint = ET.SubElement(root, 'joint', name=name, type='fixed')
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=child)
    origin_xml(joint, transform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-usd', required=True, type=Path)
    parser.add_argument('--lab-config', required=True, type=Path)
    parser.add_argument('--geometry-audit', required=True, type=Path)
    parser.add_argument('--pose-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    (out / 'meshes').mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(args.robot_usd))
    cache = UsdGeom.XformCache()
    root = stage.GetDefaultPrim()
    records, tree, bodies = extract_joints(stage)
    meta_path = args.robot_usd.parent / 'metadata.yaml'
    metadata = yaml.safe_load(meta_path.read_text())
    constants = {}
    for node in ast.parse(args.lab_config.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    xml = ET.Element('robot', name='thunder_ur5e_umi_from_isaac_usd')
    for name in sorted(bodies):
        ET.SubElement(xml, 'link', name=name)
    for j in tree:
        a, b = np.asarray(j['T0']), np.asarray(j['T1'])
        sign = 1
        if j['reverse']:
            a, b, sign = b, a, -1
        if j['fixed']:
            fixed_xml(xml, j['name'], j['parent'], j['child'], a @ np.linalg.inv(b))
            continue
        virtual = j['name'] + '_output_frame'
        ET.SubElement(xml, 'link', name=virtual)
        joint = ET.SubElement(xml, 'joint', name=j['name'], type='revolute')
        ET.SubElement(joint, 'parent', link=j['parent'])
        ET.SubElement(joint, 'child', link=virtual)
        origin_xml(joint, a)
        axis = np.eye(3)['XYZ'.index(j['axis'])] * sign
        ET.SubElement(joint, 'axis', xyz=' '.join(map(str, axis)))
        ET.SubElement(joint, 'limit', lower=str(j['lower'] if j['lower'] is not None else -2*np.pi),
                      upper=str(j['upper'] if j['upper'] is not None else 2*np.pi), effort='150', velocity='3.14')
        fixed_xml(xml, j['name'] + '_to_body', virtual, j['child'], np.linalg.inv(b))

    collision_records = []
    for body_name in sorted(bodies):
        body = stage.GetPrimAtPath(str(root.GetPath()) + '/' + body_name)
        shapes = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        for index, prim in enumerate(shapes):
            points = points_relative(prim, body, cache)
            hull = trimesh.convex.convex_hull(points)
            name = 'camera_mount_and_body' if prim.GetName() == 'D405_to_Robotiq_Mount' else f'{body_name}_shape_{index}'
            mesh_path = out / 'meshes' / f'{name}.obj'
            hull.export(str(mesh_path))
            np.save(out / 'meshes' / f'{name}_source_points.npy', points)
            link = ET.SubElement(xml, 'link', name=name)
            for kind in ['visual', 'collision']:
                part = ET.SubElement(link, kind)
                geom = ET.SubElement(part, 'geometry')
                ET.SubElement(geom, 'mesh', filename=str(mesh_path))
            fixed_xml(xml, name + '_fixed', body_name, name, np.eye(4))
            collision_records.append(dict(name=name, body=body_name, prim_path=str(prim.GetPath()),
                mesh=str(mesh_path), source_points=len(points), hull_vertices=len(hull.vertices),
                hull_faces=len(hull.faces), bounds=hull.bounds.tolist(),
                approximation='convexHull matching authored collision approximation'))

    grasp_frame = 'umi_grasp_center'
    ET.SubElement(xml, 'link', name=grasp_frame)
    tool = np.eye(4)
    offset = metadata['gripper_offset']
    tool[:3, 3] = offset['pos']
    q = np.asarray(offset['quat'])
    tool[:3, :3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    fixed_xml(xml, 'umi_grasp_center_fixed', 'robotiq_base_link', grasp_frame, tool)
    ET.indent(xml, space='  ')
    urdf = out / 'thunder_umi.urdf'
    ET.ElementTree(xml).write(str(urdf), encoding='utf-8', xml_declaration=True)

    manifest = json.loads(args.pose_manifest.read_text())
    validation = []
    for sample in manifest['samples']:
        values = dict(zip(manifest['joint_names'], sample['joint_positions']))
        computed = fk(tree, values)
        recorded = dict(zip(manifest['body_names'], sample['body_poses']))
        base_pose = np.asarray(recorded['base_link'])
        base = np.eye(4)
        base[:3, :3] = Rotation.from_quat(base_pose[[4,5,6,3]]).as_matrix()
        base[:3, 3] = base_pose[:3]
        for name, local in computed.items():
            target = np.asarray(recorded[name])
            actual = base @ local
            r_target = Rotation.from_quat(target[[4,5,6,3]])
            validation.append(dict(family=sample['family'], row=sample['row'], body=name,
                position_error_m=float(np.linalg.norm(actual[:3,3]-target[:3])),
                rotation_error_rad=float((Rotation.from_matrix(actual[:3,:3]).inv()*r_target).magnitude())))

    camera = stage.GetPrimAtPath(str(root.GetPath())+'/robotiq_base_link/visuals/D405_camera_envelope')
    body = stage.GetPrimAtPath(str(root.GetPath())+'/robotiq_base_link')
    camera_points = points_relative(camera, body, cache)
    mount = next(c for c in collision_records if c['name']=='camera_mount_and_body')
    hull = ConvexHull(np.load(out/'meshes'/'camera_mount_and_body_source_points.npy'))
    signed = camera_points @ hull.equations[:,:3].T + hull.equations[:,3]

    scene_audit = json.loads(args.geometry_audit.read_text())
    scene = {'cuboid': {}}
    world_T_base = np.eye(4)
    q = np.asarray(constants['ROBOT_ROT'])
    world_T_base[:3,:3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    world_T_base[:3,3] = constants['ROBOT_POS']
    base_T_world = np.linalg.inv(world_T_base)
    for component in scene_audit['components']:
        bounds = component['collision_world_bounds']
        world = np.eye(4)
        world[:3,3] = (np.asarray(bounds['min']) + bounds['max'])/2
        local = base_T_world @ world
        quat = Rotation.from_matrix(local[:3,:3]).as_quat()
        scene['cuboid'][f'lab_{component["index"]}'] = {'dims':bounds['size'],
            'pose':[*local[:3,3].tolist(),*quat[[3,0,1,2]].tolist()]}
    (out/'lab_scene.yml').write_text(yaml.safe_dump(scene, sort_keys=False))
    report = dict(robot_usd=str(args.robot_usd), robot_sha256=hashlib.sha256(args.robot_usd.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(), lab_constants=constants,
        source_joints=records, articulation_tree=tree, collision_shapes=collision_records,
        urdf=str(urdf), tool_frame=grasp_frame, tool_parent='robotiq_base_link', tool_offset=offset,
        tool_offset_provenance=metadata.get('finger_model',{}).get('grasp_center_provenance'),
        camera_envelope=dict(vertices=camera_points.tolist(), contained_by_mount_hull=bool((signed<=1e-7).all()),
            minimum_containment_margin_m=float(-signed.max())),
        fk_validation=validation,
        max_fk_position_error_m=max(v['position_error_m'] for v in validation),
        max_fk_rotation_error_rad=max(v['rotation_error_rad'] for v in validation))
    (out/'export_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Exported',len(bodies),'bodies,',len(collision_records),'collision hulls,',len(scene['cuboid']),'lab boxes',flush=True)
    print('USD FK vs recorded Isaac body poses:',report['max_fk_position_error_m'],'m,',report['max_fk_rotation_error_rad'],'rad',flush=True)
    print('Camera hull containment:', report['camera_envelope']['contained_by_mount_hull'],
          report['camera_envelope']['minimum_containment_margin_m'],'m',flush=True)
    if report['max_fk_position_error_m'] > .002 or report['max_fk_rotation_error_rad'] > .02:
        raise RuntimeError('USD joint-tree export disagrees with Isaac witnesses; inspect before using the model')


if __name__ == '__main__':
    main()
