"""Replace only D415 camera assembly geometry with the verified D405 R2 mount.

The calibrated arm, wrist limits, stock fingers, and all authored dynamics are
inherited. PhysX may derive different dynamics from the replacement colliders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

SOURCE_SHA = "b00829fbce65001dc3dc88e6078fc353b3df3946b069efcb276f7d9016f7b817"
META_SHA = "625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242"
R2_STL_SHA = "9c350d21205a0faee85acca7afb5d60a310e1c03f1203e23c381fcc883bde388"
OLD_ROOT = "/ur5e_robotiq_gripper_d415_mount"
NEW_ROOT = "/ur5e_robotiq_gripper_d405_mount"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def material(stage, path, color):
    """Ported unchanged from UWLab's validated robotiq_r2_mount.py."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.7)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def define_mesh(stage, path, mesh, mat, collision=False):
    """Ported unchanged from UWLab's validated robotiq_r2_mount.py."""
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    usd = UsdGeom.Mesh.Define(stage, path)
    usd.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh.vertices])
    usd.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    usd.CreateSubdivisionSchemeAttr('none')
    usd.CreateExtentAttr([Gf.Vec3f(*v) for v in mesh.bounds])
    UsdShade.MaterialBindingAPI.Apply(usd.GetPrim()).Bind(mat)
    if collision:
        usd.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(usd.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(usd.GetPrim()).CreateApproximationAttr('convexHull')
    return usd.GetPrim()


def snapshot(stage: Usd.Stage) -> dict:
    result = {}
    root = str(stage.GetDefaultPrim().GetPath())
    for prim in stage.Traverse():
        path = str(prim.GetPath()).replace(root, "/ROBOT", 1)
        result[path] = {
            "type": str(prim.GetTypeName()),
            "schemas": list(prim.GetAppliedSchemas()),
            "attrs": {attr.GetName(): repr(attr.Get()).replace(root, "/ROBOT") for attr in prim.GetAttributes()},
            "rels": {rel.GetName(): [str(x).replace(root, "/ROBOT") for x in rel.GetTargets()] for rel in prim.GetRelationships()},
            "custom_data": prim.GetCustomData(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--r2-stl", type=Path, required=True)
    parser.add_argument("--mechanical-checks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.expanduser().resolve() / "robot.usd"
    output = args.output_dir.expanduser().resolve() / "robot.usd"
    part = args.r2_stl.expanduser().resolve()
    mechanical_checks = args.mechanical_checks.expanduser().resolve()
    if sha(source) != SOURCE_SHA or sha(source.parent / "metadata.yaml") != META_SHA:
        raise RuntimeError("Current reviewed wrist180 asset changed")
    if sha(part) != R2_STL_SHA:
        raise RuntimeError("R2 mount STL changed")
    if source == output or output.exists() or output.parent.exists() and any(output.parent.iterdir()):
        raise RuntimeError("Use a new empty output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    shutil.copy2(source.parent / "metadata.yaml", output.parent / "metadata.yaml")
    stage = Usd.Stage.Open(str(output))
    before = snapshot(stage)
    base = OLD_ROOT + "/robotiq_base_link"
    for path in (
        base + "/visuals/D415_to_Robotiq_Mount",
        base + "/collisions/D415_to_Robotiq_Mount",
        base + "/collisions/d415_and_cable",
    ):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing expected D415 prim: {path}")
        stage.RemovePrim(path)

    housing = stage.GetPrimAtPath(base + "/visuals/mesh_0")
    vendor_to_body = np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(
        housing, stage.GetPrimAtPath(base))[0]).T
    if not np.allclose(vendor_to_body[:3, :3].T @ vendor_to_body[:3, :3], np.eye(3), atol=1e-6):
        raise RuntimeError("Unexpected stock housing transform")
    mount = trimesh.load(part, force="mesh")
    if not mount.is_watertight or mount.volume <= 0:
        raise RuntimeError("R2 mount mesh is not a watertight solid")
    camera_to_vendor = np.eye(4)
    camera_to_vendor[:3, :3] = (
        Rotation.from_euler("y", 180, degrees=True).as_matrix()
        @ Rotation.from_euler("x", -60, degrees=True).as_matrix()
    )
    camera_to_vendor[:3, 3] = (0, .070, .064)
    mount.apply_scale(.001)
    mount.apply_translation((0, 0, -.028))
    mount.apply_transform(camera_to_vendor)
    checks = json.loads(mechanical_checks.read_text())
    if not np.allclose(mount.bounds, np.asarray(checks["world_bounds_mm"]) * .001, atol=2e-6):
        raise RuntimeError("R2 source mount bounds changed")
    # Reuse the R2 registration and material/mesh functions verbatim. The
    # geometry and nominal optical pose match the previously validated asset.
    mount.apply_transform(vendor_to_body)
    camera_to_body = vendor_to_body @ camera_to_vendor
    camera = trimesh.creation.box(extents=(.042, .042, .023))
    camera.apply_translation((0, 0, -.0115))
    camera.apply_transform(camera_to_body)
    mount_mat = material(stage, OLD_ROOT + "/r2_materials/Mount", (.42, .48, .57))
    camera_mat = material(stage, OLD_ROOT + "/r2_materials/Camera", (.15, .17, .2))
    define_mesh(stage, base + "/visuals/D405_to_Robotiq_Mount", mount, mount_mat)
    define_mesh(stage, base + "/collisions/D405_to_Robotiq_Mount", mount, mount_mat, True)
    camera_prim = define_mesh(stage, base + "/visuals/D405_camera_envelope", camera, camera_mat)
    camera_prim.SetCustomDataByKey("geometryProvenance", "Nominal 42 x 42 x 23 mm box; not vendor housing CAD")
    camera_pos = (camera_to_body @ np.array([-.009, 0, -.0037, 1]))[:3]
    forward, up = camera_to_body[:3, 2], camera_to_body[:3, 1]
    camera_rot = Rotation.from_matrix(np.column_stack((np.cross(forward, up), up, -forward))).as_quat()
    optical = UsdGeom.Xform.Define(stage, base + "/d405_r2_nominal_optical_frame")
    optical.AddTranslateOp().Set(Gf.Vec3d(*camera_pos))
    optical.AddOrientOp().Set(Gf.Quatf(float(camera_rot[3]), Gf.Vec3f(*camera_rot[:3])))
    optical.GetPrim().SetCustomDataByKey("poseProvenance", "R2 CAD nominal left lens; not measured hand-eye calibration")

    editor = Usd.NamespaceEditor(stage)
    if not editor.RenamePrim(stage.GetDefaultPrim(), "ur5e_robotiq_gripper_d405_mount") or not editor.CanApplyEdits() or not editor.ApplyEdits():
        raise RuntimeError("Could not rename robot root")
    stage.SetDefaultPrim(stage.GetPrimAtPath(NEW_ROOT))
    after = snapshot(stage)
    removed = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    common_changes = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    expected_removed = sorted([
        "/ROBOT/robotiq_base_link/visuals/D415_to_Robotiq_Mount",
        "/ROBOT/robotiq_base_link/collisions/D415_to_Robotiq_Mount",
        "/ROBOT/robotiq_base_link/collisions/d415_and_cable",
    ])
    if removed != expected_removed or common_changes:
        raise RuntimeError(f"Unexpected USD changes: removed={removed}, changed={common_changes}")
    stage.GetRootLayer().Save()
    if sha(output.parent / "metadata.yaml") != META_SHA:
        raise RuntimeError("Metadata changed")
    report = {
        "status": "PASS", "scope": "R2 D405 geometry and nominal optical frame only; no authored mass/inertia, joints, limits, mimics, or finger changes",
        "source": str(source), "source_sha256": SOURCE_SHA, "source_metadata_sha256": META_SHA,
        "output": str(output), "output_sha256": sha(output), "output_metadata_sha256": sha(output.parent / "metadata.yaml"),
        "r2_stl": str(part), "r2_stl_sha256": R2_STL_SHA,
        "removed_prims": removed, "added_prims": added, "unchanged_common_prim_count": len(before.keys() & after.keys()),
        "camera_nominal_pos_in_robotiq_base_m": camera_pos.tolist(),
        "camera_nominal_opengl_quat_wxyz": [float(camera_rot[3]), *map(float, camera_rot[:3])],
        "mass_caveat": "PhysX auto-computed robotiq_base_link mass changes from 1.337838 kg on D415 to 0.709552 kg on D405 R2; no mass/inertia is authored by this builder.",
    }
    (output.parent / "geometry_only_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "output", "output_sha256", "removed_prims", "added_prims", "unchanged_common_prim_count")}, indent=2))


if __name__ == "__main__":
    main()
