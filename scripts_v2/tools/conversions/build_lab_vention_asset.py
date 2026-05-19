# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a UWLab-style USD asset from the lab Vention STEP conversion.

This script expects a de-instanced USD from ``convert_step_to_usd.py
--no-instancing``. It creates a processed asset with the same high-level pattern
as UWLab's support assets:

    /lab_vention
      /visuals
      /collisions

The CAD-exported UR arms are removed from both visuals and collisions because
the simulated UR5e articulation is spawned separately by the task scene.

The collision side follows the UWLab ``pat_vention.usd`` pattern: simple
invisible ``Cube`` prims with ``PhysicsCollisionAPI``. It deliberately does not
copy CAD meshes into collisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_ARM_GROUP_NAMES = ("tn__0_", "tn__0_1_")
DEFAULT_TABLETOP_SOURCE_REL_PATH = "tn__importedPANEL72E663E8E8F2_1147116_om0E/Mesh"
DEFAULT_TABLETOP_TARGET_NAME = "vention_mat"


def _find_assembly_child(source_root: Usd.Prim) -> Usd.Prim:
    """Return the main CAD assembly child under the converter root."""

    candidates = [
        child
        for child in source_root.GetChildren()
        if child.GetTypeName() == "Xform" and child.GetName() not in {"Looks", "Prototypes"}
    ]
    if len(candidates) != 1:
        names = [str(child.GetPath()) for child in candidates]
        raise RuntimeError(f"Expected exactly one assembly Xform under {source_root.GetPath()}, found: {names}")
    return candidates[0]


def _as_tuple(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _compute_component_boxes(
    source_stage: Usd.Stage,
    assembly_child: Usd.Prim,
    arm_group_names: tuple[str, ...],
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Return AABB cube proxies for retained top-level CAD components."""

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
    for child in assembly_child.GetChildren():
        if child.GetName() in arm_group_names:
            continue
        aligned_box = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
        size = aligned_box.GetSize()
        if size[0] <= 0.0 or size[1] <= 0.0 or size[2] <= 0.0:
            continue
        boxes.append((child.GetName(), _as_tuple(aligned_box.GetMidpoint()), _as_tuple(size)))
    return boxes


def _define_collision_cubes(
    stage: Usd.Stage,
    collisions_path: str,
    boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
) -> None:
    """Author invisible cube collision proxies matching UWLab table assets."""

    stage.DefinePrim(collisions_path, "Xform")
    for index, (_source_name, center, size) in enumerate(boxes):
        cube = UsdGeom.Cube.Define(stage, f"{collisions_path}/mesh_{index}")
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(prim).MakeInvisible()

        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
        xformable.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xformable.AddScaleOp().Set(Gf.Vec3d(*size))


def _promote_tabletop_mesh(
    stage: Usd.Stage,
    visuals_path: str,
    assembly_path: str,
    tabletop_source_rel_path: str,
    tabletop_target_name: str,
) -> tuple[str, str | None]:
    """Copy the tabletop mesh to a stable UWLab-style visual path."""

    source_path = f"{assembly_path}/{tabletop_source_rel_path.strip('/')}"
    target_path = f"{visuals_path}/{tabletop_target_name}"

    source_prim = stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid() or source_prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"Expected tabletop source Mesh at {source_path}")

    if stage.GetPrimAtPath(target_path).IsValid():
        stage.RemovePrim(target_path)

    root_layer = stage.GetRootLayer()
    Sdf.CopySpec(root_layer, source_path, root_layer, target_path)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    visuals_to_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(visuals_path))
    source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
    target_transform = source_to_world * visuals_to_world.GetInverse()

    target_prim = stage.GetPrimAtPath(target_path)
    xformable = UsdGeom.Xformable(target_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(target_transform)

    source_parent = source_prim.GetParent()
    removed_parent_path: str | None = None
    if source_parent.IsValid() and source_parent.GetPath() != Sdf.Path(assembly_path):
        if len(source_parent.GetChildren()) == 1:
            removed_parent_path = str(source_parent.GetPath())
            stage.RemovePrim(removed_parent_path)

    return target_path, removed_parent_path


def _count_meshes(stage: Usd.Stage, path: str) -> int:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return 0
    return sum(1 for child in Usd.PrimRange(prim) if child.GetTypeName() == "Mesh")


def build_asset(
    input_path: Path,
    output_path: Path,
    root_name: str,
    arm_group_names: tuple[str, ...],
    tabletop_source_rel_path: str,
    tabletop_target_name: str,
) -> None:
    source_stage = Usd.Stage.Open(input_path.as_posix())
    if source_stage is None:
        raise RuntimeError(f"Failed to open source USD: {input_path}")

    source_root = source_stage.GetDefaultPrim()
    if not source_root.IsValid():
        raise RuntimeError(f"Source USD has no valid default prim: {input_path}")
    assembly_child = _find_assembly_child(source_root)

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(output_path.as_posix())
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))

    root_path = f"/{root_name}"
    visuals_path = f"{root_path}/visuals"
    collisions_path = f"{root_path}/collisions"

    root = stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(root)

    source_layer = source_stage.GetRootLayer()
    target_layer = stage.GetRootLayer()
    Sdf.CopySpec(source_layer, str(source_root.GetPath()), target_layer, visuals_path)

    assembly_rel_path = assembly_child.GetName()
    collision_boxes = _compute_component_boxes(source_stage, assembly_child, arm_group_names)
    removed_paths: list[str] = []
    for arm_group_name in arm_group_names:
        arm_path = f"{visuals_path}/{assembly_rel_path}/{arm_group_name}"
        if stage.RemovePrim(arm_path):
            removed_paths.append(arm_path)

    tabletop_target_path, removed_tabletop_source_path = _promote_tabletop_mesh(
        stage=stage,
        visuals_path=visuals_path,
        assembly_path=f"{visuals_path}/{assembly_rel_path}",
        tabletop_source_rel_path=tabletop_source_rel_path,
        tabletop_target_name=tabletop_target_name,
    )

    _define_collision_cubes(stage, collisions_path, collision_boxes)
    UsdGeom.Imageable(stage.GetPrimAtPath(collisions_path)).MakeInvisible()
    UsdGeom.Imageable(stage.GetPrimAtPath(visuals_path)).MakeVisible()

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    visual_mesh_count = _count_meshes(stage, visuals_path)

    stage.GetRootLayer().Save()

    print(f"[INFO]: Source USD: {input_path}")
    print(f"[INFO]: Output USD: {output_path}")
    print(f"[INFO]: Root prim: {root_path}")
    print(f"[INFO]: Removed CAD arm groups: {len(removed_paths)}")
    for path in removed_paths:
        print(f"  - {path}")
    print(f"[INFO]: Tabletop visual mesh: {tabletop_target_path}")
    if removed_tabletop_source_path is not None:
        print(f"[INFO]: Removed duplicate raw tabletop visual: {removed_tabletop_source_path}")
    print(f"[INFO]: Visual meshes: {visual_mesh_count}")
    print(f"[INFO]: Collision cube proxies: {len(collision_boxes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="De-instanced raw USD from the STEP converter.")
    parser.add_argument("--output", required=True, help="Processed lab Vention USD to write.")
    parser.add_argument("--root-name", default="lab_vention", help="Default/root prim name for the processed asset.")
    parser.add_argument(
        "--remove-arm-group",
        action="append",
        dest="arm_groups",
        help=(
            "CAD arm group name under the main assembly to remove. Can be passed more than once. "
            f"Defaults to {DEFAULT_ARM_GROUP_NAMES}."
        ),
    )
    parser.add_argument(
        "--tabletop-source-rel-path",
        default=DEFAULT_TABLETOP_SOURCE_REL_PATH,
        help=(
            "Tabletop Mesh path relative to the copied CAD assembly prim. "
            f"Defaults to {DEFAULT_TABLETOP_SOURCE_REL_PATH}."
        ),
    )
    parser.add_argument(
        "--tabletop-target-name",
        default=DEFAULT_TABLETOP_TARGET_NAME,
        help=(
            "Stable Mesh prim name to create directly under /<root-name>/visuals. "
            f"Defaults to {DEFAULT_TABLETOP_TARGET_NAME}, matching UWLab's visuals/vention_mat convention."
        ),
    )
    args = parser.parse_args()

    build_asset(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        root_name=args.root_name,
        arm_group_names=tuple(args.arm_groups) if args.arm_groups else DEFAULT_ARM_GROUP_NAMES,
        tabletop_source_rel_path=args.tabletop_source_rel_path,
        tabletop_target_name=args.tabletop_target_name,
    )


if __name__ == "__main__":
    main()
