# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Apply Thunder arm calibration to the upstream D415 stock robot asset only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import shutil
import yaml
from pathlib import Path

from pxr import Gf, Usd, UsdPhysics

CALIBRATION_SOURCE_REPOSITORY = "https://github.com/RPM-lab-UMN/spark-data-collection"
CALIBRATION_SOURCE_PATH = "Hardware/robot_calibration/ur5e/thunder_kinematics.yaml"
ROBOT_NAME = "thunder"
SOURCE_SHA256 = "88e72878dc44065c12a73f60bc791f2fc8f8fee4178a44ad11b4f8bc49cd5378"
SOURCE_METADATA_SHA256 = "c9b6bc99aa76ef4cbb9e4400b552b931c639e39c83fc9ae0710a618f4c3f475b"
JOINT_SPECS = (
    ("shoulder", "shoulder_pan_joint", "shoulder_link"),
    ("upper_arm", "shoulder_lift_joint", "upper_arm_link"),
    ("forearm", "elbow_joint", "forearm_link"),
    ("wrist_1", "wrist_1_joint", "wrist_1_link"),
    ("wrist_2", "wrist_2_joint", "wrist_2_link"),
    ("wrist_3", "wrist_3_joint", "wrist_3_link"),
)

R_180_Z = np.diag((-1.0, -1.0, 1.0))


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale

    quat = np.asarray((w, x, y, z), dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return tuple(float(value) for value in quat)


def _quat_wxyz_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _make_transform(xyz: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _load_kinematics(path: Path) -> tuple[list[dict[str, object]], str]:
    with path.open() as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("kinematics"), dict):
        raise ValueError(f"Expected a top-level 'kinematics' mapping in {path}")

    kinematics = document["kinematics"]
    calibration_hash = str(kinematics.get("hash", ""))
    if not calibration_hash.startswith("calib_"):
        raise ValueError(f"Missing or invalid UR calibration hash in {path}: {calibration_hash!r}")

    transforms: list[dict[str, object]] = []
    for yaml_name, joint_name, link_name in JOINT_SPECS:
        values = kinematics.get(yaml_name)
        if not isinstance(values, dict):
            raise ValueError(f"Missing kinematics.{yaml_name} in {path}")
        xyz = tuple(float(values[axis]) for axis in ("x", "y", "z"))
        rpy = tuple(float(values[axis]) for axis in ("roll", "pitch", "yaw"))
        if not np.all(np.isfinite((*xyz, *rpy))):
            raise ValueError(f"Non-finite transform for kinematics.{yaml_name} in {path}")
        rotation = _rpy_to_matrix(rpy)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"Invalid rotation matrix for kinematics.{yaml_name} in {path}")
        transforms.append(
            {
                "yaml_name": yaml_name,
                "joint_name": joint_name,
                "link_name": link_name,
                "xyz": xyz,
                "rpy": rpy,
                "rotation": rotation,
            }
        )
    return transforms, calibration_hash


def _prim_local_transform(prim: Usd.Prim) -> np.ndarray:
    translate_attr = prim.GetAttribute("xformOp:translate")
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not translate_attr or not orient_attr:
        raise RuntimeError(f"Expected translate/orient xform ops on rigid body {prim.GetPath()}")
    position = translate_attr.Get()
    orient = orient_attr.Get()
    quat = (
        float(orient.GetReal()),
        float(orient.GetImaginary()[0]),
        float(orient.GetImaginary()[1]),
        float(orient.GetImaginary()[2]),
    )
    return _make_transform(tuple(float(value) for value in position), _quat_wxyz_to_matrix(quat))


def _set_prim_local_transform(prim: Usd.Prim, transform: np.ndarray) -> None:
    translate_attr = prim.GetAttribute("xformOp:translate")
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not translate_attr or not orient_attr:
        raise RuntimeError(f"Expected translate/orient xform ops on rigid body {prim.GetPath()}")
    position = transform[:3, 3]
    quat = _matrix_to_quat_wxyz(transform[:3, :3])
    translate_attr.Set(Gf.Vec3d(*(float(value) for value in position)))
    orient_attr.Set(Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:])))


def _find_downstream_rigid_bodies(stage: Usd.Stage, start_path: str) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) == 1 and len(body1) == 1:
            adjacency.setdefault(str(body0[0]), set()).add(str(body1[0]))

    visited = {start_path}
    frontier = [start_path]
    while frontier:
        parent = frontier.pop()
        for child in adjacency.get(parent, set()):
            if child not in visited:
                visited.add(child)
                frontier.append(child)
    return visited


def _apply_calibration(stage: Usd.Stage, transforms: list[dict[str, object]]) -> dict[str, float]:
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError("Input USD does not have a default robot prim")
    root_path = str(root.GetPath())

    wrist_path = f"{root_path}/wrist_3_link"
    downstream_paths = _find_downstream_rigid_bodies(stage, wrist_path)
    if wrist_path not in downstream_paths:
        raise RuntimeError(f"Could not identify the downstream subtree from {wrist_path}")

    old_downstream: dict[str, np.ndarray] = {}
    for path in downstream_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Joint graph target is not a rigid body: {path}")
        if prim.GetParent() != root:
            raise RuntimeError(f"Expected flattened rigid body directly below {root_path}: {path}")
        old_downstream[path] = _prim_local_transform(prim)

    old_wrist = old_downstream[wrist_path]
    cumulative = np.eye(4, dtype=float)
    rep103_from_controller = _make_transform((0.0, 0.0, 0.0), R_180_Z)
    expected_links: dict[str, np.ndarray] = {}

    for index, transform in enumerate(transforms):
        fixed = _make_transform(transform["xyz"], transform["rotation"])
        cumulative = cumulative @ fixed
        expected_link = rep103_from_controller @ cumulative
        expected_links[transform["link_name"]] = expected_link

        joint_path = f"{root_path}/{transform['joint_name']}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"Expected calibrated revolute joint: {joint_path}")
        joint = UsdPhysics.Joint(joint_prim)
        xyz = transform["xyz"]
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(float(value) for value in xyz)))

        if index == 0:
            # The importer merges base_link_inertia into base_link.  Its fixed
            # pi rotation remains on localRot0 and the shoulder calibration is
            # represented by the inverse child-side frame.
            inverse_quat = _matrix_to_quat_wxyz(transform["rotation"].T)
            joint.GetLocalRot1Attr().Set(Gf.Quatf(inverse_quat[0], Gf.Vec3f(*inverse_quat[1:])))
        else:
            quat = _matrix_to_quat_wxyz(transform["rotation"])
            joint.GetLocalRot0Attr().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))

        link_prim = stage.GetPrimAtPath(f"{root_path}/{transform['link_name']}")
        if not link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Expected arm rigid body: {link_prim.GetPath()}")
        _set_prim_local_transform(link_prim, expected_link)

    new_wrist = expected_links["wrist_3_link"]
    wrist_delta = new_wrist @ np.linalg.inv(old_wrist)
    for path, old_transform in old_downstream.items():
        if path == wrist_path:
            continue
        _set_prim_local_transform(stage.GetPrimAtPath(path), wrist_delta @ old_transform)

    root.SetCustomDataByKey("ur5eCalibrationRobot", ROBOT_NAME)
    root.SetCustomDataByKey("ur5eCalibrationSource", CALIBRATION_SOURCE_PATH)

    max_position_error = 0.0
    max_rotation_error = 0.0
    for link_name, expected in expected_links.items():
        actual = _prim_local_transform(stage.GetPrimAtPath(f"{root_path}/{link_name}"))
        max_position_error = max(max_position_error, float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])))
        relative_rotation = actual[:3, :3] @ expected[:3, :3].T
        cos_angle = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        max_rotation_error = max(max_rotation_error, math.acos(cos_angle))

    if max_position_error > 1.0e-9 or max_rotation_error > 1.0e-7:
        raise RuntimeError(
            "Authored zero-pose link transforms do not match the Thunder chain: "
            f"position={max_position_error:.3e} m, rotation={max_rotation_error:.3e} rad"
        )
    return {
        "max_zero_pose_position_error_m": max_position_error,
        "max_zero_pose_rotation_error_rad": max_rotation_error,
        "downstream_body_count": float(len(downstream_paths) - 1),
    }


def _write_metadata(
    source_metadata_path: Path,
    output_metadata_path: Path,
    transforms: list[dict[str, object]],
    calibration_hash: str,
) -> None:
    with source_metadata_path.open() as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a metadata mapping in {source_metadata_path}")

    metadata["calibrated_joints"] = {
        "xyz": [list(transform["xyz"]) for transform in transforms],
        "rpy": [list(transform["rpy"]) for transform in transforms],
    }
    metadata["kinematics_calibration"] = {
        "robot": ROBOT_NAME,
        "hash": calibration_hash,
        "source_repository": CALIBRATION_SOURCE_REPOSITORY,
        "source_path": CALIBRATION_SOURCE_PATH,
    }
    # These parameters came with UWLab's reference asset.  Keep them so the
    # existing Stage-2 configuration remains loadable, but record that they
    # have not yet been identified on Thunder.
    if "sysid" in metadata:
        metadata["sysid_provenance"] = {
            "status": "inherited_reference_values",
            "robot": "UWLab reference UR5e",
            "validated_for_thunder": False,
        }

    with output_metadata_path.open("w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(stage: Usd.Stage) -> dict:
    result = {}
    for prim in stage.Traverse():
        result[str(prim.GetPath())] = {
            "type": str(prim.GetTypeName()),
            "schemas": list(prim.GetAppliedSchemas()),
            "attrs": {attr.GetName(): repr(attr.Get()) for attr in prim.GetAttributes()},
            "rels": {rel.GetName(): list(map(str, rel.GetTargets())) for rel in prim.GetRelationships()},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kinematics-yaml", type=Path, default=Path(__file__).with_name("thunder_kinematics.yaml"))
    args = parser.parse_args()
    source = args.source_usd.resolve()
    output = args.output_dir.resolve()
    if digest(source) != SOURCE_SHA256:
        raise SystemExit("Source USD differs from pinned upstream D415 asset")
    source_metadata = source.with_name("metadata.yaml")
    if not source_metadata.is_file():
        raise SystemExit("Source metadata.yaml is missing")
    if digest(source_metadata) != SOURCE_METADATA_SHA256:
        raise SystemExit("Source metadata differs from pinned upstream D415 metadata")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output}")
    transforms, calibration_hash = _load_kinematics(args.kinematics_yaml)
    output.mkdir(parents=True, exist_ok=True)
    robot = output / "robot.usd"
    shutil.copy2(source, robot)
    stage = Usd.Stage.Open(str(robot))
    before = snapshot(stage)
    validation = _apply_calibration(stage, transforms)
    after = snapshot(stage)
    if before.keys() != after.keys():
        raise RuntimeError("Calibration changed the USD prim set")
    arm_joints = {item["joint_name"] for item in transforms}
    changes = []
    for path, old in before.items():
        new = after[path]
        if (
            any(old[key] != new[key] for key in ("type", "schemas", "rels"))
            or old["attrs"].keys() != new["attrs"].keys()
        ):
            raise RuntimeError(f"Calibration changed prim structure: {path}")
        prim = stage.GetPrimAtPath(path)
        for key, value in old["attrs"].items():
            if value == new["attrs"][key]:
                continue
            allowed = (
                prim.HasAPI(UsdPhysics.RigidBodyAPI) and key in ("xformOp:translate", "xformOp:orient")
            ) or (
                prim.GetName() in arm_joints and key in ("physics:localPos0", "physics:localRot0", "physics:localRot1")
            )
            if not allowed:
                raise RuntimeError(f"Disallowed USD attribute change: {path} {key}")
            changes.append({"prim": path, "attribute": key, "before": value, "after": new["attrs"][key]})
    stage.GetRootLayer().Save()
    metadata = output / "metadata.yaml"
    _write_metadata(source_metadata, metadata, transforms, calibration_hash)
    source_meta = yaml.safe_load(source_metadata.read_text())
    output_meta = yaml.safe_load(metadata.read_text())
    permitted = {"calibrated_joints", "kinematics_calibration", "sysid_provenance"}
    if {key: value for key, value in source_meta.items() if key not in permitted} != {
        key: value for key, value in output_meta.items() if key not in permitted
    }:
        raise RuntimeError("Unrelated robot metadata changed")
    report = {
        "status": "PASS", "source": str(source), "source_sha256": digest(source),
        "source_metadata": str(source_metadata), "source_metadata_sha256": digest(source_metadata),
        "kinematics_yaml": str(args.kinematics_yaml.resolve()), "kinematics_yaml_sha256": digest(args.kinematics_yaml),
        "output": str(robot), "output_sha256": digest(robot), "metadata_sha256": digest(metadata),
        "calibration_hash": calibration_hash, "validation": validation, "attribute_changes": changes,
        "allowed_attributes": ["arm-joint physics:localPos0/Rot0/Rot1", "rigid-body xformOp:translate/orient"],
        "preserved": (
            "All geometry, mass properties, materials, collisions, gripper linkage and mimic schemas, "
            "joint limits and controller gains."
        ),
        "provenance": (
            "Calibration geometry helpers ported from UWLab "
            "scripts_v2/tools/conversions/build_thunder_d405_robot_asset.py; "
            "pure-calibration procedure checked against R86 build_may_thunder.py. "
            "No D405 camera, mount, mass, or mimic repair is applied."
        ),
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {key: report[key] for key in ("status", "output", "output_sha256", "calibration_hash", "validation")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
