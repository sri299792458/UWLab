# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze reachable tabletop regions for the lab right-arm Vention setup.

This is a fast offline probe. It reads the processed lab Vention USD,
samples points on ``/lab_vention/visuals/vention_mat``, transforms them into
the selected UR5e base frame, and solves calibrated UR5e IK for simple probe
poses.

The result is a conservative planning aid, not a final collision-checked reset
policy. Use it to decide where to focus object reset sampling before running
heavier Isaac/PhysX validation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from pxr import Gf, Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_LAYOUT_CFG = (
    REPO_ROOT
    / "source"
    / "uwlab_tasks"
    / "uwlab_tasks"
    / "manager_based"
    / "manipulation"
    / "omnireset"
    / "config"
    / "ur5e_robotiq_2f85"
    / "lab_layout_cfg.py"
)

ROBOT_METADATA_REL_PATH = (
    "Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/metadata.yaml"
)
FALLBACK_METADATA_PATH = Path("/data/kanth042/tmp/metadata.yaml")

R_180Z = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _load_lab_layout() -> Any:
    spec = importlib.util.spec_from_file_location("lab_layout_cfg", LAB_LAYOUT_CFG)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load lab layout config: {LAB_LAYOUT_CFG}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_metadata_path(metadata_path: str | None) -> Path:
    if metadata_path:
        return Path(metadata_path).expanduser().resolve()

    try:
        from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path

        return Path(resolve_cloud_path(f"{UWLAB_CLOUD_ASSETS_DIR}/{ROBOT_METADATA_REL_PATH}")).resolve()
    except Exception:
        if FALLBACK_METADATA_PATH.is_file():
            return FALLBACK_METADATA_PATH
        raise RuntimeError(
            "Could not resolve robot metadata.yaml. Pass --metadata-path explicitly."
        )


def _gf_matrix_from_quat_wxyz(quat: tuple[float, float, float, float]) -> Gf.Matrix4d:
    w, x, y, z = quat
    return Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def _gf_translate(vec: tuple[float, float, float]) -> Gf.Matrix4d:
    return Gf.Matrix4d().SetTranslate(Gf.Vec3d(*vec))


def _gf_scale(vec: tuple[float, float, float]) -> Gf.Matrix4d:
    return Gf.Matrix4d().SetScale(Gf.Vec3d(*vec))


def _transform_direction(matrix: Gf.Matrix4d, direction: tuple[float, float, float]) -> torch.Tensor:
    if hasattr(matrix, "TransformDir"):
        out = matrix.TransformDir(Gf.Vec3d(*direction))
    else:
        origin = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        end = matrix.Transform(Gf.Vec3d(*direction))
        out = end - origin
    vec = torch.tensor([float(out[0]), float(out[1]), float(out[2])], dtype=torch.float32)
    return vec / torch.linalg.norm(vec)


def _quat_wxyz_to_matrix_torch(quat: list[float] | tuple[float, float, float, float]) -> torch.Tensor:
    w, x, y, z = [float(v) for v in quat]
    return torch.tensor(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ]
    )


def _load_kinematics(metadata_path: Path, device: torch.device) -> dict[str, torch.Tensor | float | list[float]]:
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    if not metadata or "calibrated_joints" not in metadata:
        raise RuntimeError(f"Invalid UR5e metadata file: {metadata_path}")

    joints = metadata["calibrated_joints"]
    return {
        "joints_xyz": torch.tensor(joints["xyz"], dtype=torch.float32, device=device),
        "joints_rpy": torch.tensor(joints["rpy"], dtype=torch.float32, device=device),
        "gripper_offset_pos": metadata.get("gripper_offset", {}).get("pos", [0.1345, 0.0, 0.0]),
        "gripper_offset_quat": metadata.get("gripper_offset", {}).get("quat", [0.5, 0.5, 0.5, 0.5]),
        "finger_clearance": float(metadata.get("finger_clearance", 0.06)),
    }


def _tabletop_points_world(
    usd_path: Path,
    mesh_path: str,
    table_pos: tuple[float, float, float],
    table_rot: tuple[float, float, float, float],
    table_scale: tuple[float, float, float],
) -> tuple[list[Gf.Vec3d], dict[str, Any]]:
    stage = Usd.Stage.Open(usd_path.as_posix())
    if stage is None:
        raise RuntimeError(f"Could not open lab Vention USD: {usd_path}")

    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"Expected tabletop Mesh at {mesh_path} in {usd_path}")

    mesh = UsdGeom.Mesh(prim)
    points = [Gf.Vec3d(p) for p in mesh.GetPointsAttr().Get()]
    asset_local = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    spawn = _gf_scale(table_scale) * _gf_matrix_from_quat_wxyz(table_rot) * _gf_translate(table_pos)
    xform = asset_local * spawn
    world = [xform.Transform(p) for p in points]

    mins = Gf.Vec3d(
        min(p[0] for p in world),
        min(p[1] for p in world),
        min(p[2] for p in world),
    )
    maxs = Gf.Vec3d(
        max(p[0] for p in world),
        max(p[1] for p in world),
        max(p[2] for p in world),
    )
    info = {
        "world_min": [float(mins[0]), float(mins[1]), float(mins[2])],
        "world_max": [float(maxs[0]), float(maxs[1]), float(maxs[2])],
        "size": [float(maxs[i] - mins[i]) for i in range(3)],
        "top_z": float(maxs[2]),
    }
    return world, info


def _make_grid(table_info: dict[str, Any], nx: int, ny: int, z_clearance: float) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    x_min, y_min, _ = table_info["world_min"]
    x_max, y_max, z_max = table_info["world_max"]
    xs = torch.linspace(float(x_min), float(x_max), nx, dtype=torch.float32)
    ys = torch.linspace(float(y_min), float(y_max), ny, dtype=torch.float32)

    points = []
    rows = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            points.append([float(x), float(y), float(z_max) + z_clearance])
            rows.append({"ix": ix, "iy": iy, "world_x": float(x), "world_y": float(y), "world_z": float(z_max)})
    return torch.tensor(points, dtype=torch.float32), rows


def _world_points_to_robot(points_w: torch.Tensor, robot_pos: tuple[float, float, float], robot_rot: tuple[float, float, float, float]) -> tuple[torch.Tensor, Gf.Matrix4d]:
    robot_world = _gf_matrix_from_quat_wxyz(robot_rot) * _gf_translate(robot_pos)
    world_to_robot = robot_world.GetInverse()
    converted = []
    for point in points_w:
        p = world_to_robot.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        converted.append([float(p[0]), float(p[1]), float(p[2])])
    return torch.tensor(converted, dtype=torch.float32), world_to_robot


def _make_top_down_wrist_rotations(
    world_to_robot: Gf.Matrix4d,
    gripper_offset_quat: list[float],
    yaw_count: int,
) -> torch.Tensor:
    approach_axis_robot = _transform_direction(world_to_robot, (0.0, 0.0, -1.0))
    x_axis = approach_axis_robot / torch.linalg.norm(approach_axis_robot)
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if abs(float(torch.dot(ref, x_axis))) > 0.95:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    y0 = torch.cross(ref, x_axis, dim=0)
    y0 = y0 / torch.linalg.norm(y0)
    z0 = torch.cross(x_axis, y0, dim=0)
    z0 = z0 / torch.linalg.norm(z0)

    if yaw_count <= 1:
        yaws = torch.tensor([0.0], dtype=torch.float32)
    else:
        yaws = torch.linspace(-math.pi, math.pi, yaw_count, dtype=torch.float32)

    rotations = []
    r_wrist_to_gripper = _quat_wxyz_to_matrix_torch(gripper_offset_quat)
    for yaw in yaws:
        y_axis = torch.cos(yaw) * y0 + torch.sin(yaw) * z0
        y_axis = y_axis / torch.linalg.norm(y_axis)
        z_axis = torch.cross(x_axis, y_axis, dim=0)
        z_axis = z_axis / torch.linalg.norm(z_axis)
        r_gripper = torch.stack([x_axis, y_axis, z_axis], dim=1)
        rotations.append(r_gripper @ r_wrist_to_gripper.T)
    return torch.stack(rotations, dim=0)


def _make_ik_seeds(device: torch.device) -> torch.Tensor:
    base = torch.tensor([0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708], dtype=torch.float32)
    seeds = [base]
    for shoulder_pan in (-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi):
        seed = base.clone()
        seed[0] = shoulder_pan
        seeds.append(seed)
    for shoulder_lift in (-2.2, -1.57, -1.0):
        seed = base.clone()
        seed[1] = shoulder_lift
        seeds.append(seed)
    for elbow in (0.8, 1.57, 2.2):
        seed = base.clone()
        seed[2] = elbow
        seeds.append(seed)
    return torch.stack(seeds, dim=0).to(device)


def _fk_and_jacobian(
    q: torch.Tensor,
    joints_xyz: torch.Tensor,
    joints_rpy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = q.shape[0]
    device = q.device
    dtype = q.dtype
    fixed = []
    for i in range(6):
        tf = torch.eye(4, dtype=dtype, device=device)
        tf[:3, :3] = _rpy_to_matrix(joints_rpy[i]).to(device=device, dtype=dtype)
        tf[:3, 3] = joints_xyz[i].to(dtype=dtype)
        fixed.append(tf.unsqueeze(0).expand(n, -1, -1))

    t = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(n, -1, -1).clone()
    joint_frames = []
    for i in range(6):
        joint_frame = torch.bmm(t, fixed[i])
        joint_frames.append(joint_frame)
        theta = q[:, i]
        ct, st = torch.cos(theta), torch.sin(theta)
        rz = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(n, -1, -1).clone()
        rz[:, 0, 0] = ct
        rz[:, 0, 1] = -st
        rz[:, 1, 0] = st
        rz[:, 1, 1] = ct
        t = torch.bmm(joint_frame, rz)

    pos = t[:, :3, 3]
    rot = t[:, :3, :3]
    jac = torch.zeros(n, 6, 6, dtype=dtype, device=device)
    for i, joint_frame in enumerate(joint_frames):
        z_i = joint_frame[:, :3, 2]
        p_i = joint_frame[:, :3, 3]
        jac[:, :3, i] = torch.cross(z_i, pos - p_i, dim=1)
        jac[:, 3:, i] = z_i

    r180 = R_180Z.to(device=device, dtype=dtype)
    pos = torch.matmul(r180, pos.unsqueeze(-1)).squeeze(-1)
    rot = torch.matmul(r180.unsqueeze(0), rot)
    jac[:, :3, :] = torch.matmul(r180.unsqueeze(0), jac[:, :3, :])
    jac[:, 3:, :] = torch.matmul(r180.unsqueeze(0), jac[:, 3:, :])
    return pos, rot, jac


def _rotation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    r_err = torch.bmm(desired, current.transpose(1, 2))
    trace = r_err[:, 0, 0] + r_err[:, 1, 1] + r_err[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-0.999999, 0.999999)
    angle = torch.acos(cos_angle)
    vee = torch.stack(
        [
            r_err[:, 2, 1] - r_err[:, 1, 2],
            r_err[:, 0, 2] - r_err[:, 2, 0],
            r_err[:, 1, 0] - r_err[:, 0, 1],
        ],
        dim=1,
    )
    sin_angle = torch.sin(angle).unsqueeze(1)
    axis = vee / (2.0 * sin_angle.clamp_min(1e-6))
    small = angle < 1e-4
    rotvec = axis * angle.unsqueeze(1)
    rotvec[small] = 0.5 * vee[small]
    return rotvec


def _solve_ik_batch(
    target_pos: torch.Tensor,
    target_rot: torch.Tensor | None,
    seeds: torch.Tensor,
    kinematics: dict[str, torch.Tensor | float | list[float]],
    *,
    iterations: int,
    pos_tol: float,
    rot_tol: float,
    damping: float,
    step_scale: float,
    max_step: float,
    joint_limit: float,
    joint_margin: float,
    orientation_weight: float,
) -> dict[str, torch.Tensor]:
    device = target_pos.device
    target_count = target_pos.shape[0]
    seed_count = seeds.shape[0]
    q = seeds.repeat(target_count, 1)
    target_pos_candidates = target_pos.repeat_interleave(seed_count, dim=0)
    if target_rot is not None:
        target_rot_candidates = target_rot.repeat_interleave(seed_count, dim=0)
    else:
        target_rot_candidates = None

    joints_xyz = kinematics["joints_xyz"]
    joints_rpy = kinematics["joints_rpy"]
    assert isinstance(joints_xyz, torch.Tensor)
    assert isinstance(joints_rpy, torch.Tensor)

    pos_err_norm = torch.full((q.shape[0],), float("inf"), device=device)
    rot_err_norm = torch.zeros((q.shape[0],), device=device)

    for _ in range(iterations):
        pos, rot, jac = _fk_and_jacobian(q, joints_xyz, joints_rpy)
        pos_err = target_pos_candidates - pos
        pos_err_norm = torch.linalg.norm(pos_err, dim=1)

        if target_rot_candidates is None:
            error = pos_err
            jac_used = jac[:, :3, :]
        else:
            rot_err = _rotation_error(target_rot_candidates, rot)
            rot_err_norm = torch.linalg.norm(rot_err, dim=1)
            error = torch.cat([pos_err, orientation_weight * rot_err], dim=1)
            jac_used = torch.cat([jac[:, :3, :], orientation_weight * jac[:, 3:, :]], dim=1)

        jj_t = torch.bmm(jac_used, jac_used.transpose(1, 2))
        eye = torch.eye(jj_t.shape[1], dtype=q.dtype, device=device).unsqueeze(0)
        rhs = error.unsqueeze(-1)
        dq = torch.bmm(jac_used.transpose(1, 2), torch.linalg.solve(jj_t + damping * damping * eye, rhs)).squeeze(-1)
        dq = torch.clamp(step_scale * dq, min=-max_step, max=max_step)
        q = q + dq
        q = (q + math.pi) % (2.0 * math.pi) - math.pi

    pos, rot, _ = _fk_and_jacobian(q, joints_xyz, joints_rpy)
    pos_err_norm = torch.linalg.norm(target_pos_candidates - pos, dim=1)
    if target_rot_candidates is not None:
        rot_err_norm = torch.linalg.norm(_rotation_error(target_rot_candidates, rot), dim=1)
        pose_success = (pos_err_norm <= pos_tol) & (rot_err_norm <= rot_tol)
    else:
        pose_success = pos_err_norm <= pos_tol

    within_joint_limit = torch.all(torch.abs(q) <= (joint_limit - joint_margin), dim=1)
    success = pose_success & within_joint_limit
    return {
        "success": success.reshape(target_count, seed_count),
        "pos_err": pos_err_norm.reshape(target_count, seed_count),
        "rot_err": rot_err_norm.reshape(target_count, seed_count),
    }


def _summarize_points(
    name: str,
    accepted: torch.Tensor,
    min_pos_err: torch.Tensor,
    min_rot_err: torch.Tensor,
    rows: list[dict[str, Any]],
    table_info: dict[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten().tolist()
    area = table_info["size"][0] * table_info["size"][1]
    summary: dict[str, Any] = {
        "name": name,
        "accepted_points": len(accepted_indices),
        "total_points": total,
        "accepted_fraction": len(accepted_indices) / total if total else 0.0,
        "approx_area_m2": area * len(accepted_indices) / total if total else 0.0,
        "min_position_error_m": float(torch.min(min_pos_err).item()) if total else None,
        "median_position_error_m": float(torch.median(min_pos_err).item()) if total else None,
        "max_position_error_m": float(torch.max(min_pos_err).item()) if total else None,
    }
    if torch.any(min_rot_err > 0.0):
        summary.update(
            {
                "min_rotation_error_rad": float(torch.min(min_rot_err).item()),
                "median_rotation_error_rad": float(torch.median(min_rot_err).item()),
                "max_rotation_error_rad": float(torch.max(min_rot_err).item()),
            }
        )

    if accepted_indices:
        xs = [rows[i]["world_x"] for i in accepted_indices]
        ys = [rows[i]["world_y"] for i in accepted_indices]
        summary["accepted_world_bbox"] = {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z_top": table_info["top_z"],
        }
        summary["accepted_bbox_size"] = [max(xs) - min(xs), max(ys) - min(ys)]
    return summary


def _ascii_map(accepted: torch.Tensor, nx: int, ny: int) -> list[str]:
    result = []
    grid = accepted.reshape(ny, nx)
    for iy in reversed(range(ny)):
        result.append("".join("#" if bool(v) else "." for v in grid[iy]))
    return result


def _largest_true_rectangle(
    accepted: torch.Tensor,
    rows: list[dict[str, Any]],
    nx: int,
    ny: int,
    table_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the largest all-accepted grid-aligned rectangle."""

    grid = accepted.reshape(ny, nx).bool()
    heights = [0] * nx
    best: tuple[int, int, int, int, int] | None = None  # cells, x0, x1, y0, y1

    for iy in range(ny):
        for ix in range(nx):
            heights[ix] = heights[ix] + 1 if bool(grid[iy, ix]) else 0

        stack: list[int] = []
        for ix in range(nx + 1):
            current_height = heights[ix] if ix < nx else 0
            while stack and heights[stack[-1]] > current_height:
                top = stack.pop()
                height = heights[top]
                x0 = stack[-1] + 1 if stack else 0
                x1 = ix - 1
                cells = height * (x1 - x0 + 1)
                y0 = iy - height + 1
                y1 = iy
                if best is None or cells > best[0]:
                    best = (cells, x0, x1, y0, y1)
            stack.append(ix)

    if best is None or best[0] == 0:
        return None

    cells, x0, x1, y0, y1 = best

    def row_at(ix: int, iy: int) -> dict[str, Any]:
        return rows[iy * nx + ix]

    min_row = row_at(x0, y0)
    max_row = row_at(x1, y1)
    total_area = table_info["size"][0] * table_info["size"][1]
    return {
        "grid_cells": cells,
        "grid_indices": {"ix": [x0, x1], "iy": [y0, y1]},
        "world_bbox": {
            "x": [min_row["world_x"], max_row["world_x"]],
            "y": [min_row["world_y"], max_row["world_y"]],
            "z_top": table_info["top_z"],
        },
        "bbox_size": [max_row["world_x"] - min_row["world_x"], max_row["world_y"] - min_row["world_y"]],
        "grid_fraction": cells / (nx * ny),
        "approx_area_m2": total_area * cells / (nx * ny),
    }


def _write_outputs(
    output_dir: Path,
    summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    result_columns: dict[str, dict[str, torch.Tensor]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "right_arm_tabletop_reachability.json"
    csv_path = output_dir / "right_arm_tabletop_reachability.csv"

    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    fieldnames = ["ix", "iy", "world_x", "world_y", "world_z"]
    for name in result_columns:
        fieldnames.extend([f"{name}_reachable", f"{name}_min_pos_err_m", f"{name}_min_rot_err_rad"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = dict(row)
            for name, columns in result_columns.items():
                out[f"{name}_reachable"] = int(bool(columns["accepted"][idx]))
                out[f"{name}_min_pos_err_m"] = float(columns["min_pos_err"][idx])
                out[f"{name}_min_rot_err_rad"] = float(columns["min_rot_err"][idx])
            writer.writerow(out)

    return json_path, csv_path


def main() -> None:
    lab_layout = _load_lab_layout()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-vention-usd", default=lab_layout.LAB_VENTION_USD_PATH)
    parser.add_argument("--tabletop-mesh-path", default=f"/lab_vention/{lab_layout.LAB_TABLETOP_PANEL_REL_PATH}")
    parser.add_argument("--metadata-path", default=None, help="UR5e metadata.yaml. Defaults to UWLab asset metadata.")
    parser.add_argument("--output-dir", default="/data/kanth042/converted_assets/lab_vention_asset_v60/reachability")
    parser.add_argument("--nx", type=int, default=41, help="Number of grid samples along tabletop x.")
    parser.add_argument("--ny", type=int, default=21, help="Number of grid samples along tabletop y.")
    parser.add_argument("--probe", choices=("wrist-position", "top-down-gripper", "both"), default="both")
    parser.add_argument("--wrist-clearance", type=float, default=0.05, help="Wrist-position probe height above tabletop.")
    parser.add_argument("--tool-clearance", type=float, default=0.08, help="Tool point height above tabletop for gripper probe.")
    parser.add_argument("--yaw-count", type=int, default=7, help="Top-down gripper yaw samples around approach axis.")
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--pos-tol", type=float, default=0.025)
    parser.add_argument("--rot-tol", type=float, default=0.40)
    parser.add_argument("--damping", type=float, default=0.06)
    parser.add_argument("--step-scale", type=float, default=0.65)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--joint-limit", type=float, default=2.0 * math.pi)
    parser.add_argument("--joint-margin", type=float, default=0.05)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-map", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    metadata_path = _resolve_metadata_path(args.metadata_path)
    kinematics = _load_kinematics(metadata_path, device)
    seeds = _make_ik_seeds(device)

    _, table_info = _tabletop_points_world(
        usd_path=Path(args.lab_vention_usd).expanduser().resolve(),
        mesh_path=args.tabletop_mesh_path,
        table_pos=tuple(lab_layout.LAB_VENTION_POS),
        table_rot=tuple(lab_layout.LAB_VENTION_ROT),
        table_scale=tuple(lab_layout.LAB_VENTION_SCALE),
    )

    summaries: dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "lab_vention_usd": str(Path(args.lab_vention_usd).expanduser().resolve()),
        "tabletop_mesh_path": args.tabletop_mesh_path,
        "tabletop": table_info,
        "robot": {
            "pos": list(lab_layout.LAB_RIGHT_ARM_ROBOT_POS),
            "rot": list(lab_layout.LAB_RIGHT_ARM_ROBOT_ROT),
        },
        "grid": {"nx": args.nx, "ny": args.ny},
        "ik": {
            "seed_count": int(seeds.shape[0]),
            "iterations": args.iterations,
            "pos_tol": args.pos_tol,
            "rot_tol": args.rot_tol,
            "joint_limit": args.joint_limit,
            "joint_margin": args.joint_margin,
        },
        "probes": {},
    }

    result_columns: dict[str, dict[str, torch.Tensor]] = {}
    probe_names = []
    if args.probe in ("wrist-position", "both"):
        probe_names.append("wrist_position")
    if args.probe in ("top-down-gripper", "both"):
        probe_names.append("top_down_gripper")

    wrist_points_w, rows = _make_grid(table_info, args.nx, args.ny, args.wrist_clearance)
    wrist_points_robot, world_to_robot = _world_points_to_robot(
        wrist_points_w, tuple(lab_layout.LAB_RIGHT_ARM_ROBOT_POS), tuple(lab_layout.LAB_RIGHT_ARM_ROBOT_ROT)
    )
    wrist_points_robot = wrist_points_robot.to(device)

    if "wrist_position" in probe_names:
        result = _solve_ik_batch(
            wrist_points_robot,
            None,
            seeds,
            kinematics,
            iterations=args.iterations,
            pos_tol=args.pos_tol,
            rot_tol=args.rot_tol,
            damping=args.damping,
            step_scale=args.step_scale,
            max_step=args.max_step,
            joint_limit=args.joint_limit,
            joint_margin=args.joint_margin,
            orientation_weight=args.orientation_weight,
        )
        accepted = result["success"].any(dim=1).cpu()
        min_pos_err = result["pos_err"].amin(dim=1).cpu()
        min_rot_err = result["rot_err"].amin(dim=1).cpu()
        result_columns["wrist_position"] = {
            "accepted": accepted,
            "min_pos_err": min_pos_err,
            "min_rot_err": min_rot_err,
        }
        summaries["probes"]["wrist_position"] = _summarize_points(
            "wrist_position", accepted, min_pos_err, min_rot_err, rows, table_info
        )
        summaries["probes"]["wrist_position"]["z_clearance_m"] = args.wrist_clearance
        summaries["probes"]["wrist_position"]["largest_all_reachable_rectangle"] = _largest_true_rectangle(
            accepted, rows, args.nx, args.ny, table_info
        )
        summaries["probes"]["wrist_position"]["ascii_map"] = _ascii_map(accepted, args.nx, args.ny)

    if "top_down_gripper" in probe_names:
        tool_points_w, _ = _make_grid(table_info, args.nx, args.ny, args.tool_clearance)
        tool_points_robot, world_to_robot = _world_points_to_robot(
            tool_points_w, tuple(lab_layout.LAB_RIGHT_ARM_ROBOT_POS), tuple(lab_layout.LAB_RIGHT_ARM_ROBOT_ROT)
        )
        tool_points_robot = tool_points_robot.to(device)
        wrist_rotations = _make_top_down_wrist_rotations(
            world_to_robot,
            kinematics["gripper_offset_quat"],  # type: ignore[arg-type]
            args.yaw_count,
        ).to(device)
        offset = torch.tensor(kinematics["gripper_offset_pos"], dtype=torch.float32, device=device)
        wrist_targets = []
        wrist_target_rots = []
        for rot in wrist_rotations:
            wrist_targets.append(tool_points_robot - torch.matmul(rot, offset))
            wrist_target_rots.append(rot.unsqueeze(0).expand(tool_points_robot.shape[0], -1, -1))
        target_pos = torch.cat(wrist_targets, dim=0)
        target_rot = torch.cat(wrist_target_rots, dim=0)

        result = _solve_ik_batch(
            target_pos,
            target_rot,
            seeds,
            kinematics,
            iterations=args.iterations,
            pos_tol=args.pos_tol,
            rot_tol=args.rot_tol,
            damping=args.damping,
            step_scale=args.step_scale,
            max_step=args.max_step,
            joint_limit=args.joint_limit,
            joint_margin=args.joint_margin,
            orientation_weight=args.orientation_weight,
        )
        point_count = len(rows)
        yaw_count = args.yaw_count
        success = result["success"].reshape(yaw_count, point_count, seeds.shape[0]).permute(1, 0, 2)
        pos_err = result["pos_err"].reshape(yaw_count, point_count, seeds.shape[0]).permute(1, 0, 2)
        rot_err = result["rot_err"].reshape(yaw_count, point_count, seeds.shape[0]).permute(1, 0, 2)
        accepted = success.any(dim=(1, 2)).cpu()
        min_pos_err = pos_err.amin(dim=(1, 2)).cpu()
        min_rot_err = rot_err.amin(dim=(1, 2)).cpu()
        result_columns["top_down_gripper"] = {
            "accepted": accepted,
            "min_pos_err": min_pos_err,
            "min_rot_err": min_rot_err,
        }
        summaries["probes"]["top_down_gripper"] = _summarize_points(
            "top_down_gripper", accepted, min_pos_err, min_rot_err, rows, table_info
        )
        summaries["probes"]["top_down_gripper"]["tool_clearance_m"] = args.tool_clearance
        summaries["probes"]["top_down_gripper"]["yaw_count"] = args.yaw_count
        summaries["probes"]["top_down_gripper"]["gripper_offset_pos"] = kinematics["gripper_offset_pos"]
        summaries["probes"]["top_down_gripper"]["gripper_offset_quat"] = kinematics["gripper_offset_quat"]
        summaries["probes"]["top_down_gripper"]["largest_all_reachable_rectangle"] = _largest_true_rectangle(
            accepted, rows, args.nx, args.ny, table_info
        )
        summaries["probes"]["top_down_gripper"]["ascii_map"] = _ascii_map(accepted, args.nx, args.ny)

    json_path, csv_path = _write_outputs(Path(args.output_dir), summaries, rows, result_columns)

    print(f"[INFO]: Metadata: {metadata_path}")
    print(f"[INFO]: Tabletop size: {[round(v, 6) for v in table_info['size']]}")
    print(f"[INFO]: Grid: {args.nx} x {args.ny}")
    for name, summary in summaries["probes"].items():
        bbox = summary.get("accepted_world_bbox")
        print(
            f"[INFO]: {name}: {summary['accepted_points']}/{summary['total_points']} "
            f"({summary['accepted_fraction']:.3f}), area ~= {summary['approx_area_m2']:.3f} m^2"
        )
        if bbox:
            print(
                f"        bbox x={bbox['x'][0]:.4f}..{bbox['x'][1]:.4f}, "
                f"y={bbox['y'][0]:.4f}..{bbox['y'][1]:.4f}"
            )
        rect = summary.get("largest_all_reachable_rectangle")
        if rect:
            rect_bbox = rect["world_bbox"]
            print(
                f"        largest all-reachable rect: x={rect_bbox['x'][0]:.4f}..{rect_bbox['x'][1]:.4f}, "
                f"y={rect_bbox['y'][0]:.4f}..{rect_bbox['y'][1]:.4f}, "
                f"area ~= {rect['approx_area_m2']:.3f} m^2"
            )
        if args.print_map:
            print(f"[INFO]: {name} map (# reachable):")
            for line in summary["ascii_map"]:
                print(line)
    print(f"[INFO]: Wrote {json_path}")
    print(f"[INFO]: Wrote {csv_path}")


if __name__ == "__main__":
    main()
