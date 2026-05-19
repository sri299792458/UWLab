# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the lab right-arm reset scene with repeated physics resets.

This is a lightweight sanity check for the migration from the old flat UWLab
Vention setup to the lab Vention frame. It samples the lab-specific
ObjectAnywhereEEAnywhere reset config, lets physics settle briefly, then reports
whether objects remain on/above the measured tabletop.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate lab right-arm reset placement over repeated resets.")
parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments to sample per reset.")
parser.add_argument("--num-resets", type=int, default=10, help="Number of reset batches to sample.")
parser.add_argument("--settle-steps", type=int, default=8, help="RL steps to simulate after each reset batch.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("/data/kanth042/converted_assets/lab_vention_asset_v60/validation"),
    help="Directory for CSV/JSON validation outputs.",
)
parser.add_argument(
    "--z-margin",
    type=float,
    default=0.03,
    help="Allowed distance below tabletop top before an object is considered fallen through.",
)
parser.add_argument(
    "--receptive-drift-threshold",
    type=float,
    default=0.05,
    help="Reported receptive-object XY drift threshold after settling, in meters.",
)
parser.add_argument(
    "--fail-on-receptive-drift",
    action="store_true",
    help="Treat receptive-object drift as a failure. By default it is only telemetry.",
)
parser.add_argument(
    "--xy-escape-margin",
    type=float,
    default=0.15,
    help="Allowed margin outside the reset XY rectangle after settling, in meters.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit non-zero when any support failure is observed. By default this tool reports candidate statistics.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.lab_layout_cfg import (  # noqa: E402
    LAB_RIGHT_ARM_RESET_X_RANGE,
    LAB_RIGHT_ARM_RESET_Y_RANGE,
    LAB_TABLETOP_TOP_Z,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.reset_states_cfg import (  # noqa: E402
    LabRightArmObjectAnywhereEEAnywhereResetStatesCfg,
)


TASK = "OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-LabRightArm-v0"


def _positions_w(env, asset_name: str) -> np.ndarray:
    asset = env.scene[asset_name]
    return asset.data.root_pos_w.detach().cpu().numpy().copy()


def _local_xy(env, pos_w: np.ndarray) -> np.ndarray:
    origins = env.scene.env_origins.detach().cpu().numpy()
    return pos_w[:, :2] - origins[:, :2]


def _action(env) -> torch.Tensor:
    action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    if action.shape[-1] > 0:
        action[:, -1] = 1.0
    return action


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(x) for x in values.tolist()]


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "lab_right_arm_reset_validation.csv"
    json_path = args_cli.output_dir / "lab_right_arm_reset_validation_summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO]: Wrote {csv_path}")
    print(f"[INFO]: Wrote {json_path}")


def main() -> None:
    env_cfg = LabRightArmObjectAnywhereEEAnywhereResetStatesCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = None
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, (args_cli.settle_steps + 2) * env_cfg.decimation * env_cfg.sim.dt)

    env = gym.make(TASK, cfg=env_cfg).unwrapped

    rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    action = _action(env)

    x_min, x_max = LAB_RIGHT_ARM_RESET_X_RANGE
    y_min, y_max = LAB_RIGHT_ARM_RESET_Y_RANGE

    print("[INFO]: Validating lab right-arm reset scene")
    print(f"[INFO]: task={TASK}")
    print(f"[INFO]: num_envs={args_cli.num_envs}, num_resets={args_cli.num_resets}, settle_steps={args_cli.settle_steps}")
    print(f"[INFO]: tabletop_top_z={LAB_TABLETOP_TOP_Z:.5f}")
    print(f"[INFO]: reset_x={LAB_RIGHT_ARM_RESET_X_RANGE}, reset_y={LAB_RIGHT_ARM_RESET_Y_RANGE}")

    for reset_idx in range(args_cli.num_resets):
        env.reset()
        receptive_start_w = _positions_w(env, "receptive_object")
        insertive_start_w = _positions_w(env, "insertive_object")

        for _ in range(args_cli.settle_steps):
            env.step(action)

        receptive_end_w = _positions_w(env, "receptive_object")
        insertive_end_w = _positions_w(env, "insertive_object")
        receptive_start_xy = _local_xy(env, receptive_start_w)
        receptive_end_xy = _local_xy(env, receptive_end_w)
        insertive_start_xy = _local_xy(env, insertive_start_w)
        insertive_end_xy = _local_xy(env, insertive_end_w)

        receptive_drift_xy = np.linalg.norm(receptive_end_xy - receptive_start_xy, axis=1)

        for env_idx in range(args_cli.num_envs):
            nonfinite = not (
                np.isfinite(receptive_start_w[env_idx]).all()
                and np.isfinite(receptive_end_w[env_idx]).all()
                and np.isfinite(insertive_start_w[env_idx]).all()
                and np.isfinite(insertive_end_w[env_idx]).all()
            )
            receptive_below = receptive_end_w[env_idx, 2] < LAB_TABLETOP_TOP_Z - args_cli.z_margin
            insertive_below = insertive_end_w[env_idx, 2] < LAB_TABLETOP_TOP_Z - args_cli.z_margin
            receptive_drifted = receptive_drift_xy[env_idx] > args_cli.receptive_drift_threshold
            receptive_escaped = (
                receptive_end_xy[env_idx, 0] < x_min - args_cli.xy_escape_margin
                or receptive_end_xy[env_idx, 0] > x_max + args_cli.xy_escape_margin
                or receptive_end_xy[env_idx, 1] < y_min - args_cli.xy_escape_margin
                or receptive_end_xy[env_idx, 1] > y_max + args_cli.xy_escape_margin
            )
            insertive_escaped = (
                insertive_end_xy[env_idx, 0] < x_min - args_cli.xy_escape_margin
                or insertive_end_xy[env_idx, 0] > x_max + args_cli.xy_escape_margin
                or insertive_end_xy[env_idx, 1] < y_min - args_cli.xy_escape_margin
                or insertive_end_xy[env_idx, 1] > y_max + args_cli.xy_escape_margin
            )
            failed = nonfinite or receptive_below or insertive_below or receptive_escaped or insertive_escaped
            if args_cli.fail_on_receptive_drift:
                failed = failed or receptive_drifted

            row = {
                "reset_idx": reset_idx,
                "env_idx": env_idx,
                "receptive_start_xyz_w": _as_float_list(receptive_start_w[env_idx]),
                "receptive_end_xyz_w": _as_float_list(receptive_end_w[env_idx]),
                "insertive_start_xyz_w": _as_float_list(insertive_start_w[env_idx]),
                "insertive_end_xyz_w": _as_float_list(insertive_end_w[env_idx]),
                "receptive_start_xy_local": _as_float_list(receptive_start_xy[env_idx]),
                "receptive_end_xy_local": _as_float_list(receptive_end_xy[env_idx]),
                "insertive_start_xy_local": _as_float_list(insertive_start_xy[env_idx]),
                "insertive_end_xy_local": _as_float_list(insertive_end_xy[env_idx]),
                "receptive_xy_drift": float(receptive_drift_xy[env_idx]),
                "nonfinite": nonfinite,
                "receptive_below_tabletop": bool(receptive_below),
                "insertive_below_tabletop": bool(insertive_below),
                "receptive_drifted": bool(receptive_drifted),
                "receptive_escaped": bool(receptive_escaped),
                "insertive_escaped": bool(insertive_escaped),
                "failed": bool(failed),
            }
            rows.append(row)
            if receptive_drifted:
                drift_rows.append(row)
            if failed:
                bad_rows.append(row)

        print(
            f"[INFO]: reset {reset_idx + 1:03d}/{args_cli.num_resets}: "
            f"support failures {len(bad_rows)}/{len(rows)}, "
            f"drift events {len(drift_rows)}/{len(rows)}"
        )

    env.close()

    receptive_end_z = np.array([row["receptive_end_xyz_w"][2] for row in rows], dtype=np.float64)
    insertive_end_z = np.array([row["insertive_end_xyz_w"][2] for row in rows], dtype=np.float64)
    receptive_drift = np.array([row["receptive_xy_drift"] for row in rows], dtype=np.float64)
    summary = {
        "task": TASK,
        "num_envs": args_cli.num_envs,
        "num_resets": args_cli.num_resets,
        "num_samples": len(rows),
        "num_failures": len(bad_rows),
        "failure_rate": float(len(bad_rows) / len(rows)) if rows else 0.0,
        "num_receptive_drift_events": len(drift_rows),
        "receptive_drift_event_rate": float(len(drift_rows) / len(rows)) if rows else 0.0,
        "tabletop_top_z": LAB_TABLETOP_TOP_Z,
        "reset_x_range": LAB_RIGHT_ARM_RESET_X_RANGE,
        "reset_y_range": LAB_RIGHT_ARM_RESET_Y_RANGE,
        "z_margin": args_cli.z_margin,
        "receptive_drift_threshold": args_cli.receptive_drift_threshold,
        "fail_on_receptive_drift": args_cli.fail_on_receptive_drift,
        "strict": args_cli.strict,
        "xy_escape_margin": args_cli.xy_escape_margin,
        "min_receptive_end_z": float(receptive_end_z.min()) if len(receptive_end_z) else None,
        "min_insertive_end_z": float(insertive_end_z.min()) if len(insertive_end_z) else None,
        "max_receptive_xy_drift": float(receptive_drift.max()) if len(receptive_drift) else None,
        "failure_examples": bad_rows[:10],
        "receptive_drift_examples": drift_rows[:10],
    }

    _write_outputs(rows, summary)

    print("[INFO]: Validation summary")
    print(f"  samples: {summary['num_samples']}")
    print(f"  support failures: {summary['num_failures']} ({summary['failure_rate']:.2%})")
    print(
        f"  receptive drift events: "
        f"{summary['num_receptive_drift_events']} ({summary['receptive_drift_event_rate']:.2%})"
    )
    print(f"  min receptive end z: {summary['min_receptive_end_z']:.5f}")
    print(f"  min insertive end z: {summary['min_insertive_end_z']:.5f}")
    print(f"  max receptive xy drift: {summary['max_receptive_xy_drift']:.5f}")

    if args_cli.strict and bad_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
