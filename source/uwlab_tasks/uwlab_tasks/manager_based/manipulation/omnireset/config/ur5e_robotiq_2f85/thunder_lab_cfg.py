# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thunder-calibrated easy cube task on the measured lab mount and tabletop.

The easy goal/start patch keeps its world XY coordinates. Object heights are
measured from the lab tabletop; the stock D415 hand, 40 mm cubes, control,
reward, and reset distributions are inherited from ThunderCalibration.
"""

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import thunder_calibration_cfg as thunder


TABLE_POS = (1.793445, 0.340075, -0.030851)
TABLE_ROT = (0.5, 0.5, 0.5, 0.5)
TABLE_TOP_Z = 0.84235
ROBOT_POS = (0.177660, 0.377695, 1.466000)
ROBOT_ROT = (0.5, 0.5, 0.5, 0.5)


def _configure_thunder_lab(cfg):
    old_robot_pos = cfg.scene.robot.init_state.pos
    old_support_z = cfg.scene.ur5_metal_support.init_state.pos[2]
    old_jitter_centers = [0.0, 0.0, 0.0]
    table_usd = Path(os.environ.get("UWLAB_ASSET_ROOT", "/data/kanth042/converted_assets")) / (
        "lab_vention_asset_v60/lab_vention.usd"
    )
    if not table_usd.is_file():
        raise FileNotFoundError(f"Thunder lab table USD is missing: {table_usd}")

    cfg.scene.robot.init_state.pos = ROBOT_POS
    cfg.scene.robot.init_state.rot = ROBOT_ROT
    cfg.scene.table.spawn.usd_path = str(table_usd.resolve())
    cfg.scene.table.spawn.scale = (0.001, 0.001, 0.001)
    cfg.scene.table.init_state.pos = TABLE_POS
    cfg.scene.table.init_state.rot = TABLE_ROT
    cfg.scene.ground.init_state.pos = (0.0, 0.0, 0.0)
    # The measured lab CAD includes the mount; the upstream plate is redundant.
    cfg.scene.ur5_metal_support = None

    if hasattr(cfg.events, "reset_robot_pose"):
        root_reset = cfg.events.reset_robot_pose.params
        root_reset["asset_cfgs"] = {"robot": SceneEntityCfg("robot")}
        # Recenter the original jitter without changing its width: X/Z +/-10
        # mm and Y +/-20 mm. The upstream Y range has a nonzero center.
        for i, axis in enumerate(("x", "y", "z")):
            low, high = root_reset["pose_range"][axis]
            old_jitter_centers[i] = (low + high) / 2
            half_width = (high - low) / 2
            root_reset["pose_range"][axis] = (-half_width, half_width)

    for name in ("reset_receptive_object_pose", "reset_insertive_object_pose"):
        if hasattr(cfg.events, name):
            params = getattr(cfg.events, name).params
            params.pop("offset_asset_cfg", None)
            # Keep the easy XY/yaw proposals and bottom offset; only move the
            # vertical reference from the old support to the lab tabletop.
            z_range = params["pose_range"]["z"]
            params["pose_range"]["z"] = tuple(TABLE_TOP_Z + z for z in z_range)

    if hasattr(cfg.events, "reset_end_effector_pose"):
        # The sampler adds pose_range_b translations to the robot's WORLD root
        # position. Preserve its old world XY and table-relative Z proposals.
        xyz = cfg.events.reset_end_effector_pose.params["pose_range_b"]
        for i, axis in enumerate(("x", "y", "z")):
            shift = old_robot_pos[i] + old_jitter_centers[i] - ROBOT_POS[i]
            if axis == "z":
                shift += TABLE_TOP_Z - old_support_z
            xyz[axis] = tuple(value + shift for value in xyz[axis])

    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_LAB_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderLab"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset

    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success.params["pos_z_threshold"] += TABLE_TOP_Z - old_support_z


@configclass
class ThunderLabObjectAnywhereEEAnywhereCfg(thunder.ThunderCalibrationObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_lab(self)


@configclass
class ThunderLabObjectRestingEEGraspedCfg(thunder.ThunderCalibrationObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_lab(self)


@configclass
class ThunderLabObjectAnywhereEEGraspedCfg(thunder.ThunderCalibrationObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_lab(self)


@configclass
class ThunderLabObjectPartiallyAssembledEEGraspedCfg(thunder.ThunderCalibrationObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_lab(self)


@configclass
class ThunderLabTrainCfg(thunder.ThunderCalibrationTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_lab(self)
