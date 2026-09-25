# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thunder arm calibration on the upstream stock-hand, fixed-goal cube task.

This combines Thunder arm calibration with +/-180 degree wrist limits. The
upstream mount, D415, cubes, controller gains and reset proposals are retained;
reset acceptance and bank loading also check the wrist bounds. Generate
compatible full-arm states in a separate dataset directory.
"""

import os
from pathlib import Path

import yaml
from pxr import Usd, UsdPhysics
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import easy_cube_cfg as easy

CALIBRATION_HASH = "calib_10185139869934003756"
WRIST_JOINTS = ["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def _configure_thunder_calibration(cfg):
    asset_directory = os.environ.get("UWLAB_THUNDER_CALIBRATION_ASSET_DIR")
    if not asset_directory:
        raise ValueError(
            "Set UWLAB_THUNDER_CALIBRATION_ASSET_DIR to the output of "
            "scripts_v2/tools/thunder_calibration/apply_wrist_limits.py"
        )
    asset_directory = Path(asset_directory).expanduser().resolve()
    robot = asset_directory / "robot.usd"
    metadata_path = asset_directory / "metadata.yaml"
    if not robot.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Expected robot.usd and metadata.yaml in {asset_directory}")
    metadata = yaml.safe_load(metadata_path.read_text())
    if metadata.get("kinematics_calibration", {}).get("hash") != CALIBRATION_HASH:
        raise ValueError(f"Unexpected Thunder calibration in {metadata_path}")
    stage = Usd.Stage.Open(str(robot))
    if not stage or not stage.GetDefaultPrim().IsValid():
        raise ValueError(f"Cannot open calibrated wrist-limit USD: {robot}")
    for name in WRIST_JOINTS:
        prim = stage.GetPrimAtPath(f"{stage.GetDefaultPrim().GetPath()}/{name}")
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Missing wrist revolute joint {name} in {robot}")
        joint = UsdPhysics.RevoluteJoint(prim)
        if joint.GetLowerLimitAttr().Get() != -180.0 or joint.GetUpperLimitAttr().Get() != 180.0:
            raise ValueError(f"Expected [-180, 180] degree wrist limits on {name} in {robot}")
    cfg.scene.robot.spawn.usd_path = str(robot)
    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_CALIBRATION_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderWrist180"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset


def _configure_generation_limits(cfg):
    cfg.terminations.success.params["joint_limit_joint_names"] = WRIST_JOINTS.copy()


@configclass
class ThunderCalibrationObjectAnywhereEEAnywhereCfg(easy.CubeEasyObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
        _configure_generation_limits(self)


@configclass
class ThunderCalibrationObjectRestingEEGraspedCfg(easy.CubeEasyObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
        _configure_generation_limits(self)


@configclass
class ThunderCalibrationObjectAnywhereEEGraspedCfg(easy.CubeEasyObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
        _configure_generation_limits(self)


@configclass
class ThunderCalibrationObjectPartiallyAssembledEEGraspedCfg(easy.CubeEasyObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
        _configure_generation_limits(self)


@configclass
class ThunderCalibrationTrainCfg(easy.CubeEasyTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
        self.events.reset_from_reset_states.params["joint_limit_joint_names"] = WRIST_JOINTS.copy()
