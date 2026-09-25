# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thunder arm calibration on the upstream stock-hand, fixed-goal cube task.

This is the calibration-only step of the lab port. The upstream mount, D415,
cubes, physics, controller gains, sampling and acceptance settings are retained.
Generate calibration-compatible full-arm states in a separate dataset directory.
"""

import os
from pathlib import Path

import yaml
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import easy_cube_cfg as easy

CALIBRATION_HASH = "calib_10185139869934003756"


def _configure_thunder_calibration(cfg):
    asset_directory = os.environ.get("UWLAB_THUNDER_CALIBRATION_ASSET_DIR")
    if not asset_directory:
        raise ValueError(
            "Set UWLAB_THUNDER_CALIBRATION_ASSET_DIR to the output of "
            "scripts_v2/tools/thunder_calibration/build_asset.py"
        )
    asset_directory = Path(asset_directory).expanduser().resolve()
    robot = asset_directory / "robot.usd"
    metadata_path = asset_directory / "metadata.yaml"
    if not robot.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Expected robot.usd and metadata.yaml in {asset_directory}")
    metadata = yaml.safe_load(metadata_path.read_text())
    if metadata.get("kinematics_calibration", {}).get("hash") != CALIBRATION_HASH:
        raise ValueError(f"Unexpected Thunder calibration in {metadata_path}")
    cfg.scene.robot.spawn.usd_path = str(robot)
    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_CALIBRATION_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderCalibration"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset


@configclass
class ThunderCalibrationObjectAnywhereEEAnywhereCfg(easy.CubeEasyObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)


@configclass
class ThunderCalibrationObjectRestingEEGraspedCfg(easy.CubeEasyObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)


@configclass
class ThunderCalibrationObjectAnywhereEEGraspedCfg(easy.CubeEasyObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)


@configclass
class ThunderCalibrationObjectPartiallyAssembledEEGraspedCfg(easy.CubeEasyObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)


@configclass
class ThunderCalibrationTrainCfg(easy.CubeEasyTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_thunder_calibration(self)
