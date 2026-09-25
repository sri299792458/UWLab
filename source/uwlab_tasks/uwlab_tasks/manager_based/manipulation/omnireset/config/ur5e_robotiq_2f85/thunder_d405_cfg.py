# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Centered Thunder cube task with the D405 R2 camera assembly and stock hand.

The R2 mount, camera envelope, and collider replace the D415 assembly. All
other scene, reset, controller, reward, and PPO settings come from the centered
D415 task. PhysX derives a different gripper-base inertia from these colliders.
"""

import hashlib
import os
from pathlib import Path

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import thunder_centered_cfg as centered


ROBOT_SHA256 = "00eb01b7c169b223bfe3e76f8be1285bc008e784db43a4a21b6eef9f5372ee90"
METADATA_SHA256 = "625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _configure_d405(cfg):
    asset_directory = os.environ.get("UWLAB_THUNDER_D405_ASSET_DIR")
    if not asset_directory:
        raise ValueError("Set UWLAB_THUNDER_D405_ASSET_DIR to the reviewed D405 R2 stock-hand asset directory")
    asset_directory = Path(asset_directory).expanduser().resolve()
    robot = asset_directory / "robot.usd"
    metadata = asset_directory / "metadata.yaml"
    if not robot.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Expected robot.usd and metadata.yaml in {asset_directory}")
    if _digest(robot) != ROBOT_SHA256 or _digest(metadata) != METADATA_SHA256:
        raise ValueError(f"D405 R2 stock-hand asset differs from reviewed build: {asset_directory}")
    cfg.scene.robot.spawn.usd_path = str(robot)

    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_D405_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderD405"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset


@configclass
class ThunderD405ObjectAnywhereEEAnywhereCfg(centered.ThunderCenteredObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_d405(self)


@configclass
class ThunderD405ObjectRestingEEGraspedCfg(centered.ThunderCenteredObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_d405(self)


@configclass
class ThunderD405ObjectAnywhereEEGraspedCfg(centered.ThunderCenteredObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_d405(self)


@configclass
class ThunderD405ObjectPartiallyAssembledEEGraspedCfg(centered.ThunderCenteredObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_d405(self)


@configclass
class ThunderD405TrainCfg(centered.ThunderCenteredTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_d405(self)
