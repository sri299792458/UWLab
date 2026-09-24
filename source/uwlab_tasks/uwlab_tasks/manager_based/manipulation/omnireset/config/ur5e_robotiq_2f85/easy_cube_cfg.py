# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-goal cube pilot using the upstream robot, controller and reset generator.

This is an explicit local easy-task definition, not a released upstream benchmark.
The lower cube has fixed XY/yaw. Free-hand and carrying proposals use a 2 cm XY
patch; heights, object orientations and grasp diversity retain upstream settings.
Table-grasp states are generated from the settled free-hand bank, while near-goal
states use the original object-relative partial assemblies at the fixed goal.
"""

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import reset_states_cfg as resets
from . import rl_state_cfg as state

GOAL_XY = (0.45, 0.15)
START_X_RANGE = (0.35, 0.37)
START_Y_RANGE = (-0.01, 0.01)


def dataset_directory():
    return str(Path(os.environ.get("UWLAB_EASY_CUBE_DATASET_DIR", "Datasets/OmniResetCubeEasy")).resolve())


def _configure_cubes(cfg, variants):
    cfg.scene.insertive_object = variants["scene.insertive_object"]["cube"].copy()
    cfg.scene.receptive_object = variants["scene.receptive_object"]["cube"].copy()
    cfg.variants = {}
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset_directory()


def _configure_reset(cfg):
    _configure_cubes(cfg, resets.variants)
    goal = cfg.events.reset_receptive_object_pose.params["pose_range"]
    goal["x"] = (GOAL_XY[0], GOAL_XY[0])
    goal["y"] = (GOAL_XY[1], GOAL_XY[1])
    goal["yaw"] = (0.0, 0.0)
    # Preserve upstream support/bottom offsets and the robot's root jitter.
    if hasattr(cfg.events, "reset_insertive_object_pose"):
        start = cfg.events.reset_insertive_object_pose.params["pose_range"]
        start["x"] = START_X_RANGE
        start["y"] = START_Y_RANGE


@configclass
class CubeEasyObjectAnywhereEEAnywhereCfg(resets.ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_reset(self)


@configclass
class CubeEasyObjectRestingEEGraspedCfg(resets.ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_reset(self)


@configclass
class CubeEasyObjectAnywhereEEGraspedCfg(resets.ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_reset(self)


@configclass
class CubeEasyObjectPartiallyAssembledEEGraspedCfg(resets.ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_reset(self)


@configclass
class CubeEasyTrainCfg(state.Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_cubes(self, state.variants)
