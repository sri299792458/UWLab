# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thunder easy task centered on the right half of the measured tabletop.

Translate the two-cube workspace together, retaining their relative spacing,
proposal widths, heights, rotations, grasp library, and native reset solver.
The calibrated D415 arm with stock fingers is inherited from ThunderLab.
"""

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import thunder_lab_cfg as lab


# Approved from the native lab render: midpoint of the two nominal cube centers
# at the center of the table's right half. These are world coordinates in meters.
WORKSPACE_SHIFT_XY = (-0.1260896, -0.1473046)
GOAL_XY = (0.3239104, 0.0026954)
START_X_RANGE = (0.2239104, 0.2439104)
START_Y_RANGE = (-0.1573046, -0.1373046)


def _configure_centered_workspace(cfg):
    if hasattr(cfg.events, "reset_receptive_object_pose"):
        goal = cfg.events.reset_receptive_object_pose.params["pose_range"]
        goal["x"] = (GOAL_XY[0], GOAL_XY[0])
        goal["y"] = (GOAL_XY[1], GOAL_XY[1])
    if hasattr(cfg.events, "reset_insertive_object_pose"):
        start = cfg.events.reset_insertive_object_pose.params["pose_range"]
        start["x"] = START_X_RANGE
        start["y"] = START_Y_RANGE
    if hasattr(cfg.events, "reset_end_effector_pose"):
        # This sampler adds translations to the world root position. Move its
        # targets with the cubes, keeping the same range widths and orientations.
        targets = cfg.events.reset_end_effector_pose.params["pose_range_b"]
        for axis, shift in zip(("x", "y"), WORKSPACE_SHIFT_XY):
            targets[axis] = tuple(value + shift for value in targets[axis])

    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_CENTERED_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderCentered"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset


@configclass
class ThunderCenteredObjectAnywhereEEAnywhereCfg(lab.ThunderLabObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_centered_workspace(self)


@configclass
class ThunderCenteredObjectRestingEEGraspedCfg(lab.ThunderLabObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_centered_workspace(self)


@configclass
class ThunderCenteredObjectAnywhereEEGraspedCfg(lab.ThunderLabObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_centered_workspace(self)


@configclass
class ThunderCenteredObjectPartiallyAssembledEEGraspedCfg(
    lab.ThunderLabObjectPartiallyAssembledEEGraspedCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_centered_workspace(self)


@configclass
class ThunderCenteredTrainCfg(lab.ThunderLabTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_centered_workspace(self)
