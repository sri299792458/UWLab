# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured lab Vention layout constants for the UR5e right-arm migration.

These values describe the processed Vention v60 USD asset and the selected
right-arm mounting station. They are kept in one place so viewer scripts and
future task scene variants do not drift apart while the lab migration is being
validated.
"""

from __future__ import annotations

LAB_VENTION_USD_PATH = "/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd"

# Spawn transform for the converted Vention STEP/USD. Quaternion order is wxyz.
LAB_VENTION_POS = (1.793445, 0.340075, -0.030851)
LAB_VENTION_ROT = (0.5, 0.5, 0.5, 0.5)
LAB_VENTION_SCALE = (0.001, 0.001, 0.001)

# Simulated UWLab UR5e root pose aligned to the selected CAD right-arm station.
LAB_RIGHT_ARM_ROBOT_POS = (0.177660, 0.377695, 1.466000)
LAB_RIGHT_ARM_ROBOT_ROT = (0.7071068, 0.0, 0.7071068, 0.0)

# Top surface of the workspace tabletop after LAB_VENTION_* is applied.
LAB_TABLETOP_TOP_Z = 0.84235
LAB_TABLETOP_PANEL_REL_PATH = "visuals/vention_mat"

# Kinematic reachability probe over /lab_vention/visuals/vention_mat, generated
# by scripts_v2/tools/analyze_lab_tabletop_reachability.py.
LAB_RIGHT_ARM_REACHABLE_RECT_X = (-0.11708964407444, 0.6501603722572327)
LAB_RIGHT_ARM_REACHABLE_RECT_Y = (-0.4083046615123749, 0.28769537806510925)

# Slightly shrunken first reset range. This avoids sampling exactly on the IK
# grid boundary before we add the heavier collision validation pass.
LAB_RIGHT_ARM_RESET_X_RANGE = (-0.10, 0.625)
LAB_RIGHT_ARM_RESET_Y_RANGE = (-0.385, 0.265)


def lab_tabletop_panel_prim_path(parent_prim_path: str) -> str:
    """Return the absolute tabletop panel prim path under the spawned Vention parent."""

    return f"{parent_prim_path.rstrip('/')}/{LAB_TABLETOP_PANEL_REL_PATH}"
