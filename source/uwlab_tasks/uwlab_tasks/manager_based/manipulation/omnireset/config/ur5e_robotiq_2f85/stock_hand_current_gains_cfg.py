"""Stock hand inertial properties with unchanged current UMI arm gains."""

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from .may_hand_dynamics_cfg import verify_stock_hand_dynamics
from .umi_training_cfg import UmiCubeTrainCfg
from .corrected_mount_audit import configure_corrected_mount_audit


@configclass
class UmiStockHandCurrentGainsTrainCfg(UmiCubeTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        # Inherit the selected UMI gains without a controller or gain override.
        # Reuse the exact UMI-geometry/stock-properties asset from the May-gain run.
        asset = Path(os.environ["STOCK_HAND_DYNAMICS_USD"]).resolve()
        self.scene.robot.spawn.usd_path = str(asset)
        self.events.verify_stock_hand_dynamics = EventTermCfg(
            func=verify_stock_hand_dynamics,
            mode="startup",
            params={
                "properties_path": str(asset.with_name("stock_hand_properties.json")),
                "report_dir": os.environ["STOCK_HAND_DYNAMICS_REPORT_DIR"],
                "expected_kp_values": (500.0, 500.0, 500.0, 60.0, 60.0, 60.0),
                "expected_kd_values": (160.0, 160.0, 160.0, 0.1, 0.1, 0.1),
            },
        )
        configure_corrected_mount_audit(self)
