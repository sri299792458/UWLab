"""Explicit stock D405 hand inertials on the existing centered 60 mm task."""
import os
from pathlib import Path

from isaaclab.utils import configclass

from . import thunder_60mm_cfg as previous
from .thunder_d405_cfg import _digest

ROBOT_SHA256 = "1dbff7240e9f0210daceca04cca9e824ebe2c0dd70a01bf204bcd1ed04e2b4d9"
HAND_SHA256 = "dbcc0acc7905eb9634904e84f2273e0c56f97e974998a39f71704ab22104ea1a"
METADATA_SHA256 = "625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242"


def configure_stock_model(cfg, *, standalone=False):
    directory = os.environ.get("UWLAB_THUNDER_STOCK_CORRECTED_ASSET_DIR")
    if not directory:
        raise ValueError("Set UWLAB_THUNDER_STOCK_CORRECTED_ASSET_DIR to the corrected stock D405 build")
    role = "hand" if standalone else "robot"
    path = Path(directory).expanduser().resolve() / role / f"{role}.usd"
    expected = HAND_SHA256 if standalone else ROBOT_SHA256
    if _digest(path) != expected or _digest(path.with_name("metadata.yaml")) != METADATA_SHA256:
        raise ValueError(f"Corrected stock D405 asset differs from the reviewed build: {path}")
    cfg.scene.robot.spawn.usd_path = str(path)
    # In particular, remove the standalone hand's inherited 0.5 kg override.
    cfg.scene.robot.spawn.mass_props = None


@configclass
class ThunderStockCorrectedGraspSamplingCfg(previous.ThunderD40560GraspSamplingCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self, standalone=True)


@configclass
class ThunderStockCorrectedObjectAnywhereEEAnywhereCfg(previous.ThunderD40560ObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self)


@configclass
class ThunderStockCorrectedObjectRestingEEGraspedCfg(previous.ThunderD40560ObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self)


@configclass
class ThunderStockCorrectedObjectAnywhereEEGraspedCfg(previous.ThunderD40560ObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self)


@configclass
class ThunderStockCorrectedObjectPartiallyAssembledEEGraspedCfg(previous.ThunderD40560ObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self)


@configclass
class ThunderStockCorrectedTrainCfg(previous.ThunderD40560TrainCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_stock_model(self)
