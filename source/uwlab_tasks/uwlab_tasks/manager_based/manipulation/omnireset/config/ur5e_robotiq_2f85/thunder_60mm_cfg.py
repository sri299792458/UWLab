# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thunder D405 stock-hand task with rounded 60 mm AprilCubes.

Use the assets' authored 100 g mass for generation and the inherited training
mass randomization. All data is separate from the 40 mm task. Object metadata
supplies the 30 mm bottom/assembly offsets without changing the approved XY.
"""

import os
from pathlib import Path

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import grasp_sampling_cfg as grasps
from . import partial_assemblies_cfg as partial
from . import thunder_d405_cfg as d405


ASSET_SHA256 = "10c1083be6deb752421ad36bea130d5ba131a69dc4a49a9a6b14abf603f8dfde"
METADATA_SHA256 = {
    "InsertiveAprilCube60": "a8321bd212e6fb87f995f00c4234c6bf78da79db3784ebce7dff1380caaea29d",
    "ReceptiveAprilCube60": "b9e45b8171f0ea7bff2e498ba26ebcdac6af33fffa60763336b8ad5f3faf83dc",
}


def _configure_object(obj, role):
    directory = os.environ.get("UWLAB_THUNDER_60MM_ASSET_DIR")
    if not directory:
        raise ValueError("Set UWLAB_THUNDER_60MM_ASSET_DIR to the rounded AprilCube60 asset directory")
    directory = Path(directory).expanduser().resolve() / role
    usd = directory / "aprilcube_60.usd"
    metadata = directory / "metadata.yaml"
    if d405._digest(usd) != ASSET_SHA256 or d405._digest(metadata) != METADATA_SHA256[role]:
        raise ValueError(f"60 mm cube differs from the reviewed asset: {directory}")
    obj.spawn.usd_path = str(usd)
    # Unlike the old 40 mm USD, this USD has MassAPI. Inherited 1 g / 500 g
    # overrides would therefore take effect; retain the authored 100 g instead.
    obj.spawn.mass_props = None


def _configure_pair(cfg):
    _configure_object(cfg.scene.insertive_object, "InsertiveAprilCube60")
    _configure_object(cfg.scene.receptive_object, "ReceptiveAprilCube60")
    cfg.variants = {}
    dataset = str(Path(os.environ.get(
        "UWLAB_THUNDER_60MM_DATASET_DIR", "Datasets/OmniResetCubeEasyThunderD40560"
    )).expanduser().resolve())
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = dataset


@configclass
class ThunderD40560GraspSamplingCfg(grasps.Robotiq2f85GraspSamplingCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = grasps.variants["scene.object"]["cube"].copy()
        _configure_object(self.scene.object, "InsertiveAprilCube60")
        self.variants = {}


@configclass
class ThunderD40560PartialAssembliesCfg(partial.PartialAssembliesCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.insertive_object = partial.variants["scene.insertive_object"]["cube"].copy()
        self.scene.receptive_object = partial.variants["scene.receptive_object"]["cube"].copy()
        _configure_pair(self)


@configclass
class ThunderD40560ObjectAnywhereEEAnywhereCfg(d405.ThunderD405ObjectAnywhereEEAnywhereCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_pair(self)


@configclass
class ThunderD40560ObjectRestingEEGraspedCfg(d405.ThunderD405ObjectRestingEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_pair(self)


@configclass
class ThunderD40560ObjectAnywhereEEGraspedCfg(d405.ThunderD405ObjectAnywhereEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_pair(self)


@configclass
class ThunderD40560ObjectPartiallyAssembledEEGraspedCfg(d405.ThunderD405ObjectPartiallyAssembledEEGraspedCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_pair(self)


@configclass
class ThunderD40560TrainCfg(d405.ThunderD405TrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_pair(self)
