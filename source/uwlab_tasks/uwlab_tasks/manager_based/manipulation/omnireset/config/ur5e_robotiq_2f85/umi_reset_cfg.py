"""UMI/AprilCube60 hardware inputs applied to the upstream reset pipeline.

Upstream height/orientation proposals and contact settings are retained.
Cube placements use full-table map coverage and 50 mm edge clearance. Arm IK
is seeded from the atlas; final states use its column and collision checks.
"""
import os

from copy import deepcopy
from math import sqrt

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...mdp.umi_map_reset import (MapEndEffectorAnywhere, MapEndEffectorGrasped,
                                 MapResetSuccess, MapCubePlacement)
from . import grasp_sampling_cfg as grasp
from . import partial_assemblies_cfg as partial
from . import reset_states_cfg as resets

UMI_ROBOT_USD = os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets') + '/thunder_d405_umi_rigid_asset/ur5e_robotiq_d405_umi_rigid_thunder.usd'
UMI_HAND_USD = os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets') + '/thunder_d405_umi_gripper_asset/robotiq_d405_umi_rigid.usd'
CUBE_DIR = os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets') + '/aprilcube_60mm_rounded'
INSERTIVE_USD = f"{CUBE_DIR}/InsertiveAprilCube60/aprilcube_60.usd"
RECEPTIVE_USD = f"{CUBE_DIR}/ReceptiveAprilCube60/aprilcube_60.usd"
DATASET_DIR = os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911') + '/49_clearance_and_placement/OmniReset'
TABLE_USD = os.environ.get('UWLAB_ASSET_ROOT', '/data/kanth042/converted_assets') + '/lab_vention_asset_v60/lab_vention.usd'
TABLE_POS = (1.793445, 0.340075, -0.030851)
TABLE_ROT = (0.5, 0.5, 0.5, 0.5)
TABLE_Z = 0.84235
ROBOT_POS = (0.177660, 0.377695, 1.466000)
ROBOT_ROT = (0.7071068, 0.0, 0.7071068, 0.0)
WORKSPACE_X = (-0.11708964407444, 0.6501603722572327)
WORKSPACE_Y = (-0.4083046615123749, 0.28769537806510925)
CUBE_MASS_RANGE = (0.02, 0.20)
UMI_OSC_KP = (500.0, 500.0, 500.0, 60.0, 60.0, 60.0)
UMI_OSC_KD = (160.0, 160.0, 160.0, 0.1, 0.1, 0.1)


def _configure_osc(cfg):
    cfg.actions.arm.motion_stiffness = UMI_OSC_KP
    cfg.actions.arm.motion_damping_ratio = tuple(
        kd / (2.0 * sqrt(kp)) for kp, kd in zip(UMI_OSC_KP, UMI_OSC_KD)
    )


def _configure_map_resets(cfg):
    _configure_osc(cfg)
    for name in ('reset_receptive_object_pose', 'reset_insertive_object_pose'):
        if hasattr(cfg.events, name):
            term = getattr(cfg.events, name)
            term.func = MapCubePlacement
            # Uniform table proposals; the rotated footprint and position map
            # determine which placements survive. Do not snap cube positions.
            term.params['pose_range']['x'] = (-0.834839610138, 0.650160389862)
            term.params['pose_range']['y'] = (-0.432304636132, 0.287695363868)
    if hasattr(cfg.events, "reset_end_effector_pose"):
        cfg.events.reset_end_effector_pose.func = MapEndEffectorAnywhere
        # XYZ comes from the map's tool-point domain; retain stock hand angles.
        ranges = cfg.events.reset_end_effector_pose.params["pose_range_b"]
        cfg.events.reset_end_effector_pose.params["pose_range_b"] = {
            key: ranges[key] for key in ("roll", "pitch", "yaw")
        }
    if hasattr(cfg.events, "reset_end_effector_pose_from_grasp_dataset"):
        cfg.events.reset_end_effector_pose_from_grasp_dataset.func = MapEndEffectorGrasped
    cfg.terminations.success.func = MapResetSuccess


@configclass
class UmiGraspSamplingCfg(grasp.Robotiq2f85GraspSamplingCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = UMI_HAND_USD
        self.scene.robot.spawn.mass_props = None
        self.scene.object = deepcopy(grasp.variants["scene.object"]["cube"])
        self.scene.object.spawn.usd_path = INSERTIVE_USD
        # Keep this cube's authored 100 g mass and matching inertia. The stock
        # cube's configured 1 g override is ineffective without its MassAPI;
        # forcing a true 1 g reproduced zero accepted grasps with this UMI hand.
        self.scene.object.spawn.mass_props = None
        # Select hardware here; named Hydra variants would replace child overrides.
        self.variants = {}


@configclass
class UmiPartialAssembliesCfg(partial.PartialAssembliesCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, usd in (("insertive_object", INSERTIVE_USD), ("receptive_object", RECEPTIVE_USD)):
            cube = deepcopy(partial.variants[f"scene.{name}"]["cube"])
            cube.spawn.usd_path = usd
            # Use the selected cube assets’ authored 100 g mass and inertia.
            cube.spawn.mass_props = None
            setattr(self.scene, name, cube)
        self.variants = {}


def _configure_hardware(cfg, *, for_reset_generation: bool):
    cfg.scene.robot.spawn.usd_path = UMI_ROBOT_USD
    cfg.scene.robot.init_state.pos = ROBOT_POS
    cfg.scene.robot.init_state.rot = ROBOT_ROT
    cfg.scene.table.spawn.usd_path = TABLE_USD
    cfg.scene.table.spawn.scale = (0.001, 0.001, 0.001)
    cfg.scene.table.init_state.pos = TABLE_POS
    cfg.scene.table.init_state.rot = TABLE_ROT
    cfg.scene.ground.init_state.pos = (0.0, 0.0, 0.0)
    # The lab CAD already contains the physical mounting station.
    cfg.scene.ur5_metal_support = None
    if hasattr(cfg.events, "reset_robot_pose"):
        root_reset = cfg.events.reset_robot_pose.params
        root_reset["asset_cfgs"] = {"robot": SceneEntityCfg("robot")}
        root_reset["pose_range"]["y"] = (-0.01, 0.01)

    for name, usd in (("insertive_object", INSERTIVE_USD), ("receptive_object", RECEPTIVE_USD)):
        cube = deepcopy(resets.variants[f"scene.{name}"]["cube"])
        cube.spawn.usd_path = usd
        cube.spawn.mass_props = None
        # Preserve upstream motion modes: movable upper cube, fixed lower cube.
        # The lower cube can be repositioned by resets but cannot be pushed.
        cube.spawn.rigid_props.disable_gravity = False
        setattr(cfg.scene, name, cube)
        if for_reset_generation:
            # Keep authored mass/inertia throughout offline generation.
            setattr(cfg.events, f"randomize_{name}_mass", None)
        else:
            setattr(cfg.events, f"randomize_{name}_mass", EventTermCfg(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "mass_distribution_params": CUBE_MASS_RANGE,
                    "operation": "abs", "distribution": "uniform", "recompute_inertia": True,
                },
            ))
    cfg.variants = {}

    for name in ("reset_receptive_object_pose", "reset_insertive_object_pose"):
        if not hasattr(cfg.events, name):
            continue
        params = getattr(cfg.events, name).params
        params.pop("offset_asset_cfg", None)
        ranges = params["pose_range"]
        ranges["x"], ranges["y"] = WORKSPACE_X, WORKSPACE_Y
        ranges["z"] = tuple(TABLE_Z + z for z in ranges["z"])

    if hasattr(cfg.events, "reset_end_effector_pose"):
        # Upstream implements these as WORLD-axis displacements from root
        # position, despite the parameter name pose_range_b. Orientations too
        # are world-frame Euler angles; retain its original downward range.
        ranges = cfg.events.reset_end_effector_pose.params["pose_range_b"]
        ranges["x"] = tuple(x - ROBOT_POS[0] for x in WORKSPACE_X)
        ranges["y"] = tuple(y - ROBOT_POS[1] for y in WORKSPACE_Y)
        ranges["z"] = tuple(TABLE_Z + z - ROBOT_POS[2] for z in ranges["z"])

    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = DATASET_DIR
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success.params["pos_z_threshold"] += TABLE_Z


@configclass
class UmiObjectAnywhereEEAnywhereCfg(resets.ObjectAnywhereEEAnywhereResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hardware(self, for_reset_generation=True)
        _configure_map_resets(self)


@configclass
class UmiObjectRestingEEGraspedCfg(resets.ObjectRestingEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hardware(self, for_reset_generation=True)
        _configure_map_resets(self)


@configclass
class UmiObjectAnywhereEEGraspedCfg(resets.ObjectAnywhereEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hardware(self, for_reset_generation=True)
        _configure_map_resets(self)


@configclass
class UmiObjectPartiallyAssembledEEGraspedCfg(resets.ObjectPartiallyAssembledEEGraspedResetStatesCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_hardware(self, for_reset_generation=True)
        _configure_map_resets(self)
