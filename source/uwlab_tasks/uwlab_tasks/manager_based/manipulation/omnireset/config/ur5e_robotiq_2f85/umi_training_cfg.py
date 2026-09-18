"""State-policy training using the default-based UMI reset pipeline.

Hardware and explicit-controller gains are task-specific. Rewards, alignment
success, sampling, and PPO configuration remain inherited from upstream. The gains were
selected with nominal lift/carry/release trials at 120 Hz; policy validation and
training remain separate from that controller capability test.
"""
import os
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from . import rl_state_cfg as state
from .umi_reset_cfg import _configure_hardware, _configure_osc

TRAIN_DATASET_DIR = os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917') + '/49_clearance_and_placement/OmniReset'


def _configure_training(cfg):
    _configure_hardware(cfg, for_reset_generation=False)
    cfg.sim.render_interval = cfg.decimation
    # The controller computes Kd = 2 * sqrt(Kp) * motion_damping_ratio.
    # Configure the measured gains without changing its explicit torque law.
    _configure_osc(cfg)
    for term in vars(cfg.events).values():
        if isinstance(term, EventTermCfg) and "dataset_dir" in term.params:
            term.params["dataset_dir"] = TRAIN_DATASET_DIR


@configclass
class UmiCubeTrainCfg(state.Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_training(self)


@configclass
class UmiCubeEvalCfg(state.Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_training(self)
