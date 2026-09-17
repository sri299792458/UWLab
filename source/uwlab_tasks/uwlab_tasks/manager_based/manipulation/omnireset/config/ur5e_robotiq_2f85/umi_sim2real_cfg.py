"""Thunder/UMI system identification and Stage-2 state-policy tasks."""
import os

from isaaclab.utils import configclass
from uwlab_assets.robots.ur5e_robotiq_gripper.sysid_profile import load_profile

from ...mdp.thunder_sysid import ThunderArmSysid
from . import rl_state_cfg as state
from .sysid_cfg import SysidEnvCfg
from .umi_reset_cfg import UMI_ROBOT_USD
from .umi_training_cfg import _configure_training


@configclass
class ThunderSysidCfg(SysidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = UMI_ROBOT_USD
        # Fitting uses base-relative poses in free space, with compensated
        # gravity, as in UWLab. Lab table/cubes do not belong in this experiment.
        self.scene.ground = None
        self.sim.render_interval = self.decimation


def _configure_stage2(cfg, *, evaluation):
    # Apply the complete current UMI task after upstream selects its explicit
    # actuator robot, preserving the delay-capable actuator and Thunder geometry.
    _configure_training(cfg)
    profile, digest = load_profile(os.environ.get("THUNDER_SYSID_PROFILE", ""))
    cfg.thunder_profile_sha256 = digest
    event = cfg.events.randomize_arm_sysid
    event.func = ThunderArmSysid
    event.params.update(sysid=profile["sysid"], profile_sha256=digest,
                        delay_range=(0, 1), initial_scale_progress=1.0 if evaluation else 0.0)
    ctrl = profile["controller"]
    # Curriculum begins at the Stage-1 UMI gains. Its endpoint is the controller
    # used for the measured fit; it is not assumed to increase every gain.
    cfg.actions.arm.torque_limit = tuple(ctrl["torque_max"])
    if evaluation:
        cfg.actions.arm.motion_stiffness = tuple(ctrl["motion_stiffness"])
        cfg.actions.arm.motion_damping_ratio = tuple(ctrl["motion_damping_ratio"])
    else:
        cfg.events.randomize_osc_gains.params.update(
            terminal_kp=tuple(ctrl["motion_stiffness"]),
            terminal_damping_ratio=tuple(ctrl["motion_damping_ratio"]),
        )


@configclass
class UmiCubeFinetuneCfg(state.Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_stage2(self, evaluation=False)


@configclass
class UmiCubeFinetuneEvalCfg(state.Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_stage2(self, evaluation=True)
