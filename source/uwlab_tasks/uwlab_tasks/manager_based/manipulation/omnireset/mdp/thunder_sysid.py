"""Use an explicit Thunder profile without changing shared robot metadata."""
from isaaclab.managers import ManagerTermBase

from .events import randomize_arm_from_sysid


class ThunderArmSysid(randomize_arm_from_sysid):
    def __init__(self, cfg, env):
        ManagerTermBase.__init__(self, cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.robot = env.scene[self.asset_cfg.name]
        self.joint_ids = self.robot.find_joints(cfg.params["joint_names"], preserve_order=True)[0]
        self.actuator_name = cfg.params["actuator_name"]
        for key in ("armature", "static_friction", "dynamic_ratio", "viscous_friction"):
            setattr(self, key, cfg.params["sysid"][key])
        self.scale_progress = cfg.params.get("initial_scale_progress", 0.0)

    def __call__(self, env, env_ids, asset_cfg, joint_names, actuator_name,
                 sysid, profile_sha256, scale_range=(0.8, 1.2), delay_range=(0, 1),
                 initial_scale_progress=0.0):
        return super().__call__(env, env_ids, asset_cfg, joint_names, actuator_name,
                                scale_range, delay_range, initial_scale_progress)
