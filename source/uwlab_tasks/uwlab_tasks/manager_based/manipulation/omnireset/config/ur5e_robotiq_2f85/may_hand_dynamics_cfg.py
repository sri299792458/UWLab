"""Isolated UMI experiment using stock hand inertial properties and May gains."""

import json
import os
from pathlib import Path

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from .umi_training_cfg import UmiCubeTrainCfg


def verify_stock_hand_dynamics(
    env, env_ids, properties_path: str, report_dir: str,
    expected_kp_values=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    expected_kd_values=(84.8528137423857, 84.8528137423857, 84.8528137423857,
                        3.4641016151377544, 3.4641016151377544, 3.4641016151377544),
):
    """Read native properties after startup randomization; never alter state or RNG."""
    robot = env.scene["robot"]
    view = robot.root_physx_view
    source = json.loads(Path(properties_path).read_text())
    requested = source["body_properties"]
    reference = json.loads(Path(properties_path).with_name("reference_umi_properties.json").read_text())
    masses = view.get_masses().cpu()
    inertias = view.get_inertias().cpu()
    coms = view.get_coms().cpu()
    default_masses = robot.data.default_mass.cpu()
    default_inertias = robot.data.default_inertia.cpu()
    errors = {}
    for index, name in enumerate(robot.body_names):
        if name in requested:
            expected = requested[name]
        else:
            old_index = reference["body_names"].index(name)
            expected = {
                "mass": reference["masses_kg"][old_index],
                "inertia": reference["inertias_kg_m2"][old_index],
                "com": reference["com_poses"][old_index][:3],
            }
        expected_i = torch.tensor(expected["inertia"], dtype=default_inertias.dtype)
        expected_c = torch.tensor(expected["com"], dtype=coms.dtype)
        mass_error = (default_masses[:, index] - expected["mass"]).abs().max().item()
        inertia_error = (default_inertias[:, index] - expected_i).abs().max().item()
        com_error = (coms[:, index, :3] - expected_c).abs().max().item()
        assert mass_error < max(2e-6, abs(expected["mass"]) * 2e-6), (name, mass_error)
        assert inertia_error < 2e-8, (name, inertia_error)
        assert com_error < 2e-6, (name, com_error)
        errors[name] = {"mass_kg": mass_error, "inertia_kg_m2": inertia_error, "com_m": com_error}

    scale = masses / default_masses
    assert torch.isfinite(scale).all()
    assert scale.min() >= 0.7 - 1e-6 and scale.max() <= 1.3 + 1e-6
    inertia_scale_error = (inertias - default_inertias * scale[..., None]).abs().max().item()
    assert inertia_scale_error < 2e-7, inertia_scale_error
    arm = env.action_manager.get_term("arm")
    expected_kp = torch.tensor(expected_kp_values, device=arm._kp.device)
    expected_kd = torch.tensor(expected_kd_values, device=arm._kd.device)
    assert torch.allclose(arm._kp, expected_kp.expand_as(arm._kp))
    assert torch.allclose(arm._kd, expected_kd.expand_as(arm._kd), rtol=1e-6, atol=1e-6)
    assert abs(env.physics_dt - 1 / 120) < 1e-12
    assert abs(env.step_dt - 0.1) < 1e-12
    report = {
        "status": "PASS",
        "pid": os.getpid(),
        "rank": int(os.environ.get("LOCAL_RANK", "0")),
        "num_envs": env.num_envs,
        "seed": env.cfg.seed,
        "device": str(env.device),
        "asset": robot.cfg.spawn.usd_path,
        "body_names": robot.body_names,
        "default_masses_kg": default_masses[0].tolist(),
        "default_inertias_kg_m2": default_inertias[0].tolist(),
        "com_poses": coms[0].tolist(),
        "randomized_mass_scale_range": [scale.min().item(), scale.max().item()],
        "inertia_randomization_max_error": inertia_scale_error,
        "expected_property_errors": errors,
        "kp": arm._kp[0].tolist(),
        "kd": arm._kd[0].tolist(),
        "action_scale": arm._scale.tolist(),
        "physics_dt": env.physics_dt,
        "policy_dt": env.step_dt,
        "read_only_check": True,
    }
    output = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank_{report['rank']}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("MAY_HAND_DYNAMICS_VERIFIED", json.dumps({k: report[k] for k in ("rank", "num_envs", "seed", "status")}), flush=True)


@configclass
class UmiMayHandDynamicsTrainCfg(UmiCubeTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        asset = Path(os.environ["MAY_HAND_DYNAMICS_USD"]).resolve()
        self.scene.robot.spawn.usd_path = str(asset)
        self.actions.arm.motion_stiffness = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
        self.actions.arm.motion_damping_ratio = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
        self.events.verify_stock_hand_dynamics = EventTermCfg(
            func=verify_stock_hand_dynamics,
            mode="startup",
            params={
                "properties_path": str(asset.with_name("stock_hand_properties.json")),
                "report_dir": os.environ["MAY_HAND_DYNAMICS_REPORT_DIR"],
            },
        )
