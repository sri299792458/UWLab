"""Verify the physical mounting after the first actual reset, without changing physics."""

import json
import os
from pathlib import Path

import torch
from isaaclab.managers import EventTermCfg


EXPECTED_ROOT_QUATERNION = (0.5, 0.5, 0.5, 0.5)
EXPECTED_ROOT_POSITION = (0.177660, 0.377695, 1.466000)


def verify_corrected_mount(env, env_ids, report_dir: str):
    # This flag only avoids repeated file writes. It never changes simulator state or RNG.
    if getattr(env, "_corrected_mount_first_reset_verified", False):
        return
    count = env.num_envs if env_ids is None else len(env_ids)
    assert count == env.num_envs, "Expected the first reset to include every training environment"
    assert tuple(env.cfg.scene.robot.init_state.rot) == EXPECTED_ROOT_QUATERNION
    robot = env.scene["robot"]
    native = robot.root_physx_view.get_root_transforms()
    # Native PhysX transforms store quaternions as x,y,z,w.
    quaternion = native[:, (6, 3, 4, 5)]
    expected = torch.tensor(EXPECTED_ROOT_QUATERNION, dtype=quaternion.dtype, device=quaternion.device)
    dot_error = (1.0 - (quaternion * expected).sum(-1).abs()).abs()
    assert torch.isfinite(native).all()
    assert dot_error.max() < 2e-6, dot_error.max().item()
    position = native[:, :3] - env.scene.env_origins
    nominal = torch.tensor(EXPECTED_ROOT_POSITION, dtype=position.dtype, device=position.device)
    deviation = (position - nominal).abs().max(0).values
    # The existing reset generator jitters the base in all three position axes.
    assert deviation.max() <= 0.01005, deviation.tolist()
    report = {
        "status": "PASS", "pid": os.getpid(), "rank": int(os.environ.get("LOCAL_RANK", "0")),
        "num_envs": env.num_envs, "verified_after_first_reset": count,
        "configured_root_quaternion_wxyz": list(EXPECTED_ROOT_QUATERNION),
        "native_root_quaternion_first_wxyz": quaternion[0].tolist(),
        "max_quaternion_dot_error": dot_error.max().item(),
        "max_root_position_deviation_m": deviation.tolist(),
        "reset_dataset_dir": env.cfg.events.reset_from_reset_states.params["dataset_dir"],
        "no_physics_or_rng_changes": True,
    }
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"rank_{report['rank']}.json").write_text(json.dumps(report, indent=2) + "\n")
    env._corrected_mount_first_reset_verified = True
    print("CORRECTED_MOUNT_VERIFIED", json.dumps({k: report[k] for k in ("rank", "num_envs", "status", "native_root_quaternion_first_wxyz")}), flush=True)


def configure_corrected_mount_audit(cfg):
    cfg.events.verify_corrected_mount = EventTermCfg(
        func=verify_corrected_mount, mode="reset",
        params={"report_dir": os.environ["CORRECTED_MOUNT_REPORT_DIR"]},
    )
