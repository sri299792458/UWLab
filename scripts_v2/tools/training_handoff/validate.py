"""Native smoke check with relocated assets; no policy optimization or robot connection."""

import argparse
import importlib
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = env = None
try:
    app = AppLauncher(args).app
    import gymnasium as gym
    import torch
    import isaaclab
    import uwlab_tasks  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.corrected_mount_audit import configure_corrected_mount_audit

    module, name = gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module), name)()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    configure_corrected_mount_audit(cfg)
    env = gym.make(args.task, cfg=cfg).unwrapped
    with torch.inference_mode():
        observations, _ = env.reset(seed=42)
        assert all(torch.isfinite(v).all() for v in observations.values())
        reset = env.event_manager.get_term_cfg("reset_from_reset_states").func
        assert torch.allclose(reset.probs, torch.full_like(reset.probs, 0.25))
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        actions[:, -1] = -1
        terminations = 0
        for _ in range(20):
            observations, rewards, done, timeout, _ = env.step(actions)
            assert all(torch.isfinite(v).all() for v in observations.values())
            assert torch.isfinite(rewards).all()
            terminations += int(done.sum())
        stock_audit = getattr(cfg.events, "verify_stock_hand_dynamics", None)
        if stock_audit:
            stock_audit.func(env, None, **dict(stock_audit.params, report_dir=str(args.output / "after_steps")))
        robot = env.scene["robot"]
        arm = env.action_manager.get_term("arm")
        assert torch.isfinite(robot.data.default_mass).all() and torch.isfinite(robot.data.default_inertia).all()
        dump_yaml(str(args.output / "env.yaml"), cfg)
        report = {
            "status": "PASS", "task": args.task, "num_envs": env.num_envs, "policy_steps": 20,
            "asset": cfg.scene.robot.spawn.usd_path, "reset_dataset_dir": cfg.events.reset_from_reset_states.params["dataset_dir"],
            "isaaclab_module": isaaclab.__file__, "uwlab_tasks_module": uwlab_tasks.__file__,
            "body_names": robot.body_names, "default_masses_kg": robot.data.default_mass[0].cpu().tolist(),
            "default_inertias_kg_m2": robot.data.default_inertia[0].cpu().tolist(),
            "kp": arm._kp[0].cpu().tolist(), "kd": arm._kd[0].cpu().tolist(),
            "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
            "finite_observations_and_rewards": True, "termination_events": terminations,
            "stock_property_audit": bool(stock_audit), "original_terminations_retained": True,
            "scope": "Native startup and finite execution; not learning success or an MSI hardware validation.",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TRAINING_HANDOFF_VALIDATED", json.dumps(report), flush=True)
except BaseException as error:
    traceback.print_exc()
    (args.output / "failure.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n")
    raise
finally:
    if env is not None:
        env.close()
    if app is not None:
        app.close()
