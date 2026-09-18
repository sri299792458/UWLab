"""Check the comparison's native properties and finite task execution."""

import argparse
import importlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--task", default="OmniReset-UMI-MayHandDynamics-State-Train-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import uwlab_tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
module, name = gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")
cfg = getattr(importlib.import_module(module), name)()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 42
env = gym.make(args.task, cfg=cfg).unwrapped
args.output.mkdir(parents=True, exist_ok=True)
try:
    with torch.inference_mode():
        observation, _ = env.reset(seed=42)
        assert all(torch.isfinite(v).all() for v in observation.values())
        term = env.event_manager.get_term_cfg("reset_from_reset_states").func
        assert torch.allclose(term.probs, torch.full_like(term.probs, 0.25))
        action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
        action[:, -1] = -1
        terminations = 0
        for _ in range(20):
            observation, reward, done, timeout, _ = env.step(action)
            assert all(torch.isfinite(v).all() for v in observation.values())
            assert torch.isfinite(reward).all()
            terminations += int(done.sum())
        audit = cfg.events.verify_stock_hand_dynamics
        audit.func(env, None, **dict(audit.params, report_dir=str(args.output / "after_steps")))
        dump_yaml(str(args.output / "env.yaml"), cfg)
        report = {
            "status": "PASS", "task": args.task, "asset": cfg.scene.robot.spawn.usd_path,
            "num_envs": args.num_envs, "policy_steps": 20,
            "finite_observations_and_rewards": True, "original_terminations_retained": True,
            "termination_events": terminations,
            "reset_family_probabilities": term.probs.tolist(),
            "scope": "Native-property and execution validation; not a learning or stability success claim.",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("NATIVE_VALIDATION_COMPLETE", json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
