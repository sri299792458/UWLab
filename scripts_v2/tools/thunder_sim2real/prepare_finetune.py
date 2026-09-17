"""Prepare a reviewable Stage-2 launch; this tool never starts training."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import torch

from records import HERE, REPO, profile_api, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--gpus", required=True, help="Physical GPU indices, e.g. 0 or 3,4,5,7")
    parser.add_argument("--num_envs", type=int, default=16384, help="Environments per GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("iterations and num_envs must be positive")
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error("gpus must be distinct comma-separated GPU indices")
    profile, profile_hash = profile_api().load_profile(args.profile)
    checkpoint = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in state:
        raise ValueError("Expected an RSL-RL Stage-1 checkpoint with model_state_dict")
    tensors = state["model_state_dict"].values()
    if any(torch.is_tensor(t) and not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Checkpoint contains non-finite policy weights")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.profile, output / "sysid_profile.json")
    command = [sys.executable]
    if len(gpus) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={len(gpus)}"]
    command += ["scripts/reinforcement_learning/rsl_rl/train.py", "--headless",
                "--task", "OmniReset-UMI-Defaults-State-Finetune-v0", "--resume_path", str(checkpoint),
                "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                "--seed", str(args.seed), "--logger", args.logger,
                "--run_name", "thunder_umi_stage2", "agent.experiment_name=thunder_umi_stage2"]
    if len(gpus) > 1:
        command.append("--distributed")
    if args.logger == "wandb":
        command += ["--log_project_name", "uwlab-lab-cube-stack"]
    pythonpath = ":".join(str(REPO / "source" / p) for p in ("uwlab", "uwlab_assets", "uwlab_tasks", "uwlab_rl"))
    environment = {"CUDA_VISIBLE_DEVICES": args.gpus, "THUNDER_SYSID_PROFILE": str(output / "sysid_profile.json"),
                   "PYTHONPATH": pythonpath, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1"}
    if os.environ.get("CONDA_PREFIX"):
        environment["CONDA_PREFIX"] = os.environ["CONDA_PREFIX"]
    # Record runtime sources so a prepared launch can be traced after later edits.
    sources = [p for base in (REPO / "source", HERE, REPO / "scripts/reinforcement_learning/rsl_rl")
               for p in base.rglob("*.py")]
    metadata = {"status": "prepared_not_launched", "command": command, "environment": environment,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "profile_sha256": profile_hash, "profile_validation_status": profile.get("validation_status"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "worktree": str(REPO), "num_envs_per_gpu": args.num_envs,
                "total_envs": args.num_envs*len(gpus), "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sources}}
    (output / "launch_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    script = "#!/usr/bin/env bash\nset -euo pipefail\n"
    script += f"cd {shlex.quote(str(REPO))}\n"
    script += "\n".join(f"export {key}={shlex.quote(value)}" for key, value in environment.items()) + "\n"
    script += "exec " + shlex.join(command) + "\n"
    (output / "launch.sh").write_text(script)
    (output / "launch.sh").chmod(0o755)
    print(f"Prepared {output / 'launch.sh'}; training has not started")


if __name__ == "__main__":
    main()
