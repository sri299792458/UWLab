"""Prepare a portable run; execution is a separate launch.sh invocation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONDITIONS = {
    "stock-may": ("OmniReset-UMI-MayHandDynamics-State-Train-v0", "MAY_HAND_DYNAMICS"),
    "stock-current": ("OmniReset-UMI-StockHandCurrentGains-State-Train-v0", "STOCK_HAND_DYNAMICS"),
    "umi-current": ("OmniReset-UMI-Defaults-State-Train-v0", None),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_launcher(path, command, environment, cwd, log, exit_file):
    lines = ["#!/usr/bin/env bash", "set -uo pipefail", "cd " + shlex.quote(str(cwd))]
    lines += ["export " + key + "=" + shlex.quote(str(value)) for key, value in environment.items()]
    lines += [shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1",
              "training_status=$?", 'printf "%s\\n" "$training_status" > ' + shlex.quote(str(exit_file)),
              'exit "$training_status"']
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    subprocess.run(["bash", "-n", str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new run directory")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Simulator environment's Python")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--envs-per-rank", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--name", help="Run label; a unique W&B run ID is generated separately")
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    parser.add_argument("--offline", action="store_true", help="Save W&B locally for later wandb sync")
    parser.add_argument("--resume-path", type=Path, help="Explicit checkpoint; otherwise always starts fresh")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only: allow incomplete source snapshot")
    args = parser.parse_args()
    assert args.nproc > 0 and args.envs_per_rank > 0 and args.max_iterations > 0
    bundle = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    python = args.python.expanduser().absolute()
    assert python.is_file(), python
    assert not output.exists(), f"Refusing to overwrite {output}"
    installation = json.loads((bundle / "training_installation.json").read_text())
    assert installation["status"] == "PASS"
    release = json.loads((HERE / "release_manifest.json").read_text())
    assert installation["release"]["archive_sha256"] == release["archive_sha256"]
    refs = json.loads((HERE / "reference_runs.json").read_text())
    reference = refs["runs"]["may_gains"]
    dataset = bundle / "data/49_clearance_and_placement/OmniReset"
    bank = dataset / "Resets/InsertiveAprilCube60__ReceptiveAprilCube60"
    hashes = {name: sha(bank / f"resets_{name}.pt") for name in reference["dataset_sha256"]}
    assert hashes == reference["dataset_sha256"], "Reset bank differs from the corrected reference"
    atlas = json.loads((bundle / "data/49_clearance_and_placement/atlas/config.json").read_text())
    assert atlas["robot_base_quaternion_world_wxyz"] == [0.5] * 4
    stock_manifest = json.loads((HERE / "stock_asset_manifest.json").read_text())
    original_asset = bundle / stock_manifest["base_usd_relative"]
    assert sha(original_asset) == stock_manifest["base_usd_sha256"]
    stock = bundle / stock_manifest["installed_directory"]
    for name, entry in stock_manifest["files"].items():
        assert sha(stock / name) == entry["sha256"], name
    task, prefix = CONDITIONS[args.condition]
    asset = stock / "umi_geometry_stock_hand_dynamics.usda" if prefix else original_asset
    run_id = uuid.uuid4().hex[:8]
    run_name = args.name or f"msi_{args.condition.replace('-', '_')}_seed{args.seed}_{run_id}"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", run_name), "Use letters, digits, dot, dash, underscore in --name"
    checkpoint = args.resume_path.expanduser().resolve() if args.resume_path else None
    if checkpoint:
        assert checkpoint.is_file(), checkpoint
    paths = [REPO / "source" / module for module in ("uwlab", "uwlab_assets", "uwlab_tasks", "uwlab_rl")]
    paths += [bundle / "sources/IsaacLab/source" / module for module in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")]
    environment = {
        "PYTHONPATH": ":".join(map(str, paths)),
        "UWLAB_DATA_ROOT": str(bundle / "data"), "UWLAB_ASSET_ROOT": str(bundle / "assets"),
        "UWLAB_CACHE_ASSETS_ROOT": str(bundle / "cache_assets"), "ISAACLAB_PATH": str(bundle / "sources/IsaacLab"),
        "UWLAB_TRAINING_BUNDLE": str(bundle), "UWLAB_STOCK_HAND_USD": str(stock / "umi_geometry_stock_hand_dynamics.usda"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HYDRA_FULL_ERROR": "1",
        "WANDB_ENTITY": refs["wandb_entity"], "WANDB_PROJECT": refs["wandb_project"],
        "WANDB_RUN_ID": run_id, "WANDB_RESUME": "never", "WANDB_MODE": "offline" if args.offline else "online",
        "WANDB_DIR": str(output), "WANDB_TAGS": f"msi,corrected-physical-mount,corrected-bank,{args.condition},seed{args.seed}",
        "CORRECTED_MOUNT_REPORT_DIR": str(output / "mount_training"),
    }
    if prefix:
        environment[prefix + "_USD"] = str(asset)
        environment[prefix + "_REPORT_DIR"] = str(output / "native_training")
    command = [str(python), "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={args.nproc}",
               str(REPO / "scripts/reinforcement_learning/rsl_rl/train.py"), "--distributed", "--headless",
               "--task", task, "--num_envs", str(args.envs_per_rank), "--max_iterations", str(args.max_iterations),
               "--logger", args.logger, "--log_project_name", refs["wandb_project"], "--seed", str(args.seed), "--run_name", run_name,
               "agent.experiment_name=umi_clearance_aprilcube60_r3", "env.events.reset_from_reset_states.params.dataset_dir=" + str(dataset)]
    if checkpoint:
        command += ["--resume_path", str(checkpoint)]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    assert not dirty or args.allow_dirty, "Commit source changes first, or explicitly use --allow-dirty for a development check"
    output.mkdir(parents=True)
    snapshot = output / "git_source_snapshot.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "--output", str(snapshot), commit], cwd=REPO, check=True)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO)
    (output / "source_changes.patch").write_bytes(patch)
    metadata = {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(), "condition": args.condition,
        "task": task, "worktree": str(REPO), "git_commit": commit, "git_status": dirty,
        "source_snapshot_sha256": sha(snapshot), "tracked_source_patch_sha256": sha(output / "source_changes.patch"),
        "untracked_source_note": "Untracked files are not included in the source snapshot; commit experiment changes before launch.",
        "bundle_archive_sha256": release["archive_sha256"], "dataset_dir": str(dataset), "dataset_sha256": hashes,
        "dataset_counts": reference["dataset_counts"], "corrected_root_quaternion_wxyz": [0.5] * 4,
        "asset": str(asset), "asset_sha256": sha(asset), "base_umi_asset_sha256": sha(original_asset),
        "seed": args.seed, "rank_seeds": list(range(args.seed, args.seed + args.nproc)),
        "num_envs_per_rank": args.envs_per_rank, "num_workers": args.nproc, "total_environments": args.nproc * args.envs_per_rank,
        "samples_per_update": 32 * args.nproc * args.envs_per_rank, "physics_hz": 120, "policy_hz": 10,
        "max_iterations_argument": args.max_iterations, "resume_checkpoint": str(checkpoint) if checkpoint else None,
        "resume_checkpoint_sha256": sha(checkpoint) if checkpoint else None, "fresh_policy": checkpoint is None,
        "wandb_entity": refs["wandb_entity"], "wandb_project": refs["wandb_project"], "wandb_run_id": run_id,
        "wandb_url": f"https://wandb.ai/{refs['wandb_entity']}/{refs['wandb_project']}/runs/{run_id}" if args.logger == "wandb" else None,
        "logger": args.logger, "run_name": run_name, "launch_environment": environment, "launch_command": command,
        "gpu_assignment": "Inherited CUDA_VISIBLE_DEVICES from the allocation; no physical GPU indices are hard-coded.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_launcher(output / "launch.sh", command, environment, output, output / "train.log", output / "train.exit")
    validation_environment = environment.copy()
    validation_environment["CORRECTED_MOUNT_REPORT_DIR"] = str(output / "validation/mount")
    if prefix:
        validation_environment[prefix + "_REPORT_DIR"] = str(output / "validation/startup")
    validation_command = [str(python), str(HERE / "validate.py"), "--task", task,
                          "--output", str(output / "validation"), "--headless", "--device", "cuda:0"]
    write_launcher(output / "validate.sh", validation_command, validation_environment, output,
                   output / "validation.log", output / "validation.exit")
    print(json.dumps({key: metadata[key] for key in ("condition", "total_environments", "samples_per_update", "wandb_url")}, indent=2))
    print("Prepared only. Run validate.sh and check validation/report.json before running launch.sh.")


if __name__ == "__main__":
    main()
