"""Prepare a fresh four-GPU comparison from the recorded seed-42 baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    args = parser.parse_args()
    worktree = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    assert not (output / "run_metadata.json").exists(), "Do not overwrite an existing run identity"
    assert json.loads(args.native_validation.read_text())["status"] == "PASS"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree), "Commit experiment changes first"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=worktree, text=True).strip()
    reference = json.loads(args.reference_metadata.read_text())
    assert reference["seed"] == 42
    for asset in reference["asset_files"]:
        assert sha(asset["path"]) == asset["sha256"], asset["path"]
    bank = Path(reference["dataset_dir"]) / "Resets/InsertiveAprilCube60__ReceptiveAprilCube60"
    for family, digest in reference["dataset_sha256"].items():
        assert sha(bank / f"resets_{family}.pt") == digest, family
    for item in reference["dataset_inputs"].values():
        assert sha(item["path"]) == item["sha256"]
    run_id = uuid.uuid4().hex[:8]
    run_name = "umi_may_hand_dynamics_seed42_4gpu_env16384"
    environment = reference["launch_environment"].copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="3,4,5,7",
        WANDB_RUN_ID=run_id,
        WANDB_RESUME="never",
        WANDB_TAGS=environment["WANDB_TAGS"] + ",may-gains,stock-hand-inertials,seed42,dynamics-comparison",
        PYTHONPATH=":".join(str(worktree / "source" / name) for name in ("uwlab", "uwlab_assets", "uwlab_tasks", "uwlab_rl")),
        MAY_HAND_DYNAMICS_USD=str(args.asset.resolve()),
        MAY_HAND_DYNAMICS_REPORT_DIR=str(output / "native_training"),
    )
    command = reference["launch_command"].copy()
    command[command.index("scripts/reinforcement_learning/rsl_rl/train.py")] = str(worktree / "scripts/reinforcement_learning/rsl_rl/train.py")
    command[command.index("--task") + 1] = "OmniReset-UMI-MayHandDynamics-State-Train-v0"
    command[command.index("--run_name") + 1] = run_name
    lines = ["#!/bin/bash", "set -u", "cd " + shlex.quote(str(worktree))]
    lines += ["export " + key + "=" + shlex.quote(value) for key, value in environment.items()]
    lines += [
        shlex.join(command) + " > " + shlex.quote(str(output / "train.log")) + " 2>&1",
        "training_status=$?",
        'printf "%s\\n" "$training_status" > ' + shlex.quote(str(output / "train.exit")),
        'exit "$training_status"',
    ]
    launch = output / "launch.sh"
    launch.write_text("\n".join(lines) + "\n")
    launch.chmod(0o755)
    subprocess.run(["bash", "-n", str(launch)], check=True)
    archive = output / "git_source_snapshot.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "--output", str(archive), commit], cwd=worktree, check=True)
    metadata = {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(worktree), "git_branch": branch, "git_commit": commit,
        "baseline_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=worktree, text=True).strip(),
        "git_status_clean": True, "code_snapshot": str(archive), "code_snapshot_sha256": sha(archive),
        "reference_metadata": str(args.reference_metadata.resolve()),
        "reference_run_id": reference["wandb_run_id"], "reference_training_log_dir": reference["training_log_dir"],
        "comparison_asset": str(args.asset.resolve()), "comparison_asset_sha256": sha(args.asset),
        "asset_build_report": str(args.asset.with_name("build_report.json").resolve()),
        "asset_files": reference["asset_files"],
        "dataset_dir": reference["dataset_dir"], "dataset_sha256": reference["dataset_sha256"],
        "dataset_counts": reference["dataset_counts"], "dataset_inputs": reference["dataset_inputs"],
        "launch_command": command, "launch_environment": environment,
        "tmux_session": "umi_may_hand_dynamics_20260918", "run_name": run_name,
        "wandb_run_id": run_id, "wandb_url": reference["wandb_url"].rsplit("/", 1)[0] + "/" + run_id,
        "seed": 42, "rank_seeds": [42, 43, 44, 45], "physical_gpus": [3, 4, 5, 7],
        "fresh_policy": True, "resume": False, "num_envs_per_rank": 16384, "total_environments": 65536,
        "samples_per_iteration": reference["samples_per_iteration"], "max_iterations": 40000,
        "physics_hz": 120, "policy_hz": 10,
        "training_log_root": str(worktree / "logs/rsl_rl/umi_clearance_aprilcube60_r3"),
        "native_validation": str(args.native_validation.resolve()),
        "intended_behavioral_changes": ["Full nominal stock inertial properties for nine hand bodies", "May arm controller gains"],
        "instrumentation": "Read-only startup property verification on every rank; no random sampling or state writes",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("status", "git_branch", "git_commit", "seed", "physical_gpus", "wandb_url")}, indent=2))


if __name__ == "__main__":
    main()
