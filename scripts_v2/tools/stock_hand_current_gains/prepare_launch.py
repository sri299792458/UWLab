"""Prepare the stock-property/current-gain run paired with the May-gain run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "may_hand_dynamics"))
from verify_run import ConfigLoader, differences


TASK = "OmniReset-UMI-StockHandCurrentGains-State-Train-v0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    args = parser.parse_args()
    worktree = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "run_metadata.json").exists(), "Do not overwrite an existing run identity"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree), "Commit experiment changes first"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=worktree, text=True).strip()
    reference = json.loads(args.reference_metadata.read_text())
    assert reference["seed"] == 42 and reference["total_environments"] == 65536
    assert reference["physics_hz"] == 120 and reference["policy_hz"] == 10
    asset = Path(reference["comparison_asset"])
    assert sha(asset) == reference["comparison_asset_sha256"]
    build = json.loads(Path(reference["asset_build_report"]).read_text())
    assert build["only_requested_mass_properties_changed"] and build["geometry_joints_relationships_unchanged"]
    assert sha(build["base_usd"]) == build["base_sha256"]
    assert sha(asset.with_name("metadata.yaml")) == build["metadata_sha256"]
    assert sha(asset.with_name("stock_hand_properties.json")) == build["stock_properties_sha256"]
    for item in reference["asset_files"]:
        assert sha(item["path"]) == item["sha256"], item["path"]
    bank = Path(reference["dataset_dir"]) / "Resets/InsertiveAprilCube60__ReceptiveAprilCube60"
    for family, digest in reference["dataset_sha256"].items():
        assert sha(bank / f"resets_{family}.pt") == digest, family
    for item in reference["dataset_inputs"].values():
        assert sha(item["path"]) == item["sha256"]

    validation = json.loads(args.native_validation.read_text())
    assert validation["status"] == "PASS" and validation["task"] == TASK
    assert Path(validation["asset"]) == asset
    old_cfg = yaml.load((Path(reference["training_log_dir"]) / "params/env.yaml").read_text(), Loader=ConfigLoader)
    new_cfg = yaml.load(args.native_validation.with_name("env.yaml").read_text(), Loader=ConfigLoader)
    preflight_changes = differences(old_cfg, new_cfg)
    prefixes = ("/seed", "/sim/device", "/log_dir", "/scene/num_envs", "/actions/arm/motion_stiffness", "/actions/arm/motion_damping_ratio", "/events/verify_stock_hand_dynamics/params")
    assert all(any(x["path"] == p or x["path"].startswith(p + "/") for p in prefixes) for x in preflight_changes), preflight_changes
    assert old_cfg["scene"]["robot"]["spawn"]["usd_path"] == new_cfg["scene"]["robot"]["spawn"]["usd_path"] == str(asset)
    preflight = {"status": "PASS", "reference_log": reference["training_log_dir"], "differences": preflight_changes}
    (output / "preflight_config_comparison.json").write_text(json.dumps(preflight, indent=2) + "\n")

    changed_source = subprocess.check_output(
        ["git", "diff", "--name-only", reference["git_commit"], commit, "--", "source", "scripts/reinforcement_learning/rsl_rl/train.py"],
        cwd=worktree, text=True,
    ).splitlines()
    cfg_root = "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/"
    assert set(changed_source) == {cfg_root + x for x in ("__init__.py", "may_hand_dynamics_cfg.py", "stock_hand_current_gains_cfg.py")}, changed_source

    run_id = uuid.uuid4().hex[:8]
    run_name = "umi_stock_hand_current_gains_seed42_4gpu_env16384"
    environment = reference["launch_environment"].copy()
    environment.pop("MAY_HAND_DYNAMICS_USD", None)
    environment.pop("MAY_HAND_DYNAMICS_REPORT_DIR", None)
    tags = [x for x in environment["WANDB_TAGS"].split(",") if x != "may-gains"]
    tags.extend(["current-umi-gains", "paired-gain-comparison"])
    environment.update(
        CUDA_VISIBLE_DEVICES="0,1,2,6", WANDB_RUN_ID=run_id, WANDB_RESUME="never", WANDB_TAGS=",".join(tags),
        PYTHONPATH=":".join(str(worktree / "source" / name) for name in ("uwlab", "uwlab_assets", "uwlab_tasks", "uwlab_rl")),
        STOCK_HAND_DYNAMICS_USD=str(asset), STOCK_HAND_DYNAMICS_REPORT_DIR=str(output / "native_training"),
    )
    command = reference["launch_command"].copy()
    train_index = next(i for i, value in enumerate(command) if value.endswith("scripts/reinforcement_learning/rsl_rl/train.py"))
    command[train_index] = str(worktree / "scripts/reinforcement_learning/rsl_rl/train.py")
    command[command.index("--task") + 1] = TASK
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
    property_files = [asset, asset.with_name("metadata.yaml"), asset.with_name("stock_hand_properties.json"), asset.with_name("reference_umi_properties.json"), asset.with_name("build_report.json")]
    metadata = {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(worktree), "git_branch": branch, "git_commit": commit,
        "baseline_git_commit": reference["baseline_git_commit"], "git_status_clean": True,
        "code_snapshot": str(archive), "code_snapshot_sha256": sha(archive),
        "reference_metadata": str(args.reference_metadata.resolve()),
        "reference_run_id": reference["wandb_run_id"], "reference_training_log_dir": reference["training_log_dir"],
        "failed_umi_baseline_metadata": reference["reference_metadata"],
        "reference_git_commit": reference["git_commit"], "changed_training_source_files": changed_source,
        "paired_native_reference_directory": str(args.reference_metadata.parent / "native_training"),
        "comparison_asset": str(asset), "comparison_asset_sha256": sha(asset),
        "asset_build_report": reference["asset_build_report"], "asset_files": reference["asset_files"],
        "stock_property_asset_files": [{"path": str(p), "sha256": sha(p)} for p in property_files],
        "dataset_dir": reference["dataset_dir"], "dataset_sha256": reference["dataset_sha256"],
        "dataset_counts": reference["dataset_counts"], "dataset_inputs": reference["dataset_inputs"],
        "launch_command": command, "launch_environment": environment,
        "tmux_session": "umi_stock_hand_current_gains_20260918", "run_name": run_name,
        "wandb_run_id": run_id, "wandb_url": reference["wandb_url"].rsplit("/", 1)[0] + "/" + run_id,
        "seed": 42, "rank_seeds": [42, 43, 44, 45], "physical_gpus": [0, 1, 2, 6],
        "fresh_policy": True, "resume": False, "num_envs_per_rank": 16384, "total_environments": 65536,
        "samples_per_iteration": reference["samples_per_iteration"], "max_iterations": reference["max_iterations"],
        "physics_hz": 120, "policy_hz": 10,
        "expected_arm_gains": {"kp": [500.0] * 3 + [60.0] * 3, "kd": [160.0] * 3 + [0.1] * 3},
        "training_log_root": str(worktree / "logs/rsl_rl/umi_clearance_aprilcube60_r3"),
        "native_validation": str(args.native_validation.resolve()), "native_validation_sha256": sha(args.native_validation),
        "intended_behavioral_changes_vs_reference": ["Current UMI arm gains instead of May gains"],
        "intended_behavioral_changes_vs_failed_baseline": ["Full nominal stock inertial properties for nine hand bodies"],
        "instrumentation": "Read-only startup property verification on every rank; no random sampling or state writes",
        "gpu2_sharing": "User explicitly authorized sharing the GPU containing the existing openpi process; that process is left untouched.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("status", "git_branch", "git_commit", "seed", "physical_gpus", "wandb_url")}, indent=2))


if __name__ == "__main__":
    main()
