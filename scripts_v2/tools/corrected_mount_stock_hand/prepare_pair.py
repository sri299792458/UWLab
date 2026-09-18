"""Record and prepare both authorized corrected-mount stock-hand training runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "may_hand_dynamics"))
from verify_run import ConfigLoader, differences

QUATERNION = [0.5, 0.5, 0.5, 0.5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allowed(changes, prefixes):
    return all(any(x["path"] == p or x["path"].startswith(p + "/") for p in prefixes) for x in changes)


def verify_dataset(data_root, output):
    bank_root = data_root / "49_clearance_and_placement"
    audit = json.loads((bank_root / "dataset_audit.json").read_text())
    atlas_path = bank_root / "atlas/config.json"
    atlas = json.loads(atlas_path.read_text())
    assert audit["all_passed"] and atlas["robot_base_quaternion_world_wxyz"] == QUATERNION
    families = {}
    torch.set_num_threads(1)
    for family, previous in audit["families"].items():
        path = bank_root / "OmniReset/Resets/InsertiveAprilCube60__ReceptiveAprilCube60" / f"resets_{family}.pt"
        digest = sha(path)
        assert digest == previous["sha256"], family
        data = torch.load(path, map_location="cpu", weights_only=False)
        count = previous["count"]
        for group in data["initial_state"].values():
            for entity in group.values():
                for values in entity.values():
                    tensor = torch.stack(values) if isinstance(values, list) else values
                    assert tensor.shape[0] == count and torch.isfinite(tensor).all(), family
        root = torch.stack(data["initial_state"]["articulation"]["robot"]["root_pose"])
        expected = torch.tensor(QUATERNION, dtype=root.dtype)
        assert torch.equal(root[:, 3:], expected.expand(count, 4)), family
        families[family] = {"path": str(path), "sha256": digest, "count": count, "all_finite": True, "all_mounts_exactly_correct": True}
    inputs = json.loads((bank_root / "fresh_inputs.json").read_text())
    for item in inputs.values():
        assert sha(item["path"]) == item["sha256"]
    result = {
        "status": "PASS", "dataset_dir": str(bank_root / "OmniReset"), "families": families,
        "total_count": sum(x["count"] for x in families.values()), "root_quaternion_wxyz": QUATERNION,
        "dataset_inputs": inputs, "atlas_config": str(atlas_path), "atlas_config_sha256": sha(atlas_path),
        "dataset_audit_sha256": sha(bank_root / "dataset_audit.json"),
    }
    assert result["total_count"] == 40194
    (output / "dataset_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-may", type=Path, required=True)
    parser.add_argument("--previous-current", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worktree = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    assert not (output / "pair_metadata.json").exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=worktree, text=True).strip()
    dataset = verify_dataset(args.data_root.resolve(), output)
    archive = output / "git_source_snapshot.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "--output", str(archive), commit], cwd=worktree, check=True)
    configurations = {}
    records = {}
    runs = [
        ("may_gains", args.previous_may, [3, 4, 5, 7], "OmniReset-UMI-MayHandDynamics-State-Train-v0", "MAY_HAND_DYNAMICS", [200.0] * 3 + [3.0] * 3, [84.8528137423857] * 3 + [3.4641016151377544] * 3),
        ("current_umi_gains", args.previous_current, [0, 1, 2, 6], "OmniReset-UMI-StockHandCurrentGains-State-Train-v0", "STOCK_HAND_DYNAMICS", [500.0] * 3 + [60.0] * 3, [160.0] * 3 + [0.1] * 3),
    ]
    for name, reference_path, gpus, task, variable_prefix, kp, kd in runs:
        root = output / name
        assert not (root / "run_metadata.json").exists()
        previous = json.loads(reference_path.read_text())
        assert previous["seed"] == 42 and previous["total_environments"] == 65536
        assert previous["status"] == "stopped_by_user_for_corrected_mount"
        asset = Path(previous["comparison_asset"])
        assert sha(asset) == previous["comparison_asset_sha256"]
        for item in previous["asset_files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
        build = json.loads(Path(previous["asset_build_report"]).read_text())
        assert build["only_requested_mass_properties_changed"] and build["geometry_joints_relationships_unchanged"]
        assert sha(build["base_usd"]) == build["base_sha256"]
        assert sha(asset.with_name("metadata.yaml")) == build["metadata_sha256"]
        assert sha(asset.with_name("stock_hand_properties.json")) == build["stock_properties_sha256"]
        native_path = root / "native_validation/report.json"
        validation = json.loads(native_path.read_text())
        assert validation["status"] == "PASS" and validation["task"] == task and validation["asset"] == str(asset)
        mount = json.loads((root / "native_validation/mount/rank_0.json").read_text())
        assert mount["status"] == "PASS" and mount["verified_after_first_reset"] == 256
        assert abs(abs(sum(a * b for a, b in zip(mount["native_root_quaternion_first_wxyz"], QUATERNION))) - 1.0) < 2e-6
        old_cfg = yaml.load((Path(previous["training_log_dir"]) / "params/env.yaml").read_text(), Loader=ConfigLoader)
        cfg = yaml.load((root / "native_validation/env.yaml").read_text(), Loader=ConfigLoader)
        changes = differences(old_cfg, cfg)
        prefixes = ("/seed", "/sim/device", "/log_dir", "/scene/num_envs", "/scene/robot/init_state/rot", "/events/reset_from_reset_states/params/dataset_dir", "/events/verify_stock_hand_dynamics/params/report_dir", "/events/verify_corrected_mount")
        assert allowed(changes, prefixes), changes
        assert cfg["scene"]["robot"]["spawn"]["usd_path"] == str(asset)
        assert cfg["events"]["reset_from_reset_states"]["params"]["dataset_dir"] == dataset["dataset_dir"]
        configurations[name] = cfg
        (root / "preflight_config_comparison.json").write_text(json.dumps({"status": "PASS", "reference_log": previous["training_log_dir"], "differences": changes}, indent=2) + "\n")

        run_id = uuid.uuid4().hex[:8]
        run_name = f"umi_corrected_mount_stock_hand_{name}_seed42_4gpu_env16384"
        environment = previous["launch_environment"].copy()
        for prefix in ("MAY_HAND_DYNAMICS", "STOCK_HAND_DYNAMICS"):
            environment.pop(prefix + "_USD", None)
            environment.pop(prefix + "_REPORT_DIR", None)
        environment.update({
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "WANDB_RUN_ID": run_id, "WANDB_RESUME": "never",
            "WANDB_TAGS": environment["WANDB_TAGS"] + ",corrected-physical-mount,20mm-map,paired-gain-comparison",
            "PYTHONPATH": ":".join(str(worktree / "source" / module) for module in ("uwlab", "uwlab_assets", "uwlab_tasks", "uwlab_rl")),
            "UWLAB_DATA_ROOT": str(args.data_root.resolve()), "UWLAB_ASSET_ROOT": "/data/kanth042/converted_assets",
            variable_prefix + "_USD": str(asset), variable_prefix + "_REPORT_DIR": str(root / "native_training"),
            "CORRECTED_MOUNT_REPORT_DIR": str(root / "mount_training"),
        })
        command = previous["launch_command"].copy()
        index = next(i for i, x in enumerate(command) if x.endswith("scripts/reinforcement_learning/rsl_rl/train.py"))
        command[index] = str(worktree / "scripts/reinforcement_learning/rsl_rl/train.py")
        command[command.index("--task") + 1] = task
        command[command.index("--run_name") + 1] = run_name
        index = next(i for i, x in enumerate(command) if x.startswith("env.events.reset_from_reset_states.params.dataset_dir="))
        command[index] = "env.events.reset_from_reset_states.params.dataset_dir=" + dataset["dataset_dir"]
        lines = ["#!/bin/bash", "set -u", "cd " + shlex.quote(str(worktree))]
        lines += ["export " + key + "=" + shlex.quote(value) for key, value in environment.items()]
        lines += [shlex.join(command) + " > " + shlex.quote(str(root / "train.log")) + " 2>&1", "training_status=$?", 'printf "%s\\n" "$training_status" > ' + shlex.quote(str(root / "train.exit")), 'exit "$training_status"']
        launcher = root / "launch.sh"
        launcher.write_text("\n".join(lines) + "\n")
        launcher.chmod(0o755)
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        property_files = [asset, asset.with_name("metadata.yaml"), asset.with_name("stock_hand_properties.json"), asset.with_name("reference_umi_properties.json"), asset.with_name("build_report.json")]
        metadata = {
            "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
            "worktree": str(worktree), "git_branch": branch, "git_commit": commit, "git_status_clean": True,
            "baseline_git_commit": previous["baseline_git_commit"], "mounting_source_commit": "afa9924fe099710a5cedd86a9b91a05ecbcd419a",
            "code_snapshot": str(archive), "code_snapshot_sha256": sha(archive),
            "reference_metadata": str(reference_path.resolve()), "reference_run_id": previous["wandb_run_id"],
            "reference_training_log_dir": previous["training_log_dir"], "comparison_asset": str(asset),
            "comparison_asset_sha256": sha(asset), "asset_build_report": previous["asset_build_report"], "asset_files": previous["asset_files"],
            "stock_property_asset_files": [{"path": str(p), "sha256": sha(p)} for p in property_files],
            "dataset_dir": dataset["dataset_dir"], "dataset_sha256": {k: v["sha256"] for k, v in dataset["families"].items()},
            "dataset_counts": {k: v["count"] for k, v in dataset["families"].items()}, "dataset_inputs": dataset["dataset_inputs"],
            "dataset_verification": str(output / "dataset_verification.json"),
            "corrected_root_quaternion_wxyz": QUATERNION, "nominal_root_position_m": [0.177660, 0.377695, 1.466],
            "mount_native_report_directory": str(root / "mount_training"),
            "launch_command": command, "launch_environment": environment,
            "tmux_session": "umi_corrected_mount_" + name + "_20260918", "run_name": run_name,
            "wandb_run_id": run_id, "wandb_url": previous["wandb_url"].rsplit("/", 1)[0] + "/" + run_id,
            "seed": 42, "rank_seeds": [42, 43, 44, 45], "physical_gpus": gpus,
            "fresh_policy": True, "resume": False, "num_envs_per_rank": 16384, "total_environments": 65536,
            "samples_per_iteration": 2097152, "max_iterations": 40000, "physics_hz": 120, "policy_hz": 10,
            "expected_arm_gains": {"kp": kp, "kd": kd}, "gains_label": name,
            "training_log_root": str(worktree / "logs/rsl_rl/umi_clearance_aprilcube60_r3"),
            "native_validation": str(native_path), "native_validation_sha256": sha(native_path),
            "intended_behavioral_changes_vs_previous_run": ["Physical base mounting correction", "Regenerated matching reset bank from corrected 20 mm map"],
            "pair_interpretation": "The two corrected-mount runs differ behaviorally only in arm gains. Comparisons to historical old-mount runs also change mounting and reset data.",
            "instrumentation": "Read-only native property startup audit and one-time native base-pose verification after the first full reset; no physics or RNG modifications",
        }
        if name == "current_umi_gains":
            metadata["paired_native_reference_directory"] = str(output / "may_gains/native_training")
        records[name] = metadata

    pair_changes = differences(configurations["may_gains"], configurations["current_umi_gains"])
    assert allowed(pair_changes, ("/sim/device", "/actions/arm/motion_stiffness", "/actions/arm/motion_damping_ratio", "/events/verify_stock_hand_dynamics/params", "/events/verify_corrected_mount/params/report_dir")), pair_changes
    for name, metadata in records.items():
        (output / name / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pair = {"status": "prepared", "git_commit": commit, "data": dataset, "preflight_pair_differences": pair_changes, "runs": {name: {k: v[k] for k in ("wandb_run_id", "wandb_url", "physical_gpus", "run_name", "tmux_session")} for name, v in records.items()}}
    (output / "pair_metadata.json").write_text(json.dumps(pair, indent=2) + "\n")
    print(json.dumps(pair["runs"], indent=2))


if __name__ == "__main__":
    main()
