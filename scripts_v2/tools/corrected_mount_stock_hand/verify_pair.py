"""Check the actual launched pair, after each run passes the shared verifier."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "may_hand_dynamics"))
from verify_run import ConfigLoader, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    names = ("may_gains", "current_umi_gains")
    metadata = [json.loads((root / name / "run_metadata.json").read_text()) for name in names]
    verified = [json.loads((root / name / "verification.json").read_text()) for name in names]
    assert all(x["status"] == "PASS" for x in verified)
    for key in ("git_commit", "comparison_asset_sha256", "dataset_sha256", "dataset_counts", "dataset_inputs", "seed", "rank_seeds", "num_envs_per_rank", "total_environments", "samples_per_iteration", "max_iterations", "physics_hz", "policy_hz", "corrected_root_quaternion_wxyz"):
        assert metadata[0][key] == metadata[1][key], key
    comparisons = {}
    for kind in ("env", "agent"):
        cfg = [yaml.load((Path(m["training_log_dir"]) / f"params/{kind}.yaml").read_text(), Loader=ConfigLoader) for m in metadata]
        changes = differences(*cfg)
        if kind == "env":
            prefixes = ("/seed", "/sim/device", "/log_dir", "/actions/arm/motion_stiffness", "/actions/arm/motion_damping_ratio", "/events/verify_stock_hand_dynamics/params", "/events/verify_corrected_mount/params/report_dir")
        else:
            prefixes = ("/seed", "/device", "/run_name")
        assert all(any(x["path"] == p or x["path"].startswith(p + "/") for p in prefixes) for x in changes), changes
        comparisons[kind] = changes
    result = {
        "status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "only_behavioral_pair_difference": "Arm controller gains",
        "corrected_root_quaternion_wxyz": metadata[0]["corrected_root_quaternion_wxyz"],
        "dataset_dir": metadata[0]["dataset_dir"], "reset_state_count": sum(metadata[0]["dataset_counts"].values()),
        "runtime_config_differences": comparisons,
        "runs": {name: {"wandb_url": m["wandb_url"], "max_verified_update": v["max_verified_update"], "finite_scalar_count": v["finite_scalar_count"], "rank_processes": v["rank_processes"]} for name, m, v in zip(names, metadata, verified)},
        "scope": "Correct startup and matched configuration; no claim of learned success or identical trajectories.",
    }
    (root / "pair_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "corrected_root_quaternion_wxyz", "reset_state_count", "runs")}, indent=2))


if __name__ == "__main__":
    main()
