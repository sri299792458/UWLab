"""Verify the launched comparison against the recorded failed seed-42 baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ConfigLoader(yaml.SafeLoader):
    pass


def preserve_tag(loader, tag, node):
    if isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_scalar(node)
    return {"python_yaml_tag": tag, "value": value}


ConfigLoader.add_multi_constructor("", preserve_tag)


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((differences(a.get(k), b.get(k), path + "/" + str(k)) for k in sorted(a.keys() | b.keys())), [])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum((differences(x, y, path + "/" + str(i)) for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [{"path": path, "reference": a, "comparison": b}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert hashlib.sha256(Path(metadata["comparison_asset"]).read_bytes()).hexdigest() == metadata["comparison_asset_sha256"]
    baseline = Path(metadata["reference_training_log_dir"])
    logs = [p for p in Path(metadata["training_log_root"]).glob("*_" + metadata["run_name"]) if (p / "model_0.pt").exists()]
    assert len(logs) == 1, logs
    log = logs[0]
    config_differences = {}
    for kind in ("env", "agent"):
        a = yaml.load((baseline / f"params/{kind}.yaml").read_text(), Loader=ConfigLoader)
        b = yaml.load((log / f"params/{kind}.yaml").read_text(), Loader=ConfigLoader)
        changes = differences(a, b)
        if kind == "env":
            prefixes = ("/seed", "/sim/device", "/log_dir", "/scene/robot/spawn/usd_path", "/actions/arm/motion_stiffness", "/actions/arm/motion_damping_ratio", "/events/verify_stock_hand_dynamics")
            assert b["scene"]["num_envs"] == 16384
        else:
            prefixes = ("/seed", "/device", "/run_name")
            assert not b["resume"] and b["max_iterations"] == 40000 and b["num_steps_per_env"] == 32
        assert b["seed"] in metadata["rank_seeds"]
        unexpected = [x for x in changes if not any(x["path"] == p or x["path"].startswith(p + "/") for p in prefixes)]
        assert not unexpected, unexpected
        config_differences[kind] = changes
    native_reports = []
    for rank in range(4):
        native = json.loads((output / f"native_training/rank_{rank}.json").read_text())
        assert native["status"] == "PASS" and native["num_envs"] == 16384
        assert native["rank"] == rank and native["seed"] == 42 + rank
        if "expected_arm_gains" in metadata:
            for key in ("kp", "kd"):
                assert np.allclose(native[key], metadata["expected_arm_gains"][key], rtol=1e-6, atol=1e-6)
        if "paired_native_reference_directory" in metadata:
            paired = json.loads((Path(metadata["paired_native_reference_directory"]) / f"rank_{rank}.json").read_text())
            for key in ("asset", "body_names", "default_masses_kg", "default_inertias_kg_m2", "com_poses", "action_scale", "physics_dt", "policy_dt"):
                assert native[key] == paired[key], (rank, key)
        native_reports.append(native)
    processes = []
    parents = []
    script = str(Path(metadata["worktree"]) / "scripts/reinforcement_learning/rsl_rl/train.py").encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
            if script not in argv or metadata["run_name"].encode() not in argv:
                continue
            environment = dict(x.split(b"=", 1) for x in proc.joinpath("environ").read_bytes().split(b"\0") if b"=" in x)
            if b"LOCAL_RANK" not in environment:
                parents.append(int(proc.name))
                continue
            rank = int(environment[b"LOCAL_RANK"])
            assert environment[b"CUDA_VISIBLE_DEVICES"].decode() == ",".join(map(str, metadata["physical_gpus"]))
            assert environment[b"PYTHONPATH"].decode() == metadata["launch_environment"]["PYTHONPATH"]
            processes.append({"pid": int(proc.name), "rank": rank, "physical_gpu": metadata["physical_gpus"][rank]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    assert sorted(p["rank"] for p in processes) == list(range(4))
    assert not (output / "train.exit").exists()
    assert "Loading model checkpoint from:" not in (output / "train.log").read_text(errors="replace")
    accumulator = EventAccumulator(str(log), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    assert scalars and all(np.isfinite([v.value for v in seq]).all() for seq in scalars.values())
    step = max(v.step for seq in scalars.values() for v in seq)
    assert step >= 2, step
    checkpoint = log / "model_0.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["iter"] == 0
    assert all(torch.isfinite(v).all() for v in saved["model_state_dict"].values())
    result = {
        "status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "training_log_dir": str(log), "max_verified_update": step, "finite_scalar_count": len(scalars),
        "config_differences": config_differences, "unexpected_config_differences": [],
        "native_verified_ranks": [r["rank"] for r in native_reports],
        "rank_processes": sorted(processes, key=lambda p: p["rank"]), "torchrun_pids": parents,
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "iteration": 0, "all_model_tensors_finite": True},
        "seed": 42, "fresh_policy_no_resume": True,
        "initial_weight_equality_to_reference": "Not measured; model_0 is already after the first update.",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata.update(status="running_verified", verified_utc=result["verified_utc"], training_log_dir=str(log), max_verified_update=step, live_rank_processes=result["rank_processes"])
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "max_verified_update", "finite_scalar_count", "native_verified_ranks", "rank_processes", "training_log_dir")}, indent=2))


if __name__ == "__main__":
    main()
