"""Verify the training bank and compare composed stock/UMI assets without a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from pxr import Usd

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.bundle.expanduser().resolve()
    torch.set_num_threads(1)
    reference = json.loads((HERE / "reference_runs.json").read_text())["runs"]["may_gains"]
    manifest = json.loads((HERE / "stock_asset_manifest.json").read_text())
    stock_dir = root / manifest["installed_directory"]
    for name, entry in manifest["files"].items():
        assert sha(stock_dir / name) == entry["sha256"], name
    original_path = root / manifest["base_usd_relative"]
    assert sha(original_path) == manifest["base_usd_sha256"]
    original = Usd.Stage.Open(str(original_path))
    stock = Usd.Stage.Open(str(stock_dir / "umi_geometry_stock_hand_dynamics.usda"))
    assert original and stock
    assert {str(p.GetPath()) for p in original.Traverse()} == {str(p.GetPath()) for p in stock.Traverse()}
    expected_changes = json.loads((stock_dir / "build_report.json").read_text())["composed_attribute_differences"]
    changes = []
    for prim in original.Traverse():
        other = stock.GetPrimAtPath(prim.GetPath())
        assert prim.GetTypeName() == other.GetTypeName()
        for attr in prim.GetAttributes():
            assert other.GetAttribute(attr.GetName()).IsValid()
            if attr.Get() != other.GetAttribute(attr.GetName()).Get():
                changes.append({"prim": str(prim.GetPath()), "attribute": attr.GetName()})
        for rel in prim.GetRelationships():
            assert rel.GetTargets() == other.GetRelationship(rel.GetName()).GetTargets()
    assert changes == expected_changes and len(changes) == 36
    counts = {}
    for name, expected_sha in reference["dataset_sha256"].items():
        path = root / "data/49_clearance_and_placement/OmniReset/Resets/InsertiveAprilCube60__ReceptiveAprilCube60" / f"resets_{name}.pt"
        assert sha(path) == expected_sha, name
        data = torch.load(path, map_location="cpu", weights_only=True)
        count = reference["dataset_counts"][name]
        for group in data["initial_state"].values():
            for entity in group.values():
                for values in entity.values():
                    tensor = torch.stack(values) if isinstance(values, list) else values
                    assert tensor.shape[0] == count and torch.isfinite(tensor).all(), name
        poses = torch.stack(data["initial_state"]["articulation"]["robot"]["root_pose"])
        assert torch.equal(poses[:, 3:], torch.full_like(poses[:, 3:], 0.5)), name
        counts[name] = count
    result = {"status": "PASS", "bundle": str(root), "reset_counts": counts,
              "total_reset_states": sum(counts.values()), "all_reset_values_finite": True,
              "root_quaternion_wxyz": [0.5] * 4, "stock_changed_attributes": len(changes),
              "stock_geometry_joints_relationships_unchanged": True,
              "scope": "Exact input hashes, reset tensors and composed USD properties; native dynamics checked separately."}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
