# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Copy the calibrated stock robot and narrow only its three wrist limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pxr import Usd, UsdPhysics

SOURCE_SHA256 = "8bfe88e5ceefdbd4524e4426cba93be94629634b642176143636935cee53610a"
METADATA_SHA256 = "625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242"
WRISTS = ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(stage: Usd.Stage) -> dict:
    result = {}
    for prim in stage.Traverse():
        result[str(prim.GetPath())] = {
            "type": str(prim.GetTypeName()),
            "schemas": list(prim.GetAppliedSchemas()),
            "attrs": {attr.GetName(): repr(attr.Get()) for attr in prim.GetAttributes()},
            "rels": {rel.GetName(): list(map(str, rel.GetTargets())) for rel in prim.GetRelationships()},
            "custom_data": prim.GetCustomData(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = source_dir / "robot.usd"
    source_metadata = source_dir / "metadata.yaml"
    if source == output_dir / "robot.usd" or output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("Output directory must be distinct and empty")
    if digest(source) != SOURCE_SHA256 or digest(source_metadata) != METADATA_SHA256:
        raise SystemExit("Source differs from reviewed calibration-only asset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "robot.usd"
    metadata = output_dir / "metadata.yaml"
    shutil.copy2(source, output)
    shutil.copy2(source_metadata, metadata)
    stage = Usd.Stage.Open(str(output))
    before = snapshot(stage)
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError("Source USD has no default robot prim")
    for name in WRISTS:
        prim = stage.GetPrimAtPath(f"{root.GetPath()}/{name}")
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"Missing revolute joint {name}")
        joint = UsdPhysics.RevoluteJoint(prim)
        if joint.GetLowerLimitAttr().Get() != -360.0 or joint.GetUpperLimitAttr().Get() != 360.0:
            raise RuntimeError(f"Unexpected source limits on {name}")
        joint.GetLowerLimitAttr().Set(-180.0)
        joint.GetUpperLimitAttr().Set(180.0)
    after = snapshot(stage)
    if before.keys() != after.keys():
        raise RuntimeError("Prim set changed")
    expected = {(f"{root.GetPath()}/{name}", attr) for name in WRISTS for attr in ("physics:lowerLimit", "physics:upperLimit")}
    changes = []
    for path, old in before.items():
        new = after[path]
        if any(old[key] != new[key] for key in ("type", "schemas", "rels", "custom_data")) or old["attrs"].keys() != new["attrs"].keys():
            raise RuntimeError(f"Prim structure changed: {path}")
        for attribute, value in old["attrs"].items():
            if value == new["attrs"][attribute]:
                continue
            if (path, attribute) not in expected:
                raise RuntimeError(f"Unexpected attribute change: {path}.{attribute}")
            changes.append({"prim": path, "attribute": attribute, "before": value, "after": new["attrs"][attribute]})
    if {(item["prim"], item["attribute"]) for item in changes} != expected or len(changes) != 6:
        raise RuntimeError("Expected exactly six wrist-limit attribute changes")
    stage.GetRootLayer().Save()
    if digest(metadata) != METADATA_SHA256:
        raise RuntimeError("Metadata changed")
    report = {
        "status": "PASS", "source": str(source), "source_sha256": digest(source),
        "source_metadata_sha256": digest(source_metadata), "output": str(output),
        "output_sha256": digest(output), "output_metadata_sha256": digest(metadata),
        "limits_degrees": {name: [-180.0, 180.0] for name in WRISTS},
        "attribute_changes": changes,
        "preserved": "All other USD attributes, prim structure, relationships, custom data, and metadata.",
    }
    (output_dir / "wrist_limit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "output", "output_sha256", "limits_degrees")}, indent=2))


if __name__ == "__main__":
    main()
