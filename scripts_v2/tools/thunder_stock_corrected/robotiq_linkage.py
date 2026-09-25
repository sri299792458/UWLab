"""Remove D/E angle couplings while preserving the closed Robotiq linkage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pxr import Usd, UsdPhysics

LINKAGE_JOINT_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("inner_knuckle_joint", "inner_finger_knuckle_joint")
)
MIMIC_SCHEMA = "PhysxMimicJointAPI:rotZ"
MIMIC_PREFIX = "physxMimicJoint:rotZ:"


def remove_linkage_mimics(stage: Usd.Stage) -> list[str]:
    """Remove only the four D/E mimic APIs and their authored properties.

    B remains fixed, C remains an enabled loop hinge, and the opposite-jaw A
    mimic remains intact. Existing limits and every physical joint remain.
    Calling this on an already-cleaned Robotiq asset is a no-op.
    """
    root = stage.GetDefaultPrim().GetPath()
    joints = [stage.GetPrimAtPath(root.AppendChild(name)) for name in LINKAGE_JOINT_NAMES]
    for prim, name in zip(joints, LINKAGE_JOINT_NAMES):
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Expected Robotiq hinge {root}/{name}")
    changed = []
    for prim in joints:
        schemas = prim.GetMetadata("apiSchemas")
        applied = schemas.GetAppliedItems() if schemas else []
        properties = [prop.GetName() for prop in prim.GetAuthoredProperties()
                      if prop.GetName().startswith(MIMIC_PREFIX)]
        if MIMIC_SCHEMA in applied or properties:
            prim.RemoveAppliedSchema(MIMIC_SCHEMA)
            for name in properties:
                prim.RemoveProperty(name)
            changed.append(prim.GetName())
    return changed


def _joint_snapshot(stage: Usd.Stage) -> dict:
    """Compare all joint definitions while excluding the authorized D/E edits."""
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        metadata = prim.GetAllMetadata()
        schemas = metadata.pop("apiSchemas", None)
        applied = list(schemas.GetAppliedItems()) if schemas else []
        if prim.GetName() in LINKAGE_JOINT_NAMES:
            applied = [name for name in applied if name != MIMIC_SCHEMA]
        properties = {}
        for prop in prim.GetAuthoredProperties():
            if prim.GetName() in LINKAGE_JOINT_NAMES and prop.GetName().startswith(MIMIC_PREFIX):
                continue
            properties[prop.GetName()] = str(prop.GetAllMetadata())
        result[str(prim.GetPath())] = (str(metadata), applied, properties)
    return result


def patch_asset(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open {path}")
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before = _joint_snapshot(stage)
    removed = remove_linkage_mimics(stage)
    assert before == _joint_snapshot(stage), "Unexpected edit to another joint setting"
    backup = path.with_name(path.stem + ".before_de_mimic_removal.usd")
    if removed:
        if backup.exists():
            raise FileExistsError(f"Preserving existing backup: {backup}")
        shutil.copy2(path, backup)
        stage.GetRootLayer().Save()
    # Also check the serialized result, not just the in-memory stage.
    stage.GetRootLayer().Reload()
    assert before == _joint_snapshot(stage)
    assert not remove_linkage_mimics(stage), "Mimic removal did not persist"
    remaining = []
    for prim in stage.Traverse():
        schemas = prim.GetMetadata("apiSchemas")
        if schemas and MIMIC_SCHEMA in schemas.GetAppliedItems():
            remaining.append(prim.GetName())
    assert remaining == ["right_outer_knuckle_joint"], remaining
    report = {"asset": str(path), "before_sha256": before_hash,
              "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "backup": str(backup) if backup.exists() else None,
              "removed_mimics": removed, "remaining_mimics": remaining,
              "other_joint_settings_unchanged": True}
    if removed:
        path.with_name("de_mimic_removal.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, action="append", required=True)
    args = parser.parse_args()
    for path in args.usd:
        print(json.dumps(patch_asset(path.resolve()), indent=2))


if __name__ == "__main__":
    main()
