"""Extract the exact stock D405 hand from the full robot for object-local grasp sampling.

Run with the UWLab Python environment. The full robot is never modified.
Body frames are expressed relative to robotiq_base_link; joint-local frames,
collision geometry, masses, and inertia tensors are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def extract(source: Path, output: Path) -> dict:
    source_stage = Usd.Stage.Open(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    source_stage.Flatten().Export(str(output))
    stage = Usd.Stage.Open(str(output))
    root = stage.GetDefaultPrim()
    base = root.GetPath().AppendChild("robotiq_base_link")
    cache = UsdGeom.XformCache()
    inverse_base = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(base)).GetInverse()
    bodies = [p for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)
              and (p.GetName() == "robotiq_base_link" or p.GetName().startswith(("left_", "right_")))]
    body_paths = {p.GetPath() for p in bodies}
    transforms = {p.GetPath(): cache.GetLocalToWorldTransform(p) * inverse_base for p in bodies}
    remove = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
            if not targets or any(p not in body_paths for p in targets):
                remove.append(prim.GetPath())
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetPath() not in body_paths:
            remove.append(prim.GetPath())
        elif prim.GetTypeName() == "DomeLight":
            remove.append(prim.GetPath())
    for path in sorted(remove, key=lambda p: len(str(p)), reverse=True):
        stage.RemovePrim(path)
    for prim in bodies:
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(transforms[prim.GetPath()])
    UsdGeom.Xformable(root).MakeMatrixXform().Set(Gf.Matrix4d(1))
    joint = UsdPhysics.FixedJoint.Define(stage, root.GetPath().AppendChild("root_joint"))
    joint.CreateBody1Rel().SetTargets([base])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    stage.GetRootLayer().Save()
    shutil.copy2(source.parent / "metadata.yaml", output.parent / "metadata.yaml")
    reference_path = source.parent / 'umi_grasp_reference.json'
    if reference_path.exists():
        shutil.copy2(reference_path, output.parent / reference_path.name)
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "bodies": [p.GetName() for p in bodies], "body_count": len(bodies),
              "purpose": "Fixed-base hand fixture for object-local grasp sampling; exact full-robot hand geometry/inertials."}
    assert len(bodies) == 9
    (output.parent / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
