"""Build an isolated USD layer that changes only nine hand bodies' mass properties."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_usd.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    usd = output / "umi_geometry_stock_hand_dynamics.usda"
    if usd.exists():
        raise FileExistsError(f"Refusing to overwrite {usd}")
    inputs = Path(__file__).resolve().parent
    properties_file = inputs / "stock_hand_properties.json"
    properties = json.loads(properties_file.read_text())
    bodies = properties["body_properties"]
    original = Usd.Stage.Open(str(base))
    stage = Usd.Stage.CreateNew(str(usd))
    stage.GetRootLayer().subLayerPaths = [str(base)]
    root = original.GetDefaultPrim().GetPath()
    stage.SetDefaultPrim(stage.GetPrimAtPath(root))
    conversions = {}
    for name, values in bodies.items():
        prim = stage.GetPrimAtPath(root.AppendChild(name))
        assert prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI), name
        inertia = np.asarray(values["inertia"]).reshape(3, 3)
        symmetric = (inertia + inertia.T) * 0.5
        diagonal, axes = np.linalg.eigh(symmetric)
        assert diagonal.min() > 0
        assert diagonal.max() <= diagonal.sum() - diagonal.max() + 1e-8
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        quaternion = Rotation.from_matrix(axes).as_quat()
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(values["mass"]))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*values["com"]))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal.tolist()))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*quaternion[:3].tolist())))
        assert np.max(np.abs(axes @ np.diag(diagonal) @ axes.T - symmetric)) < 1e-10
        conversions[name] = {
            "principal_inertias": diagonal.tolist(),
            "principal_axes_wxyz": [float(quaternion[3]), *quaternion[:3].tolist()],
            "symmetrization_max_error": float(np.max(np.abs(symmetric - inertia))),
        }
    stage.GetRootLayer().Save()
    differences = []
    for prim in original.Traverse():
        other = stage.GetPrimAtPath(prim.GetPath())
        assert other.IsValid() and prim.GetTypeName() == other.GetTypeName()
        for attribute in prim.GetAttributes():
            changed = other.GetAttribute(attribute.GetName())
            assert changed.IsValid()
            if attribute.Get() != changed.Get():
                differences.append({"prim": str(prim.GetPath()), "attribute": attribute.GetName()})
        for relationship in prim.GetRelationships():
            assert relationship.GetTargets() == other.GetRelationship(relationship.GetName()).GetTargets()
    allowed = {"physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"}
    assert all(x["attribute"] in allowed and Path(x["prim"]).name in bodies for x in differences)
    assert {str(p.GetPath()) for p in original.Traverse()} == {str(p.GetPath()) for p in stage.Traverse()}
    shutil.copyfile(base.parent / "metadata.yaml", output / "metadata.yaml")
    for name in ("stock_hand_properties.json", "reference_umi_properties.json"):
        shutil.copyfile(inputs / name, output / name)
    report = {
        "base_usd": str(base), "base_sha256": sha(base),
        "overlay_usd": str(usd), "overlay_sha256": sha(usd),
        "stock_properties_sha256": sha(properties_file),
        "metadata_sha256": sha(output / "metadata.yaml"),
        "conversions": conversions, "composed_attribute_differences": differences,
        "only_requested_mass_properties_changed": True,
        "geometry_joints_relationships_unchanged": True,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"asset": str(usd), "changed_attributes": len(differences), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
