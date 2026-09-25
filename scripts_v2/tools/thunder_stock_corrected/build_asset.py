"""Port the documented nine-body stock D405 mass model onto the current geometry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation

from extract_hand import extract
from robotiq_linkage import patch_asset

SOURCE_SHA = "00eb01b7c169b223bfe3e76f8be1285bc008e784db43a4a21b6eef9f5372ee90"
META_SHA = "625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242"
MASS_ATTRS = {"physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(stage):
    # Reuse the full composed-USD comparison from the preceding D405 stage.
    path = Path(__file__).resolve().parents[1] / "thunder_d405/build_asset.py"
    spec = importlib.util.spec_from_file_location("d405_geometry_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.snapshot(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remove-inner-mimics", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve() / "robot.usd"
    output = args.output_dir.resolve()
    assert sha(source) == SOURCE_SHA and sha(source.parent / "metadata.yaml") == META_SHA
    output.mkdir(parents=True, exist_ok=False)
    robot_dir = output / "robot"
    robot_dir.mkdir()
    target = robot_dir / "robot.usd"
    shutil.copy2(source, target)
    shutil.copy2(source.parent / "metadata.yaml", robot_dir / "metadata.yaml")
    props_path = Path(__file__).with_name("mass_properties.json")
    props = json.loads(props_path.read_text())
    stage = Usd.Stage.Open(str(target))
    before = snapshot(stage)
    root = stage.GetDefaultPrim().GetPath()
    for name, values in props["bodies"].items():
        prim = stage.GetPrimAtPath(root.AppendChild(name))
        assert prim and prim.HasAPI(UsdPhysics.RigidBodyAPI), name
        tensor = np.asarray(values["inertia_body_kg_m2"])
        moments, axes = np.linalg.eigh(tensor)
        assert np.all(moments > 0) and moments[-1] <= moments[:2].sum() + 1e-12
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        quat = Rotation.from_matrix(axes).as_quat()
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(values["mass_kg"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*values["center_of_mass_m"]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3])))
    stage.GetRootLayer().Save()
    stage.GetRootLayer().Reload()
    after = snapshot(stage)
    expected = {"/ROBOT/" + name for name in props["bodies"]}
    changes = []
    assert before.keys() == after.keys()
    for path in before:
        a, b = before[path], after[path]
        if a == b:
            continue
        assert path in expected, path
        changes.append(path)
        assert a["type"] == b["type"] and a["rels"] == b["rels"] and a["custom_data"] == b["custom_data"]
        assert set(b["schemas"]) == set(a["schemas"]) | {"PhysicsMassAPI"}
        # Adding MassAPI also exposes its unauthored density=0 schema default.
        density = stage.GetPrimAtPath(root.AppendChild(path.rsplit("/", 1)[1])).GetAttribute("physics:density")
        assert density.Get() == 0 and not density.HasAuthoredValueOpinion()
        ignored = MASS_ATTRS | {"physics:density"}
        assert {k: v for k, v in a["attrs"].items() if k not in ignored} == {
            k: v for k, v in b["attrs"].items() if k not in ignored}
    assert set(changes) == expected
    linkage = None
    if args.remove_inner_mimics:
        # Reuse the previously validated surgical joint patch before extracting
        # the standalone hand, so both carry the identical corrected linkage.
        linkage = patch_asset(target)
        assert len(linkage["removed_mimics"]) == 4
        stage.GetRootLayer().Reload()
    hand = extract(target, output / "hand/hand.usd")
    hand_stage = Usd.Stage.Open(hand["output"])
    for name in props["bodies"]:
        full_prim = stage.GetPrimAtPath(root.AppendChild(name))
        hand_prim = hand_stage.GetPrimAtPath(hand_stage.GetDefaultPrim().GetPath().AppendChild(name))
        for attr in MASS_ATTRS:
            assert full_prim.GetAttribute(attr).Get() == hand_prim.GetAttribute(attr).Get(), (name, attr)
    report = {
        "source": str(source), "source_sha256": sha(source),
        "properties_sha256": sha(props_path), "robot_sha256": sha(target),
        "hand_sha256": sha(hand["output"]), "metadata_sha256": META_SHA,
        "changed_bodies": sorted(changes), "inertial_port_only_mass_properties_changed": True,
        "standalone_matches_robot_mass_properties": True,
        "assembly_mass_kg": props["total_mass_kg"],
        "linkage_correction": linkage,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
