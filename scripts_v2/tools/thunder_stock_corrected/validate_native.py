"""Check the stock D405 model's native mass properties and standalone linkage."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kind", choices=("robot", "hand"), required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args, multi_gpu=False).app

import hashlib
import json
import numpy as np
import torch
from pxr import Usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import thunder_stock_corrected_cfg as model


def run():
    cfg = model.ThunderStockCorrectedTrainCfg() if args.kind == "robot" else model.ThunderStockCorrectedGraspSamplingCfg()
    asset_cfg = cfg.scene.robot.copy()
    asset_cfg.prim_path = "/World/Robot"
    asset_cfg.init_state.pos = (0, 0, 1)
    asset_cfg.init_state.rot = (1, 0, 0, 0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
    robot = Articulation(asset_cfg)
    sim.reset()
    robot.update(0)
    props = json.loads(Path(__file__).with_name("mass_properties.json").read_text())
    view = robot.root_physx_view
    masses = view.get_masses().cpu().numpy()[0]
    coms = view.get_coms().cpu().numpy()[0]
    inertias = view.get_inertias().cpu().numpy()[0].reshape(-1, 3, 3)
    rows = {}
    for name, expected in props["bodies"].items():
        index = robot.body_names.index(name)
        m, c, tensor = masses[index], coms[index], inertias[index]
        assert np.isclose(m, expected["mass_kg"], rtol=2e-6, atol=1e-9), (name, m)
        assert np.allclose(c[:3], expected["center_of_mass_m"], rtol=2e-6, atol=1e-7), (name, c)
        assert np.allclose(np.linalg.eigvalsh(tensor), expected["principal_inertia_kg_m2"], rtol=3e-5, atol=1e-10), (name, tensor)
        # PhysX tensor inertia is returned in the local body frame. Record the
        # full tensor and COM quaternion so frame interpretation is auditable.
        direct_error = np.max(np.abs(tensor - expected["inertia_body_kg_m2"]))
        rows[name] = {"mass_kg":float(m), "com_local_xyz_xyzw":c.tolist(),
                      "inertia_native":tensor.tolist(), "direct_tensor_max_error":float(direct_error)}
        assert direct_error < 2e-9, (name, direct_error)
    usd = Usd.Stage.Open(asset_cfg.spawn.usd_path)
    mimics = [p.GetName() for p in usd.Traverse() if "PhysxMimicJointAPI:rotZ" in p.GetAppliedSchemas()]
    assert mimics == ["right_outer_knuckle_joint"], mimics
    report = {"status":"passed", "kind":args.kind, "asset":asset_cfg.spawn.usd_path,
              "asset_sha256":hashlib.sha256(Path(asset_cfg.spawn.usd_path).read_bytes()).hexdigest(),
              "body_properties":rows, "assembly_mass_kg":sum(r["mass_kg"] for r in rows.values()),
              "remaining_mimics":mimics, "native_shapes":int(view.max_shapes),
              "arm_stiffness":list(cfg.actions.arm.motion_stiffness) if args.kind == "robot" else None,
              "arm_damping_ratio":list(cfg.actions.arm.motion_damping_ratio) if args.kind == "robot" else None,
              "scope":"Nominal native asset properties; no mass randomization, fitting, chirp or PPO."}
    if args.kind == "hand":
        finger = robot.joint_names.index("finger_joint")
        upper = float(robot.data.joint_limits[0, finger, 1])
        endpoints = []
        for label, target in (("close", upper), ("open", 0.0)):
            for _ in range(240):
                robot.set_joint_position_target(torch.tensor([[target]], device=robot.device), joint_ids=[finger])
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(sim.get_physics_dt())
                assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all()
            endpoints.append({"command":label, "target_rad":target,
                              "joint_position":robot.data.joint_pos[0].tolist()})
        report.update(joint_names=robot.joint_names, close_open=endpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_MODEL_PASS", args.kind, report["assembly_mass_kg"], flush=True)


try:
    with torch.inference_mode():
        run()
except BaseException as exc:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status":"failed", "error":repr(exc)}, indent=2) + "\n")
    raise
finally:
    app.close()
