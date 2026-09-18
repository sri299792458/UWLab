"""Read native stock housing properties after removing camera colliders in memory."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app

import hashlib,json
import numpy as np
from isaaclab.sim import SimulationContext,SimulationCfg
from isaaclab.assets import Articulation
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/23_inertia_gap')
source=Path('/home/kanth042/.cache/uwlab/assets/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd')
digest=hashlib.sha256(source.read_bytes()).hexdigest()
sim=SimulationContext(SimulationCfg(dt=1/120,device=args.device))
cases={'stock':[], 'without_safety':['d415_and_cable'],
       'without_mount':['D415_to_Robotiq_Mount'],
       'without_both':['d415_and_cable','D415_to_Robotiq_Mount']}
robots={}
for i,(name,removed) in enumerate(cases.items()):
    cfg=EXPLICIT_UR5E_ROBOTIQ_2F85.copy();cfg.prim_path=f'/World/Robot_{name}'
    cfg.spawn.usd_path=str(source);cfg.init_state.pos=(3.*i,0,0)
    robots[name]=Articulation(cfg)
    for leaf in removed:
        path=cfg.prim_path+'/robotiq_base_link/collisions/'+leaf
        assert sim.stage.GetPrimAtPath(path)
        assert sim.stage.GetPrimAtPath(path).SetActive(False)
        assert not sim.stage.GetPrimAtPath(path).IsActive()
sim.reset();sim.forward()
rows={}
for name,robot in robots.items():
    robot.update(0.);view=robot.root_physx_view;bid=robot.body_names.index('robotiq_base_link')
    rows[name]=dict(removed_collision_shapes=cases[name],mass_kg=float(view.get_masses()[0,bid]),
        com_m=view.get_coms()[0,bid,:3].cpu().tolist(),
        inertia_body_kg_m2=view.get_inertias()[0,bid].reshape(3,3).cpu().tolist())
ref=json.loads((ROOT/'stock_factorial/worker_0/inertial_intervention.json').read_text());bid=ref['body_names'].index('robotiq_base_link')
assert np.isclose(rows['stock']['mass_kg'],ref['original_masses_kg'][bid],atol=1e-7)
assert np.allclose(rows['stock']['inertia_body_kg_m2'],np.array(ref['original_inertias_kg_m2'][bid]).reshape(3,3),atol=2e-9,rtol=1e-5)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
out=dict(source_usd=str(source),source_sha256=digest,source_unchanged=True,
    method='Four independently spawned stock articulations. Named collision prims deactivated only in memory before physics initialization. No mass/inertia setter, controller action, or source-file edit. Baseline matches the previous native fixture.',cases=rows)
(ROOT/'housing_native_provenance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True);app.close(skip_cleanup=True)
