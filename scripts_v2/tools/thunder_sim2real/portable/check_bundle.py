"""Check relocated inputs, atlas lookup and optionally cuRobo GPU kinematics."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', action='store_true', help='Also run cuRobo kinematics and sphere checks on CUDA')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = Path(os.environ['UWLAB_DATA_ROOT'])
    assets = Path(os.environ['UWLAB_ASSET_ROOT'])
    model_dir = root / '15_table_reachability/model'
    cfg = json.loads((root/'49_clearance_and_placement/atlas/config.json').read_text())
    model = yaml.safe_load(Path(cfg['robot_model']).read_text())['robot_cfg']
    assert Path(model['kinematics']['urdf_path']).is_file()
    audit = json.loads((model_dir/'export_audit.json').read_text())
    assert all(Path(shape['mesh']).is_file() for shape in audit['collision_shapes'])
    scene = yaml.safe_load(Path(cfg['scene_model']).read_text())
    assert len(scene['cuboid']) == 33
    assert len(model['kinematics']['collision_link_names']) == 17
    assert sum(map(len, model['kinematics']['collision_spheres'].values())) == 239
    seed_path = root/'49_clearance_and_placement/lookup/atlas_joint_seeds.npy'
    seeds = np.load(seed_path, mmap_mode='r') if seed_path.exists() else None
    if seeds is not None:
        assert seeds.shape == (61, 73, 149, 504, 6)
    candidate = json.loads((REPO/'scripts_v2/tools/thunder_sim2real/workstation/collection.simulation_candidate.json').read_text())
    q = np.asarray(candidate['start_joint_positions_rad'], dtype=np.float64)[None]
    sys.path.insert(0, str(REPO/'scripts_v2/tools/curobo_umi'))
    from reachability_geometry import SourceCollisionChecker, batch_fk
    checker = SourceCollisionChecker(model_dir)
    check = checker.evaluate(q)
    assert check['clear'].all(), 'Selected collection endpoint failed source hull check'
    from pxr import Usd
    usd = assets/'thunder_d405_umi_rigid_asset/ur5e_robotiq_d405_umi_rigid_thunder.usd'
    stage = Usd.Stage.Open(str(usd))
    assert stage and stage.GetDefaultPrim().IsValid()
    banks = root/'49_clearance_and_placement/OmniReset/Resets/InsertiveAprilCube60__ReceptiveAprilCube60'
    import torch
    counts = {}
    for path in sorted(banks.glob('resets_*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        rows = data['initial_state']['articulation']['robot']['joint_position']
        assert all(torch.isfinite(row).all() for row in rows)
        counts[path.stem] = len(rows)
    if banks.exists():
        assert len(counts) == 4 and min(counts.values()) >= 10000
    # Load the runtime adapter directly without launching Isaac or importing task registration.
    path = REPO/'source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_sphere_geometry.py'
    spec = importlib.util.spec_from_file_location('portable_spheres', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bodies = list(dict.fromkeys(['base_link']+[j['child'] for j in audit['articulation_tree']]))
    adapter = module.UmiSphereGeometry(bodies, 'cpu')
    assert len(adapter.radii) == 239
    report = dict(passed=True, data_root=str(root), lab_boxes=33, robot_hulls=17,
                  robot_spheres=239, lookup_shape=list(seeds.shape) if seeds is not None else None, reset_counts=counts,
                  collection_endpoint_hull_clear=True, gpu_checked=False,
                  scope='Input and endpoint verification only; no route or physical robot validation.')
    if args.gpu:
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        kin = Kinematics(KinematicsCfg.from_data_dict(model['kinematics']))
        state = JointState.from_position(torch.tensor(q, device='cuda', dtype=torch.float32), joint_names=cfg['joint_names'])
        result = kin.compute_kinematics(state)
        pose = result.tool_poses.get_link_pose(audit['tool_frame'])
        expected = batch_fk(audit, q)[audit['tool_frame']][0, :3, 3]
        error = float(np.linalg.norm(pose.position.cpu().numpy().reshape(-1, 3)[0]-expected))
        assert error < 2e-5, error
        gpu_checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=model, scene_model=cfg['scene_model'], collision_activation_distance=0., self_collision_activation_distance=0.))
        # Match the accepted map's explicit sphere radii, pair masks and column margins.
        poses = batch_fk(audit, q)
        from scipy.spatial.transform import Rotation
        base_r = Rotation.from_quat(np.asarray(cfg['robot_base_quaternion_world_wxyz'])[[1,2,3,0]]).as_matrix()
        base_p = np.asarray(cfg['robot_base_position_world_m'])
        positions = np.stack([poses[n][:,:3,3] @ base_r.T+base_p for n in bodies], axis=1)
        rotations = np.stack([base_r @ poses[n][:,:3,:3] for n in bodies], axis=1)
        gpu_adapter = module.UmiSphereGeometry(bodies, 'cuda')
        self_clear, world_clear = gpu_adapter.evaluate(torch.tensor(positions, device='cuda', dtype=torch.float32),
            torch.tensor(rotations, device='cuda', dtype=torch.float32), torch.zeros(1,3,device='cuda'))
        assert bool(self_clear.all()) and bool(world_clear.all())
        report.update(gpu_checked=True, curobo_position_error_m=error, gpu_sphere_endpoint_clear=True,
                      curobo_checker_created=gpu_checker is not None)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
