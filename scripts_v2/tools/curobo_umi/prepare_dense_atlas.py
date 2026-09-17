"""Freeze the dense table pose grid, inputs, and modeling choices."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reachability_geometry import ROOT, SourceCollisionChecker
from benchmark_dense_ik import prepare_config


def main() -> None:
    out=ROOT/'20_dense_reachability'
    checker=SourceCollisionChecker(ROOT/'15_table_reachability/model',clearance=.001)
    lab=json.loads((ROOT/'14_lab_setup_audit/lab_geometry_audit.json').read_text())
    top=next(c['collision_world_bounds'] for c in lab['components'] if c['index']==28)
    def axis(lo: float, hi: float, maximum_step: float) -> list:
        return np.linspace(lo,hi,math.ceil((hi-lo)/maximum_step)+1).tolist()
    heights=[.03,.06,.09,.12,.18,.24,.30,.40,.50,.60]
    orientations=[]
    for tilt in [0,30,60,90,120,150,180]:
        for azimuth in ([0] if tilt in [0,180] else list(range(0,360,45))):
            for roll in range(0,360,30):
                quat=Rotation.from_euler('ZYZ',[azimuth,180-tilt,roll],degrees=True).as_quat()[[3,0,1,2]]
                orientations.append(dict(index=len(orientations),tilt_deg=tilt,azimuth_deg=azimuth,
                    roll_deg=roll,quaternion_world_wxyz=quat.tolist()))
    # Remove only exact orientation duplicates; poles are already handled above.
    matrices=Rotation.from_quat(np.array([o['quaternion_world_wxyz'] for o in orientations])[:,[1,2,3,0]]).as_matrix()
    assert len(np.unique(np.round(matrices.reshape(-1,9),7),axis=0))==len(orientations)==504
    fine=[]
    for roll in range(0,360,15):
        quat=Rotation.from_euler('ZYZ',[0,180,roll],degrees=True).as_quat()[[3,0,1,2]]
        fine.append(dict(index=len(fine),tilt_deg=0,azimuth_deg=0,roll_deg=roll,quaternion_world_wxyz=quat.tolist()))
    config=dict(version=1,table_bounds_world_m=top,tabletop_z_m=top['max'][2],
        tool_frame=checker.audit['tool_frame'],tool_offset=checker.audit['tool_offset'],
        robot_base_position_world_m=checker.audit['lab_constants']['ROBOT_POS'],
        robot_base_quaternion_world_wxyz=checker.audit['lab_constants']['ROBOT_ROT'],
        broad=dict(x_world_m=axis(top['min'][0],top['max'][0],.02),y_world_m=axis(top['min'][1],top['max'][1],.02),
            height_above_table_m=heights,orientations=orientations),
        fine=dict(x_world_m=axis(top['min'][0],top['max'][0],.01),y_world_m=axis(top['min'][1],top['max'][1],.01),
            height_above_table_m=[.03,.06,.09,.12],orientations=fine),
        open_hand_joint_values={},
        robot_config=prepare_config(checker),
        joint_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'],
        joint_limits_rad=checker.joint_limits.tolist(),minimum_source_hull_clearance_m=.001,
        ik_position_tolerance_m=.001,ik_rotation_tolerance_rad=.01,ik_seeds=32,
        batch_size=512,random_seed=1709,
        status_codes={'0':'not_processed','1':'nominal_hand_target_blocked','2':'no_converged_IK_found',
            '3':'all_found_configurations_blocked','4':'source_hull_clear_solution_found'},
        approach=dict(grid='fine',length_m=.10,cartesian_sample_spacing_m=.01,
            description='Reverse a sampled 10 cm vertical withdrawal from each chosen endpoint; local path only, not a path from home'),
        modeling_notes=[
            'Full physical tabletop, including outside the narrower training workspace.',
            'Open empty hand: all six finger articulation coordinates locked at zero; no payload.',
            'World + robot self collision use original exported Isaac convex hulls and lab collision boxes through FCL.',
            'cuRobo performs batched multi-seed IK; sphere overlap does not determine final acceptance.',
            'Exact target hand geometry and achieved full-arm configuration must each meet the clearance criterion.',
            'No-found labels describe this search budget, not proof of continuous geometric impossibility.',
            'Dense discrete grid; no guarantee for unsampled positions, orientations, or continuous approach intervals.',
            'The 1 mm clearance is a geometric reporting threshold, not a hardware safety margin.',
        ])
    config['open_hand_joint_values']={j['name']:0. for j in checker.audit['articulation_tree']
        if not j['fixed'] and j['name'] not in config['joint_names']}
    config['counts']={name:len(config[name]['x_world_m'])*len(config[name]['y_world_m'])*
        len(config[name]['height_above_table_m'])*len(config[name]['orientations']) for name in ['broad','fine']}
    paths=[checker.model_dir/'export_audit.json',checker.model_dir/'thunder_umi.urdf',checker.model_dir/'lab_scene.yml',
        Path(checker.audit['robot_usd']),Path(checker.audit['robot_usd']).parent/'metadata.yaml',
        ROOT/'14_lab_setup_audit/lab_geometry_audit.json',Path(__file__).with_name('reachability_geometry.py')]
    paths += [Path(s['mesh']) for s in checker.shapes]
    config['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'atlas_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(counts=config['counts'],total=sum(config['counts'].values()),
        broad_shape=[len(config['broad'][k]) for k in ['height_above_table_m','orientations','y_world_m','x_world_m']],
        fine_shape=[len(config['fine'][k]) for k in ['height_above_table_m','orientations','y_world_m','x_world_m']]),indent=2))


if __name__=='__main__':main()
