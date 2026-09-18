"""Screen full saved initial body geometry against source convex collision hulls."""
import itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911');OUT=ROOT/'24_gain_failure_diagnosis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parents[1]/'curobo_umi'))
from reachability_geometry import convex_geometry,set_transform,exact_pair_gap,pose_matrix,fcl
MODEL=ROOT/'15_table_reachability/model';audit=json.loads((MODEL/'export_audit.json').read_text())
shapes=audit['collision_shapes'];objects=[fcl.CollisionObject(convex_geometry(s['mesh'])[0]) for s in shapes]
scene=yaml.safe_load((MODEL/'lab_scene.yml').read_text())['cuboid']
world={n:fcl.CollisionObject(fcl.Box(*v['dims']),fcl.Transform(pose_matrix(v['pose'][:3],v['pose'][3:])[:3,:3],v['pose'][:3])) for n,v in scene.items()}
adj={frozenset((j['body0'],j['body1'])) for j in audit['source_joints']}
pairs=[(i,j) for i,j in itertools.combinations(range(len(shapes)),2) if shapes[i]['body']!=shapes[j]['body'] and frozenset((shapes[i]['body'],shapes[j]['body'])) not in adj]
results=[]
for phase in ['corrected_holdout','grasp']:
    snap=json.loads((ROOT/'22_explicit_gain_sweep'/phase/'worker_0/runtime_inputs.json').read_text())
    P=len(snap['poses']);indices=range(P) if phase!='grasp' else range(0,P,3)
    for k in indices:
        root=pose_matrix(snap['initial_root_pose_world'][k][:3],snap['initial_root_pose_world'][k][3:]);inverse=np.linalg.inv(root)
        transforms={n:inverse@pose_matrix(p[:3],p[3:]) for n,p in zip(snap['body_names'],snap['initial_body_poses_world'][k])}
        for shape,obj in zip(shapes,objects):set_transform(obj,transforms[shape['body']])
        near=[]
        for i,j in pairs:
            gap=exact_pair_gap(objects[i],objects[j],signed=True)
            if gap<.001:near.append(dict(kind='self',a=shapes[i]['name'],b=shapes[j]['name'],gap_m=gap))
        for i,shape in enumerate(shapes):
            for name,obj in world.items():
                if shape['body']=='base_link' and name=='lab_18':continue
                gap=exact_pair_gap(objects[i],obj,signed=True)
                if gap<.001:near.append(dict(kind='world',a=shape['name'],b=name,gap_m=gap))
        if phase=='grasp':
            p=snap['initial_cube_poses_world'][k];T=inverse@pose_matrix(p[:3],p[3:]);cube=fcl.CollisionObject(fcl.Box(.06,.06,.06),fcl.Transform(T[:3,:3],T[:3,3]))
            for i,shape in enumerate(shapes):
                gap=exact_pair_gap(objects[i],cube,signed=True)
                if gap<.001:near.append(dict(kind='cube',a=shape['name'],b='cube',gap_m=gap,expected_grasp=shape['body'] in ['left_inner_finger','right_inner_finger']))
        row=dict(phase=phase,pose_index=k,grasp_posture_index=k//3 if phase=='grasp' else None,pairs_within_1mm=near)
        results.append(row)
        bad=[x for x in near if x['gap_m']<0 and not x.get('expected_grasp')]
        if bad:print(phase,k,bad,flush=True)
(OUT/'initial_geometry_contacts.json').write_text(json.dumps(dict(
    scope='Initial saved native body transforms; all current source convex hulls and lab boxes. Includes actual saved finger positions, not open-hand approximation. Source hulls can differ slightly from cooked PhysX hulls; overlap is a candidate contact, not a measured force. Direct joint adjacency and base/mounting-plate pair omitted; intended pad/cube contacts identified separately.',results=results),indent=2)+'\n')
