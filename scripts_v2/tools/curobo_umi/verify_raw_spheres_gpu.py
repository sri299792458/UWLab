"""Verify the raw-fit candidate against cuRobo on the saved comparison sample."""
import hashlib
import json

import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from reachability_geometry import ROOT, SourceCollisionChecker, batch_fk, exact_pair_gap, set_transform

root=ROOT/'20_dense_reachability'
candidate=root/'robot_open_raw_fit_candidate.yml'
config=yaml.safe_load(candidate.read_text())['robot_cfg']
data=np.load(root/'raw_sphere_pair_attribution.npz')
q=data['q']
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
    robot_config=config,scene_model=str(ROOT/'15_table_reachability/model/lab_scene.yml'),
    collision_activation_distance=0.,self_collision_activation_distance=0.))
states=checker.get_kinematics(torch.as_tensor(q,device='cuda',dtype=torch.float32)[:,None])
spheres=states.robot_spheres.clone();spheres[...,3]+=.001
self_bad=(checker.get_self_collision_distance(spheres).reshape(len(q),-1).amax(dim=1)>0).cpu().numpy()
base_indices=checker.kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link_shape_0')
world_spheres=spheres.clone();world_spheres[:,:,base_indices,3]=-100.
world_bad=(checker.get_collision_constraint(world_spheres).reshape(len(q),-1).amax(dim=1)>0).cpu().numpy()
sphere_clear=~self_bad&~world_bad
assert np.array_equal(self_bad,data['self_rejected'])
assert np.array_equal(world_bad,data['world_rejected'])
source=SourceCollisionChecker(ROOT/'15_table_reachability/model')
disagreements=[]
for row in np.flatnonzero(sphere_clear&~data['hull_clear']):
    result=source.evaluate(q[row:row+1],details=True)
    fail=result['first_failure'][0]
    poses=batch_fk(source.audit,q[row:row+1])
    a=next(i for i,s in enumerate(source.shapes) if s['name']==fail['a'])
    set_transform(source.objects[a],poses[source.shapes[a]['body']][0])
    if fail['kind']=='self':
        b=next(i for i,s in enumerate(source.shapes) if s['name']==fail['b'])
        set_transform(source.objects[b],poses[source.shapes[b]['body']][0])
        other=source.objects[b]
    else:
        other=source.world_objects[source.world_names.index(fail['b'])]
    gap=exact_pair_gap(source.objects[a],other,signed=True)
    disagreements.append(dict(row=int(row),q=q[row].tolist(),first_hull_failure=fail,signed_hull_gap_m=gap))
report=dict(sample_count=len(q),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    gpu_cpu_self_disagreements=int((self_bad!=data['self_rejected']).sum()),
    gpu_cpu_world_disagreements=int((world_bad!=data['world_rejected']).sum()),
    sphere_clear_count=int(sphere_clear.sum()),hull_clear_accepted=int((sphere_clear&data['hull_clear']).sum()),
    hull_rejected_but_sphere_accepted=disagreements,
    scope='Native cuRobo collision query only; one fixed saved sample; no mapping or controller run.')
(root/'raw_sphere_gpu_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
