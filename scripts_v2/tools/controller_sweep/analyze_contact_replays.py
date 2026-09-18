"""Validate matched replays and attribute observed native contact forces."""
import itertools,json,sys
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911');OUT=ROOT/'24_gain_failure_diagnosis'
sys.path.insert(0,str(Path(__file__).parents[1]/'curobo_umi'))
from reachability_geometry import convex_geometry,set_transform,exact_pair_gap,pose_matrix,fcl
MODEL=ROOT/'15_table_reachability/model';audit=json.loads((MODEL/'export_audit.json').read_text())
shapes=audit['collision_shapes'];objects=[fcl.CollisionObject(convex_geometry(s['mesh'])[0]) for s in shapes]
scene=yaml.safe_load((MODEL/'lab_scene.yml').read_text())['cuboid']
world={n:fcl.CollisionObject(fcl.Box(*v['dims']),fcl.Transform(pose_matrix(v['pose'][:3],v['pose'][3:])[:3,:3],v['pose'][:3])) for n,v in scene.items()}
adj={frozenset((j['body0'],j['body1'])) for j in audit['source_joints']}
metrics=json.loads((OUT/'trajectory_metrics.json').read_text())['rows'];records=[];validation=[]
for name,phase,gain,old_worker in [('holdout_A','corrected_holdout',292,0),('holdout_B','corrected_holdout',276,1),('grasp_A','grasp',292,0),('grasp_B','grasp',276,1)]:
    folder=OUT/name/'worker_0';snap=json.loads((folder/'runtime_inputs.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    oldfolder=ROOT/'22_explicit_gain_sweep'/phase/f'worker_{old_worker}';oldsnap=json.loads((oldfolder/'runtime_inputs.json').read_text());oldsum=json.loads((oldfolder/'summary.json').read_text())
    for key in ['masses_kg','inertias_kg_m2','joint_armature','joint_stiffness','joint_damping','initial_joint_positions','initial_root_pose_world','robot_usd_sha256','controller_sha256','physics_hz','decimation','gains','poses']:
        assert snap[key]==oldsnap[key],(name,key)
    f=torch.load(folder/'trajectories.pt',map_location='cpu',weights_only=True);a=f['values'].numpy()[:,:f['pose_count']]
    old=torch.load(oldfolder/'trajectories.pt',map_location='cpu',weights_only=True)['values'].numpy()[:,:f['pose_count']]
    d=torch.load(folder/'contact_trajectories.pt',map_location='cpu',weights_only=True);P=d['pose_count']
    force=d['contact_net_forces_world'].numpy()[:,:P];magnitude=np.linalg.norm(force,axis=-1)
    body=d['body_poses_world'].numpy()[:,:P];cn=d['contact_body_names'];bn=d['body_names']
    # Expected finger/cube contact does not diagnose an obstruction. All other
    # native robot contacts remain eligible, including arm/hand self-contact.
    eligible=[i for i,n in enumerate(cn) if phase!='grasp' or n not in ['left_inner_finger','right_inner_finger']]
    eligible_forces=magnitude[:,:,eligible];peak=np.max(eligible_forces,axis=(0,2))
    validation.append(dict(replay=name,all_initial_states_properties_gains_and_source_hashes_equal=True,
        original_passed=oldsum['results'][0]['passed'],replay_passed=summary['results'][0]['passed'],
        arm_position_max_abs_trajectory_delta_rad=float(np.max(np.abs(a[:,:,6:12]-old[:,:,6:12]))),
        arm_velocity_max_abs_trajectory_delta_rad_s=float(np.max(np.abs(a[:,:,12:18]-old[:,:,12:18]))),
        target_error_max_abs_delta=float(np.max(np.abs(a[:,:,:6]-old[:,:,:6])))))
    for k in range(P):
        metric=next(x for x in metrics if x['phase']==phase and x['gain_id']==gain and x['pose_index']==k)
        active=np.max(eligible_forces[:,k],axis=-1)>.01
        row=dict(replay=name,phase=phase,gain_id=gain,pose_index=k,grasp_posture_index=k//3 if phase=='grasp' else None,
            controller_passed=metric['passed'],non_grasp_contact_peak_n=float(peak[k]),
            non_grasp_contact_step_fraction=float(np.mean(active)),
            first_contact_time_s=float((np.flatnonzero(active)[0]+1)/120) if active.any() else None,
            contact_bodies={n:float(np.max(magnitude[:,k,i])) for i,n in enumerate(cn) if i in eligible and np.max(magnitude[:,k,i])>.01})
        if peak[k]>.01:
            ti,ci=np.unravel_index(np.argmax(eligible_forces[:,k]),(1200,len(eligible)));contact_body=cn[eligible[ci]]
            inverse=np.linalg.inv(pose_matrix(snap['initial_root_pose_world'][k][:3],snap['initial_root_pose_world'][k][3:]))
            transforms={n:inverse@pose_matrix(p[:3],p[3:]) for n,p in zip(bn,body[ti,k])}
            for shape,obj in zip(shapes,objects):set_transform(obj,transforms[shape['body']])
            near=[]
            for i,shape in enumerate(shapes):
                if shape['body']!=contact_body:continue
                for j,other in enumerate(shapes):
                    if shape['body']==other['body'] or frozenset((shape['body'],other['body'])) in adj:continue
                    gap=exact_pair_gap(objects[i],objects[j],signed=True)
                    if gap<.003:near.append(dict(kind='self',a=shape['name'],b=other['name'],gap_m=gap))
                for other,obj in world.items():
                    if shape['body']=='base_link' and other=='lab_18':continue
                    gap=exact_pair_gap(objects[i],obj,signed=True)
                    if gap<.003:near.append(dict(kind='world',a=shape['name'],b=other,gap_m=gap))
            row.update(peak_contact_time_s=(ti+1)/120,peak_contact_body=contact_body,source_geometry_near_peak_contact=sorted(near,key=lambda x:x['gap_m']))
            if not metric['passed'] and (phase!='grasp' or k%3==0):print(name,k,row['contact_bodies'],row['source_geometry_near_peak_contact'],flush=True)
        records.append(row)
    print(name,'replay validation',validation[-1],flush=True)
(OUT/'native_contact_diagnosis.json').write_text(json.dumps(dict(
    scope='Native net contact forces on every robot body during exact A/B protocol replays. For grasp trials, expected left/right inner-finger contacts are excluded from obstruction counts. Threshold for reported contact is 0.01 N. Nearest source-hull pairs at peak force attribute likely geometry; they do not replace native contact-force evidence.',
    validation=validation,rows=records),indent=2)+'\n')
