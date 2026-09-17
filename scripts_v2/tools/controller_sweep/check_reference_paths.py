"""Follow commanded poses with a local IK solve, then inspect source hulls.

Successful solves establish a candidate reference path. Failed solves do not
prove a target globally unreachable; this diagnostic follows the starting branch.
"""
import itertools,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911');OUT=ROOT/'24_gain_failure_diagnosis'
MODEL=ROOT/'15_table_reachability/model';audit=json.loads((MODEL/'export_audit.json').read_text())
sys.path.insert(0,str(Path(__file__).parents[1]/'curobo_umi'))
from reachability_geometry import convex_geometry,set_transform,exact_pair_gap,pose_matrix,fcl
arm_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
arm_tree=[j for j in audit['articulation_tree'] if j['name'] in arm_names]
limits=np.array([[next(j for j in arm_tree if j['name']==n)[key] for n in arm_names] for key in ['lower','upper']])
def fk(q,full=False,joint_names=None):
    q=np.asarray(q);N=len(q);T={'base_link':np.broadcast_to(np.eye(4),(N,4,4)).copy()};origins=[];axes=[]
    for j in (audit['articulation_tree'] if full else arm_tree):
        a,b,sign=np.array(j['T0']),np.array(j['T1']),1
        if j['reverse']:a,b,sign=b,a,-1
        frame=T[j['parent']]@a;turn=np.broadcast_to(np.eye(4),(N,4,4)).copy()
        if not j['fixed']:
            axis=np.eye(3)['XYZ'.index(j['axis'])]*sign
            index=(joint_names if full else arm_names).index(j['name'])
            turn[:,:3,:3]=Rotation.from_rotvec(q[:,index,None]*axis).as_matrix()
            if j['name'] in arm_names:origins.append(frame[:,:3,3]);axes.append(np.einsum('nij,j->ni',frame[:,:3,:3],axis))
        T[j['child']]=frame@turn@np.linalg.inv(b)
    p=T['wrist_3_link'][:,:3,3];axis=np.stack(axes,axis=-1);origin=np.stack(origins,axis=-1)
    J=np.concatenate([np.cross(axis.transpose(0,2,1),(p[:,:,None]-origin).transpose(0,2,1)).transpose(0,2,1),axis],axis=1)
    return T,J

inputs=[];all_q=[];full_q=[];p0=[];R0=[]
for phase in ['corrected_holdout','grasp']:
    snap=json.loads((ROOT/'22_explicit_gain_sweep'/phase/'worker_0/runtime_inputs.json').read_text())
    indices=range(len(snap['poses'])) if phase!='grasp' else range(0,len(snap['poses']),3)
    for k in indices:
        root=pose_matrix(snap['initial_root_pose_world'][k][:3],snap['initial_root_pose_world'][k][3:])
        p=snap['initial_body_poses_world'][k][snap['body_names'].index('wrist_3_link')];T=np.linalg.inv(root)@pose_matrix(p[:3],p[3:])
        all_q.append(snap['poses'][k]['q']);full_q.append(snap['initial_joint_positions'][k]);p0.append(T[:3,3]);R0.append(T[:3,:3])
        inputs.append(dict(phase=phase,pose_index=k,grasp_posture_index=k//3 if phase=='grasp' else None))
joint_names=snap['joint_names'];q=np.array(all_q);full_q=np.array(full_q);p0=np.array(p0);R0=np.array(R0);N=len(q)
T,J=fk(q);initial_p_error=np.max(np.linalg.norm(T['wrist_3_link'][:,:3,3]-p0,axis=-1));initial_r_error=np.max(Rotation.from_matrix(R0@T['wrist_3_link'][:,:3,:3].transpose(0,2,1)).magnitude())
assert initial_p_error<2e-5 and initial_r_error<1e-4,(initial_p_error,initial_r_error)
qs=[];positions=[];angles=[];started=time.perf_counter();scales=np.array([1.,1.,1.,.1,.1,.1])
for step in range(100):
    t=step/10;phase=np.clip(t-2,0,6);window=np.sin(np.pi*phase/6)**2 if 2<=t<8 else 0.
    wave=np.sin(2*np.pi*.5*phase+np.arange(3)*2*np.pi/3);pd=p0+.02*window*wave;Rd=Rotation.from_rotvec(.1*window*wave).as_matrix()@R0
    for iteration in range(100):
        T,J=fk(q);p=T['wrist_3_link'][:,:3,3];R=T['wrist_3_link'][:,:3,:3]
        er=Rotation.from_matrix(Rd@R.transpose(0,2,1)).as_rotvec();ep=pd-p
        done=(np.linalg.norm(ep,axis=-1)<1e-5)&(np.linalg.norm(er,axis=-1)<1e-4)
        if done.all():break
        e=np.concatenate([ep,er],axis=1)*scales;Jw=J*scales[None,:,None]
        system=np.einsum('nri,nrj->nij',Jw,Jw)+1e-6*np.eye(6);rhs=np.einsum('nri,nr->ni',Jw,e)
        delta=np.linalg.solve(system,rhs[:,:,None]).squeeze(-1);delta*=np.minimum(1.,.1/np.linalg.norm(delta,axis=-1).clip(1e-20))[:,None];delta[done]=0
        q=np.clip(q+delta,limits[0],limits[1])
    T,J=fk(q);ep=np.linalg.norm(pd-T['wrist_3_link'][:,:3,3],axis=-1);er=Rotation.from_matrix(Rd@T['wrist_3_link'][:,:3,:3].transpose(0,2,1)).magnitude()
    qs.append(q.copy());positions.append(ep);angles.append(er)
qs=np.array(qs);positions=np.array(positions);angles=np.array(angles)
np.savez_compressed(OUT/'reference_ik_paths.npz',joint_positions=qs,position_error_m=positions,rotation_error_rad=angles)
print('IK complete',round(time.perf_counter()-started,2),'s',flush=True)

shapes=audit['collision_shapes'];objects=[];bounds=[]
for shape in shapes:
    geometry,b=convex_geometry(shape['mesh']);objects.append(fcl.CollisionObject(geometry));bounds.append(b)
bounds=np.array(bounds);local_c=bounds.mean(axis=1);local_e=(bounds[:,1]-bounds[:,0])/2
scene=yaml.safe_load((MODEL/'lab_scene.yml').read_text())['cuboid'];wn=list(scene);world=[];wc=[];we=[]
for n,v in scene.items():
    T=pose_matrix(v['pose'][:3],v['pose'][3:]);world.append(fcl.CollisionObject(fcl.Box(*v['dims']),fcl.Transform(T[:3,:3],T[:3,3])));wc.append(T[:3,3]);we.append(np.abs(T[:3,:3])@(np.array(v['dims'])/2))
wc,we=np.array(wc),np.array(we)
adj={frozenset((j['body0'],j['body1'])) for j in audit['source_joints']}
pairs=np.array([(i,j) for i,j in itertools.combinations(range(len(shapes)),2) if shapes[i]['body']!=shapes[j]['body'] and frozenset((shapes[i]['body'],shapes[j]['body'])) not in adj]);ia,ib=pairs.T
events=[[] for _ in inputs];speed=np.max(np.abs(np.diff(qs,axis=0))/.1/np.array([1.5708]*3+[3.1415]*3),axis=(0,2))
for step in range(100):
    v=full_q.copy();v[:,:6]=qs[step];T,_=fk(v,full=True,joint_names=joint_names);tf=np.stack([T[s['body']] for s in shapes],axis=1)
    centers=np.einsum('nbij,bj->nbi',tf[:,:,:3,:3],local_c)+tf[:,:,:3,3];extents=np.einsum('nbij,bj->nbi',np.abs(tf[:,:,:3,:3]),local_e)
    sw=(np.abs(centers[:,:,None]-wc[None,None])<=extents[:,:,None]+we[None,None]+.001).all(axis=-1)
    ss=(np.abs(centers[:,ia]-centers[:,ib])<=extents[:,ia]+extents[:,ib]+.001).all(axis=-1)
    for row in range(N):
        candidates=[]
        for ix in np.flatnonzero(ss[row]):
            i,j=pairs[ix];set_transform(objects[i],tf[row,i]);set_transform(objects[j],tf[row,j]);gap=exact_pair_gap(objects[i],objects[j],signed=True)
            if gap<.001:candidates.append(dict(kind='self',a=shapes[i]['name'],b=shapes[j]['name'],gap_m=gap))
        for i,j in np.argwhere(sw[row]):
            if shapes[i]['body']=='base_link' and wn[j]=='lab_18':continue
            set_transform(objects[i],tf[row,i]);gap=exact_pair_gap(objects[i],world[j],signed=True)
            if gap<.001:candidates.append(dict(kind='world',a=shapes[i]['name'],b=wn[j],gap_m=gap))
        if candidates:events[row].append(dict(policy_step=step,near_pairs=candidates))
results=[]
for i,row in enumerate(inputs):
    bad=np.flatnonzero((positions[:,i]>.0001)|(angles[:,i]>.001))
    result=dict(**row,maximum_position_error_m=float(positions[:,i].max()),maximum_rotation_error_rad=float(angles[:,i].max()),
        ik_failed_policy_steps=bad.tolist(),reference_maximum_joint_speed_limit_ratio=float(speed[i]),
        source_hull_clearance_events=events[i],all_100_targets_solved=len(bad)==0,
        found_source_hull_1mm_clear_reference=len(bad)==0 and not events[i])
    results.append(result)
(OUT/'reference_path_audit.json').write_text(json.dumps(dict(
    method='Local damped-least-squares IK follows each original starting branch through the 100 prescribed 10 Hz targets. Full calibrated USD articulation-tree FK/Jacobian, 0.1 m angular weighting, maximum 0.1 rad solve step. A target counts solved at <0.1 mm and <0.001 rad. Failed local solves are not proof of global unreachability. Robot source convex hulls are checked with each saved initial finger configuration frozen; cube trajectory clearance and exact cooked-hull clearance are not certified. Joint-speed ratios are consecutive target-solution differences /0.1 s, not actuator tracking predictions.',
    initial_native_position_error_m=float(initial_p_error),initial_native_rotation_error_rad=float(initial_r_error),
    elapsed_s=time.perf_counter()-started,results=results),indent=2)+'\n')
for phase in ['corrected_holdout','grasp']:
    rows=[x for x in results if x['phase']==phase]
    print(phase,'solved',sum(x['all_100_targets_solved'] for x in rows),'clear',sum(x['found_source_hull_1mm_clear_reference'] for x in rows),'/',len(rows),flush=True)
