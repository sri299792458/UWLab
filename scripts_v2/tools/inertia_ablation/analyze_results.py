"""Summarize recorded native physics; figures contain measured trajectories."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/23_inertia_gap')
NAMES=['stock_factorial','umi_factorial','grasp_factorial','umi_confirm',
       'stock_confirm_complete','loaded_residual_complete']
data={};traces={};validation={}
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
for name in NAMES:
    p=ROOT/name/'worker_0';s=json.loads((p/'summary.json').read_text())
    snap=json.loads((p/'runtime_inputs.json').read_text())
    interventions=json.loads((p/'inertial_intervention.json').read_text())
    plan=json.loads(Path(s['plan_path']).read_text())
    assert hashlib.sha256(Path(s['plan_path']).read_bytes()).hexdigest()==s['plan_sha256']
    assert hashlib.sha256((p/'run_gain_sweep.py').read_bytes()).hexdigest()==s['script_sha256']
    for path,digest in plan['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for key in ['kp','kd']:
        for row in s['results']:assert np.array_equal(row[key],s['results'][0][key])
        assert np.allclose(s['results'][0][key],([200.]*3+[3.]*3) if key=='kp' else ([84.8528137423857]*3+[3.4641016151377544]*3))
    f=torch.load(p/'trajectories.pt',map_location='cpu',weights_only=False)
    v=f['values'].numpy();P=f['pose_count'];G=len(s['results'])
    assert v.shape[:2]==(1200,P*G)
    q=np.array(snap['initial_joint_positions']).reshape(G,P,-1)
    assert np.max(np.abs(q-q[:1]))<1e-6
    body=np.array(snap['initial_body_poses_world']).reshape(G,P,-1,7)
    root=np.array(snap['initial_root_pose_world']).reshape(G,P,7)
    body_relative=body[...,:3]-root[:,:,None,:3]
    body_delta=np.max(np.abs(body_relative-body_relative[:1]))
    quat_agreement=np.abs(np.sum(body[...,3:]*body[:1,...,3:],axis=-1))
    validation[name]=dict(controller_formula_max_error_nm=s['controller_formula_max_error_nm'],
        initial_joint_case_max_delta_rad=float(np.max(np.abs(q-q[:1]))),
        initial_body_case_max_position_delta_m=float(body_delta),
        initial_body_case_max_quaternion_dot_error=float(np.max(np.abs(quat_agreement-1))),
        body_property_readback=interventions['verification'],
        current_input_hashes_match=True,archived_runner_hash_matches=True)
    hf=np.sqrt(np.mean(np.sum((.5*np.diff(v[1080:,:,15:18].astype(float),axis=0))**2,axis=-1),axis=0))
    rows=[]
    for j,row in enumerate(s['results']):
        sl=slice(j*P,(j+1)*P);m=row['per_pose'];finite=np.array(m['finite'],bool)
        speed=np.array(m['final_wrist_speed_rms_rad_s'],float);h=hf[sl]
        quiet=finite&~np.array(m['abnormal'],bool)&(speed<=.1)
        retained=None
        if 'cube_relative_error_max_m' in m:
            retained=finite&(np.array(m['cube_relative_error_max_m'],float)<.015)&(np.array(m['cube_rotation_error_max_rad'],float)<np.deg2rad(5))&(np.array(m['bilateral_contact_fraction'],float)>=.8)
        rows.append(dict(id=row['id'],label=row['label'],protocol=row['protocol'],count=P,
            finite=int(finite.sum()),abnormal=int(np.array(m['abnormal'],bool).sum()),
            quiet_count=int(quiet.sum()),rapid_oscillation_count=int((finite&(h>1)).sum()),
            finite_nonquiet_pose_indices=np.flatnonzero(finite&~quiet).tolist(),nonfinite_pose_indices=np.flatnonzero(~finite).tolist(),
            final_wrist_speed_rms_rad_s=speed,final_high_frequency_wrist_rms_rad_s=h,
            final_wrist_speed_median_rad_s=np.median(speed[finite]),final_wrist_speed_max_rad_s=np.max(speed[finite]),
            final_high_frequency_median_rad_s=np.median(h[finite]),final_high_frequency_max_rad_s=np.max(h[finite]),
            cube_retained_count=int(retained.sum()) if retained is not None else None,
            airborne_grasp_pass_count=row.get('grasp_passed'),
            rapid_by_mass={str(mass):int(np.sum(finite&(h>1)&(np.array([p.get('cube_mass_kg') for p in snap['poses']])==mass))) for mass in [.02,.1,.2]} if retained is not None else None))
    data[name]=rows
    if name in ['umi_confirm','loaded_residual_complete']:
        traces[name]=dict(values=v,pose_count=P)
report=dict(measurements=dict(quiet='Finite, no abnormal speed, final 1 s RMS norm of three wrist joint velocities <= 0.1 rad/s.',
    high_frequency='Final 1 s RMS norm of 0.5*(wrist_velocity[t+1]-wrist_velocity[t]); 120 Hz samples. The >1 rad/s flag detects the large alternating-step oscillation, not all undesirable motion.',
    medians='Usual average of middle two values; finite trials only, with failures reported separately.'),results=data,validation=validation)
(ROOT/'comparison.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
(ROOT/'validation.json').write_text(json.dumps(clean(validation),indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(13.5,9.2),layout='constrained')
colors=['#4e7188','#dc9146','#6e9b8e','#217a5c']
def merged(names,ids):
    return [np.concatenate([np.array(next(x for x in data[n] if x['id']==i)['final_wrist_speed_rms_rad_s'],float) for n,i in pairs]) for pairs in ids]
ax=axs[0,0]
free=merged(None,[[('umi_factorial',a),('umi_confirm',b)] for a,b in [(0,0),(8,2),(4,1),(12,3)]])
ax.bar(range(4),[np.nanmedian(x) for x in free],color=colors)
ax.axhline(.1,color='#6d6d6d',linestyle='--',linewidth=1)
ax.set(yscale='log',ylim=(.0003,9),ylabel='Final wrist speed RMS (rad/s)',title='UMI: both inertia restorations are needed',xticks=range(4),xticklabels=['Nominal','Knuckles','Housing','Both'])
for j,x in enumerate(free):ax.text(j,np.nanmedian(x)*1.3,f'{int(np.sum(x<=.1))}/96 quiet',ha='center',fontsize=9)
ax.text(.02,.03,'Moving targets; 16 screen + 80 reserved poses',transform=ax.transAxes,fontsize=9)
ax=axs[0,1]
stock=merged(None,[[('stock_factorial',i),('stock_confirm_complete',i)] for i in [0,2,1,3]])
ax.bar(range(4),[np.nanmedian(x) for x in stock],color=colors)
ax.axhline(.1,color='#6d6d6d',linestyle='--',linewidth=1)
ax.set(yscale='log',ylim=(.0003,9),ylabel='Final wrist speed RMS (rad/s)',title='Reverse test: lowering housing inertia destabilizes stock',xticks=range(4),xticklabels=['Stock','Lower\nknuckles','Lower\nhousing','Lower both'])
for j,x in enumerate(stock):ax.text(j,np.nanmedian(x)*1.3,f'{int(np.sum(x<=.1))}/96 quiet',ha='center',fontsize=9)
ax.text(.02,.03,'Medians shown; divergent trials remain failures',transform=ax.transAxes,fontsize=9)
ax=axs[1,0];r=data['loaded_residual_complete'];chosen=[0,1,2,4]
counts=[r[i]['rapid_oscillation_count'] for i in chosen]
ax.bar(range(4),counts,color=colors);ax.set(ylim=(0,75),ylabel='Trials with rapid oscillation / 72',title='Grasp: mass distribution supplies the remaining margin',xticks=range(4),xticklabels=['Both inertias','+ Housing\nmass','+ Housing\nCOM','+ Knuckle\nmasses'])
for j,x in enumerate(counts):ax.text(j,x+2,str(x),ha='center')
ax.text(.98,.95,'24 poses × 3 cube masses\nAll four retain 72/72 cubes',transform=ax.transAxes,ha='right',va='top',fontsize=9)
ax=axs[1,1];v=traces['loaded_residual_complete']['values'];P=traces['loaded_residual_complete']['pose_count'];k=3
t=(np.arange(1140,1177)+1)/120
for j,label,color in [(0,'Both inertias',colors[1]),(2,'+ Housing COM',colors[3])]:
    ax.plot(t,v[1140:1177,j*P+k,17],'.-',label=label,color=color,linewidth=1,markersize=3)
ax.set(title='Recorded wrist 3 velocity: same 20 g grasp',xlabel='Simulation time (s)',ylabel='Joint velocity (rad/s)');ax.legend(frameon=False,loc='upper right')
fig.suptitle('Fixed default gains · 120 Hz physics and feedback · decimation 12',fontsize=15)
fig.savefig(ROOT/'inertia_gap.png',dpi=170);fig.savefig(ROOT/'inertia_gap.svg')
print(json.dumps(clean({name:[{k:r[k] for k in ['id','quiet_count','rapid_oscillation_count','finite']} for r in rows] for name,rows in data.items()}),indent=2))
