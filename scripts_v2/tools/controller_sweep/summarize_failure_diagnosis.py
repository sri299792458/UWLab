"""Combine native contact interventions with saved kinematic diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911');OUT=ROOT/'24_gain_failure_diagnosis'
metrics=json.loads((OUT/'trajectory_metrics.json').read_text())['rows']
contacts=json.loads((OUT/'native_contact_diagnosis.json').read_text())
plan=json.loads((ROOT/'22_explicit_gain_sweep/plan_expanded.json').read_text());limits=plan['ranking_limits']
names=[('holdout_A','corrected_holdout',292),('holdout_B','corrected_holdout',276),('grasp_A','grasp',292),('grasp_B','grasp',276)]
results=[];traces={}
def passed(row):
    m=row['per_pose'];good=np.array(m['finite'],bool)&~np.array(m['abnormal'],bool)
    for k,cutoff in limits.items():good&=np.array(m[k])<=cutoff
    return good
for name,phase,gain in names:
    folder=OUT/name/'worker_0';counter=OUT/(name+'_without_robot_obstacles')/'worker_0'
    snap=json.loads((folder/'runtime_inputs.json').read_text());csnap=json.loads((counter/'runtime_inputs.json').read_text())
    s=json.loads((folder/'summary.json').read_text());c=json.loads((counter/'summary.json').read_text())
    for k in ['initial_joint_positions','initial_root_pose_world','masses_kg','inertias_kg_m2','gains','poses','joint_armature','joint_stiffness','joint_damping','robot_usd_sha256','controller_sha256']:
        assert snap[k]==csnap[k],(name,k)
    f=torch.load(folder/'trajectories.pt',map_location='cpu',weights_only=True);P=f['pose_count'];a=f['values'].numpy()[:,:P]
    b=torch.load(counter/'trajectories.pt',map_location='cpu',weights_only=True)['values'].numpy()[:,:P]
    cf=torch.load(counter/'contact_trajectories.pt',map_location='cpu',weights_only=True)
    eligible=[i for i,n in enumerate(cf['contact_body_names']) if phase!='grasp' or n not in ['left_inner_finger','right_inner_finger']]
    cfmax=float(cf['contact_net_forces_world'][:,:P,eligible].norm(dim=-1).max())
    assert cfmax<1e-7,(name,cfmax)
    rr=[x for x in metrics if x['phase']==phase and x['gain_id']==gain]
    cc=[x for x in contacts['rows'] if x['phase']==phase and x['gain_id']==gain]
    buckets={'contact_observed':[],'no_contact_nonquiet':[],'no_contact_tracking_error':[]}
    for x in rr:
        if x['passed']:continue
        contact=next(v for v in cc if v['pose_index']==x['pose_index'])
        group='contact_observed' if contact['non_grasp_contact_peak_n']>.01 else ('no_contact_nonquiet' if x['final_wrist_speed_rms_rad_s']>.1 else 'no_contact_tracking_error')
        buckets[group].append(x['pose_index'])
    good,other=passed(s['results'][0]),passed(c['results'][0]);had_contact=np.array([x['non_grasp_contact_peak_n']>.01 for x in cc])
    velocity_limits=np.array(csnap['joint_velocity_limits_rad_s'][:6]);near_limit=np.any(np.abs(a[:,:,12:18])>=.999*velocity_limits,axis=-1)
    hf_motion=np.sqrt(np.mean(np.sum((.5*np.diff(a[240:960,:,15:18].astype(float),axis=0))**2,axis=-1),axis=0))
    result=dict(name=name,phase=phase,gain_id=gain,count=P,passed=int(good.sum()),failed=int((~good).sum()),
        failure_groups=buckets,contact_observed_pose_indices=np.flatnonzero(had_contact).tolist(),
        counterfactual=dict(passed=int(other.sum()),failed=int((~other).sum()),
            previously_failed_now_passed=np.flatnonzero(~good&other).tolist(),previously_passed_now_failed=np.flatnonzero(good&~other).tolist(),
            failed_with_contact_now_passed=np.flatnonzero(~good&had_contact&other).tolist(),
            no_contact_maximum_arm_state_change=float(np.max(np.abs((a[:,:,6:18]-b[:,:,6:18])[:,~had_contact]))) if (~had_contact).any() else None,
            remaining_robot_contact_max_n=cfmax,grasp_passed=c['results'][0].get('grasp_passed')),
        maximum_torque_fraction=max(max(x['torque_utilization_by_joint']) for x in rr),
        minimum_joint_position_limit_margin_rad=min(x['minimum_joint_limit_margin_rad'] for x in rr),
        final_high_frequency_max_rad_s=max(x['final_half_step_difference_rms_rad_s'] for x in rr),
        nonquiet_count=sum(x['final_wrist_speed_rms_rad_s']>.1 for x in rr),
        nonquiet_with_90pct_weak_direction=sum(x['final_wrist_speed_rms_rad_s']>.1 and x['final_least_stiffness_direction_velocity_fraction']>.9 for x in rr),
        velocity_limits_rad_s=velocity_limits.tolist(),
        near_velocity_limit_pose_count=int(near_limit[240:960].any(axis=0).sum()),
        near_velocity_limit_motion_step_fraction_median=float(np.median(near_limit[240:960].mean(axis=0))),
        near_velocity_limit_by_action_substep=np.mean(near_limit[240:960].reshape(60,12,P),axis=(0,2)).tolist(),
        motion_high_frequency_median_rad_s=float(np.median(hf_motion)),motion_high_frequency_max_rad_s=float(np.max(hf_motion)))
    results.append(result);traces[name]=(a,b)
    print(json.dumps(result,indent=2),flush=True)
reference=json.loads((OUT/'reference_path_audit.json').read_text())
reference_summary={}
for phase in ['corrected_holdout','grasp']:
    rows=[x for x in reference['results'] if x['phase']==phase]
    reference_summary[phase]=dict(postures=len(rows),all_targets_solved=sum(x['all_100_targets_solved'] for x in rows),
        solved_with_sampled_1mm_clearance=sum(x['found_source_hull_1mm_clear_reference'] for x in rows),
        solved_clear_with_consecutive_solution_speed_within_limits=sum(
            x['found_source_hull_1mm_clear_reference'] and x['reference_maximum_joint_speed_limit_ratio']<=1 for x in rows))
(OUT/'comparison.json').write_text(json.dumps(dict(
    status='Failure analysis complete. Counterfactual contact-disabled trajectories are diagnostic only; no production gain, model, controller, or timing changes.',
    original_replay_validation=contacts['validation'],results=results,
    reference_path_summary=reference_summary,reference_path_method=reference['method']),indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#b06949','#487d9b','#ad9d64']
labels=['Contact observed','No contact; wrist not settled','No contact; tracking error only'];bottom=np.zeros(4)
for key,label,color in zip(['contact_observed','no_contact_nonquiet','no_contact_tracking_error'],labels,colors):
    h=np.array([len(r['failure_groups'][key]) for r in results]);axs[0,0].bar(range(4),h,bottom=bottom,label=label,color=color);bottom+=h
for i,n in enumerate(bottom):axs[0,0].text(i,n+.5,str(int(n)),ha='center')
axs[0,0].set(title='Observed failure groups in the original tests',ylabel='Failed trials',ylim=(0,46),xticks=range(4),xticklabels=['A · open /80','B · open /80','A · grasp /72','B · grasp /72'])
axs[0,0].legend(frameon=False,fontsize=8,loc='upper left')
data=np.load(OUT/'trajectory_analysis.npz');a=traces['grasp_A'][0];t=(np.arange(1200)+1)/120
ax=axs[0,1];sl=slice(960,1200);k=51
ax.plot(t[sl],a[sl,k,15],label='Wrist 1',color='#487d9b');ax.plot(t[sl],a[sl,k,17],label='Wrist 3',color='#b06949')
angular=np.linalg.norm(data['grasp_292__task_velocity'][sl,k,3:],axis=-1)
ax.plot(t[sl],angular,label='Hand angular speed',color='#23846a',linewidth=2)
ax.set(title='A: opposing wrist motion (grasp posture 17, 20 g)',xlabel='Time (s)',ylabel='Angular velocity / speed (rad/s)');ax.legend(frameon=False,fontsize=8)
ax=axs[1,0];a,b=traces['holdout_A'];k=49
ax.plot(t,np.linalg.norm(a[:,k,:3],axis=-1)*1000,label='Normal contacts',color='#b06949')
ax.plot(t,np.linalg.norm(b[:,k,:3],axis=-1)*1000,label='Obstacle contacts disabled',color='#23846a')
ax.axvspan(8,10,color='#dce8ee',alpha=.5);ax.set(title='A: isolate the mounting-plate obstruction (pose 49)',xlabel='Time (s)',ylabel='Hand position error (mm)');ax.legend(frameon=False,fontsize=8)
ax=axs[1,1];a=traces['holdout_A'][0];sl=(t>=4)&(t<=4.5);k=0
for at in np.arange(4,4.51,.1):ax.axvline(at,color='#c4c9cc',lw=1)
ax.plot(t[sl],a[sl,k,17],'.-',color='#487d9b',markersize=3,label='Recorded wrist 3 velocity')
for sign in [-1,1]:ax.axhline(sign*3.1415,color='#b06949',ls='--',lw=1)
ax.set(title='A: velocity bursts after 10 Hz target updates',xlabel='Time (s)',ylabel='Wrist 3 velocity (rad/s)')
fig.suptitle('Why A/B still fail · unchanged UMI dynamics · 120 Hz feedback',fontsize=15)
fig.savefig(OUT/'failure_diagnosis.png',dpi=170);fig.savefig(OUT/'failure_diagnosis.svg')
