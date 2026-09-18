"""Compare default-gain reference dynamics against the nominal-inertia gain sweep."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep')
conditions=[('Stock model + default gains','stock_reference_holdout',0),
    ('UMI + default gains + knuckle fallback inertia','fallback_holdout',0),
    ('UMI + default gains + nominal inertia','corrected_holdout',0),
    ('UMI + tuned gains + nominal inertia','corrected_holdout',292)]
metrics=['motion_position_rms_m','motion_rotation_rms_rad','final_position_rms_m',
    'final_rotation_rms_rad','final_wrist_speed_rms_rad_s']
rows=[]
raw=[]
for label,phase,gid in conditions:
    matches=[(f,r) for f in sorted((ROOT/phase).glob('worker_*/summary.json'))
        for r in json.loads(f.read_text())['results'] if r['id']==gid]
    assert len(matches)==1
    f,r=matches[0]
    assert r['count']==80
    snapshot=json.loads((f.parent/'runtime_inputs.json').read_text())
    summary=json.loads(f.read_text())
    assert summary['script_sha256']==hashlib.sha256((f.parent/'run_gain_sweep.py').read_bytes()).hexdigest()
    assert summary['controller_formula_max_error_nm']==0
    assert snapshot['physics_hz']==120 and snapshot['decimation']==12
    assert snapshot['poses']==json.loads((ROOT/'corrected_holdout/worker_0/runtime_inputs.json').read_text())['poses']
    values={k:np.asarray(r['per_pose'][k]) for k in metrics}
    stats={k:dict(median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(v.max())) for k,v in values.items()}
    rows.append(dict(condition=label,phase=phase,gain_id=gid,kp=r['kp'],kd=r['kd'],
        finite=r['finite'],prior_combined_screen_passed=r['passed'],
        final_wrist_speed_at_most_0_1_rad_s=int((values['final_wrist_speed_rms_rad_s']<=.1).sum()),
        metrics=stats,robot_usd=snapshot['robot_usd'],robot_usd_sha256=snapshot['robot_usd_sha256']))
    raw.append(values)
paired={k:dict(tuned_lower_than_stock=int((raw[-1][k]<raw[0][k]).sum()),
    median_tuned_minus_stock=float(np.median(raw[-1][k]-raw[0][k]))) for k in metrics}
loaded=[]
for label,phase,gid in [('UMI default + nominal inertia','grasp',0),('UMI default + knuckle fallback inertia','fallback_grasp',0),('UMI tuned + nominal inertia','grasp',292)]:
    r=next(r for f in sorted((ROOT/phase).glob('worker_*/summary.json')) for r in json.loads(f.read_text())['results'] if r['id']==gid)
    v=np.array(r['per_pose']['final_wrist_speed_rms_rad_s'])
    loaded.append(dict(condition=label,count=72,retained=r['grasp_passed'],
        final_wrist_speed_median_rad_s=float(np.median(v)),final_wrist_speed_maximum_rad_s=float(v.max()),
        final_wrist_speed_at_most_0_1_rad_s=int((v<=.1).sum())))
report=dict(scope='80 identical saved joint starts, common intended base pose, same 10 s base-frame relative wrist-target protocol, 120 Hz / decimation 12. The stock-model reference uses the complete original asset with original masses/inertias/geometry; it is distinct from changing only the two knuckle inertia tensors on UMI.',
    interpretation='Default-gain stock behavior is a necessary empirical reference. The former strict tracking cutoffs were not derived from it and are not a stand-alone definition of matching default performance. Tuned UMI has better median tracking but a worse worst-case settling speed in this cohort.',
    results=rows,paired_metrics=paired,loaded_umi_only=loaded)
(ROOT/'fallback_reference_comparison.json').write_text(json.dumps(report,indent=2)+'\n')

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
labels=['Stock / default','UMI / default / fallback I','UMI / default / nominal I','UMI / tuned / nominal I']
colors=['#285891','#b27d22','#a34d51','#258064']
for label,color,data in zip(labels,colors,raw):
    axes[0].step(np.sort(data['motion_position_rms_m'])*1000,np.arange(1,81)/80*100,where='post',label=label,color=color,lw=1.8)
    axes[1].step(np.sort(data['final_wrist_speed_rms_rad_s']),np.arange(1,81)/80*100,where='post',label=label,color=color,lw=1.8)
axes[0].set(title='Motion tracking error',xlabel='Position RMS (mm)',ylabel='Reserved poses at or below value (%)',ylim=(0,103))
axes[1].set(title='Wrist motion during the final hold',xlabel='RMS norm of wrist-joint speeds (rad/s)',xscale='log',ylim=(0,103))
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='lower right')
axes[1].legend(frameon=False,fontsize=8,loc='lower center')
fig.suptitle('Default-gain reference comparison · 120 Hz · decimation 12',fontsize=14)
fig.text(.5,.025,'Computed native simulation results. Stock and UMI have different geometry and dynamics; only the UMI inertia intervention isolates the two knuckle tensors.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.07,1,.93))
fig.savefig(ROOT/'fallback_reference_comparison.png',dpi=180)
fig.savefig(ROOT/'fallback_reference_comparison.svg')
plt.close(fig)

lines=['# Default gains and fallback-inertia reference','',
'The user correctly identified a missing benchmark: tuning should be judged against the behavior of the default-gain, previously stable model, not only against default gains on the corrected low-inertia UMI model. The original sweep’s “Original” row meant **default gains with nominal UMI inertias**. Its provisional 5 mm motion-tracking cutoff was not derived from stock performance. Those pass counts do not define whether a setting matches the original controller’s useful behavior.','',
'The additional runs distinguish the complete stock-model reference from an intervention that restores only the two outer-knuckle inertia tensors on the current UMI model.','',
'## Same-protocol comparison on 80 reserved poses','',
'All cases use 120 Hz physics, decimation 12, the same saved joint starts, a common intended mounting pose, and the same relative wrist-target sequence: 2 s hold, 6 s windowed motion, 2 s hold. Default gains are translation Kp/Kd 200/84.8528 and rotation Kp/Kd 3/3.4641. Tuned gains are 1200/80 and 60/0.1. Medians use the usual average of the two middle observations.','',
'| Model and gains | Motion position RMS, median | Final wrist speed RMS, median | Final wrist speed RMS, 90th percentile | Final wrist speed RMS, maximum | Wrist-speed criterion ≤0.1 rad/s |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:
    p=r['metrics']['motion_position_rms_m'];v=r['metrics']['final_wrist_speed_rms_rad_s']
    lines.append(f"| {r['condition']} | {p['median']*1000:.3f} mm | {v['median']:.6f} rad/s | {v['p90']:.6f} rad/s | {v['maximum']:.6f} rad/s | {r['final_wrist_speed_at_most_0_1_rad_s']}/80 |")
lines += ['',
'The stock reference settles quietly at every tested pose under the stated wrist-speed criterion. The tuned nominal-inertia UMI model has lower median position error and a similar median final wrist speed, but its maximum final wrist-speed RMS is much higher. It therefore has not established the stock reference’s consistency across this cohort. The 0.1 rad/s criterion is retained only to make the existing settling measure readable; the full measured distributions and maxima are the comparison.','',
'The stock model also scores 0/80 under the earlier combined strict tracking limits. That is not evidence it is equivalent to the unstable nominal-inertia default case. It shows why tracking, settling, and contact must be compared separately and why the earlier combined pass rate is not a benchmark-derived readiness test. The stock model’s tracking errors also have a substantial tail: quiet settling does not guarantee arrival at the requested pose. All five raw error/speed distributions are preserved in the JSON.','',
'## What restoring fallback inertia does and does not reproduce','',
'The UMI inertia-only reference writes 0.004 kg·m² along each axis of both `left_outer_knuckle` and `right_outer_knuckle`, using views of just those links. Readback verifies the tensors exactly to tolerance. Their masses remain 0.0138477 kg, their COM positions remain unchanged, and every other body’s mass/inertia/COM remains unchanged. Source assets are not edited. Stored principal-axis orientation can change when a tensor is diagonalized; that is distinct from moving the COM.','',
'This intervention does not reproduce the stock model’s stable response. The complete stock model differs elsewhere: its gripper-base mass is 1.337838 kg versus UMI’s 0.744002 kg, and its gripper-base body-frame inertia diagonal is approximately [0.005731, 0.003345, 0.003977] versus [0.000749, 0.000912, 0.000653] kg·m². Stock also has 1 kg outer-knuckle fallback masses, different COMs, fingers and other properties. These readbacks explain why the two models cannot be equated; they do not by themselves isolate which remaining difference causes the residual instability.','',
'The earlier controlled stock-model ablation remains valid: reducing only the knuckle inertias destabilized that stock configuration. It established their causal contribution within that model, not that reinstating those two tensors alone would make every modified model equivalent to stock. See `/data/kanth042/datasets/original_reset_audit_20260911/knuckle_ablation/README.md`.','',
'## Loaded UMI checks','',
'The same 24 UMI grasps at 20, 100 and 200 g were also tested with the inertia-only intervention. All three UMI conditions retain 72/72 cubes, but their arm settling differs:','',
'| UMI condition | Final wrist RMS, median | Final wrist RMS, maximum | Wrist-speed criterion ≤0.1 rad/s |',
'|---|---:|---:|---:|']
for r in loaded:
    lines.append(f"| {r['condition']} | {r['final_wrist_speed_median_rad_s']:.6f} rad/s | {r['final_wrist_speed_maximum_rad_s']:.6f} rad/s | {r['final_wrist_speed_at_most_0_1_rad_s']}/72 |")
lines += ['',
'The stock robot has different fingers, so replaying the UMI grasp geometries on it would not be a matched valid grasp test. No full-stock loaded comparison is claimed here. The stock and UMI free-space runs use the same joint starts and relative targets, not identical geometry or identical end-effector world poses. Their collision-clearance validity may differ; this comparison is not an isolated inertia ablation or proof of task success.','',
'## Method and artifacts','',
'`run_gain_sweep.py --knuckle-inertia fallback --gain-ids 0` produces the UMI intervention; `--robot-model stock --gain-ids 0` produces the complete stock reference for open-hand phases. Both retain the same explicit controller, timing, nominal actuator settings, and target construction. The stock USD is the saved original asset, SHA-256 `88e72878dc44065c12a73f60bc791f2fc8f8fee4178a44ad11b4f8bc49cd5378`, with its adjacent original calibration metadata. Native root-pose assertions, restored-joint checks, torque-formula checks, source hashes, and runtime mass/inertia snapshots are retained for each run. The independent UMI articulation-tree pose audit includes all inertia-intervention phases; the different stock asset is not incorrectly checked against that UMI tree.','',
'Folders `fallback_screen`, `fallback_holdout`, and `fallback_grasp` hold the inertia intervention, while `stock_reference_screen` and `stock_reference_holdout` hold the complete stock reference. Each contains its executed runner source, runtime inputs, every-step trajectories, and per-pose metrics. The added screen reference uses the original 16 poses. Initial setup attempts that stopped before simulation are retained in explicitly named aborted logs.','',
'`report_fallback_reference.py` recomputes these tables, the JSON, and the PNG/SVG plot from completed summaries. No gains or inertias have been activated in training, and no training or hardware job was launched. Future gain selection should state separately which stock-response characteristics it must match and which task-specific tracking requirements it must satisfy.','']
(ROOT/'fallback_reference.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
