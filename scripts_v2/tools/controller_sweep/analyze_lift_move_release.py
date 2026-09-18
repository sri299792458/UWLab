"""Summarize completed handling runs and validate the fixed physical inputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/25_lift_move_release')
REFERENCE=json.loads((ROOT.parent/'24_gain_failure_diagnosis/grasp_A/worker_0/runtime_inputs.json').read_text())
torch.set_num_threads(1)
rows=[];validation=[];baseline_values={}
fixed=['robot_usd_sha256','controller_sha256','masses_kg','inertias_kg_m2','joint_armature',
       'joint_stiffness','joint_damping','solver_position_iterations','solver_velocity_iterations',
       'arm_gravity_disabled','physics_hz','decimation']
for folder in sorted(ROOT.glob('*/worker_0')):
    if not (folder/'summary.json').exists():continue
    s=json.loads((folder/'summary.json').read_text())
    snap=json.loads((folder/'runtime_inputs.json').read_text())
    for key in fixed:assert snap[key]==REFERENCE[key],(folder,key)
    for key,filekey in [('robot_usd_sha256','robot_usd'),('controller_sha256','controller_file')]:
        assert hashlib.sha256(Path(snap[filekey]).read_bytes()).hexdigest()==snap[key],(folder,key)
    if snap.get('uses_task_config_gains_without_runtime_override'):
        assert hashlib.sha256(Path(snap['task_config_source']).read_bytes()).hexdigest()==snap['task_config_sha256']
        assert np.allclose(snap['actual_runtime_kp'],[500,500,500,60,60,60],rtol=0,atol=1e-5)
        assert np.allclose(snap['actual_runtime_kd'],[160,160,160,.1,.1,.1],rtol=0,atol=1e-5)
    archive=folder/'run_gain_sweep.py'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==s['script_sha256']
    assert s['controller_formula_max_error_nm']==0
    assert snap['poses']==REFERENCE['poses']
    f=torch.load(folder/'trajectories.pt',map_location='cpu',weights_only=True)
    contact=torch.load(folder/'contact_trajectories.pt',map_location='cpu',weights_only=True)
    for data in [f,contact]:
        for key,value in data.items():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all(),(folder,key)
    v=f['values'];P=f['pose_count'];assert P==72 and v.shape[0]==1320
    force=contact['contact_net_forces_world'].norm(dim=-1)
    eligible=[i for i,n in enumerate(contact['contact_body_names']) if n not in ['left_inner_finger','right_inner_finger']]
    all_contact_peak=force[:,:,eligible].amax(dim=(0,2)).numpy()
    pad_delta=0.
    for side,column in [('left',27),('right',28)]:
        index=contact['contact_body_names'].index(side+'_inner_finger')
        pad_delta=max(pad_delta,float((force[:,:,index]-v[:,:,column]).abs().max()))
    assert pad_delta<2e-5,(folder,pad_delta)
    for j,r in enumerate(s['results']):
        gain=r['gain'];m=r['per_pose'];sl=slice(j*P,(j+1)*P);a=v[:,sl].numpy()
        assert np.allclose(np.array(snap['initial_joint_positions'])[sl],np.array([p['joint_positions'] for p in snap['poses']]),rtol=0,atol=1e-6)
        contact_cases=np.flatnonzero(all_contact_peak[sl]>.01).tolist()
        before_release=np.array(m['maximum_robot_obstacle_force_n'])>.01
        assert np.array_equal(all_contact_peak[sl]>.01,before_release)
        retained=(np.array(m['carried_bilateral_fraction'])>=.8)&(np.array(m['carry_relative_position_max_m'])<=.015)&(np.array(m['carry_relative_rotation_max_rad'])<=np.deg2rad(5))
        opened=(np.array(m['release_drop_m'])>=.02)&(np.array(m['final_finger_cube_force_max_n'])<.02)&(np.array(m['final_driver_angle_rad'])<.1)
        hf=np.sqrt(np.mean(np.sum((.5*np.diff(a[900:960,:,15:18],axis=0))**2,axis=-1),axis=0))
        quiet_final_hf=np.sqrt(np.mean(np.sum((.5*np.diff(a[1200:1320,:,15:18],axis=0))**2,axis=-1),axis=0))
        clean=np.array(r['flags']['collision_free'])
        position=np.array(m['move_end_position_rms_m']);wrist=np.array(m['move_end_wrist_speed_rms_rad_s'])
        record=dict(run=folder.parent.name,gain=gain,counts=r['counts'],
            retained_during_motion=int(retained.sum()),released_after_open=int(opened.sum()),
            unwanted_contact_case_indices=contact_cases,
            median_move_endpoint_position_mm=float(np.median(position)*1000),
            maximum_move_endpoint_position_mm=float(position.max()*1000),
            median_pre_release_wrist_speed_rad_s=float(np.median(wrist)),
            maximum_pre_release_wrist_speed_rad_s=float(wrist.max()),
            pre_release_settled_count=int((wrist<=.1).sum()),
            after_release_settled_count=int((np.array(m['final_wrist_speed_rms_rad_s'])<=.1).sum()),
            maximum_after_release_wrist_speed_rad_s=max(m['final_wrist_speed_rms_rad_s']),
            maximum_pre_release_wrist_speed_collision_free_cases_rad_s=float(wrist[clean].max()),
            pre_release_half_step_difference_median_rad_s=float(np.median(hf)),
            pre_release_half_step_difference_max_rad_s=float(hf.max()),
            final_half_step_difference_max_rad_s=float(quiet_final_hf.max()),
            precise_failures_without_collision=[dict(pose_index=i,posture_index=i//3,mass_kg=snap['poses'][i]['cube_mass_kg'],
                lift_error_mm=m['lift_end_position_rms_m'][i]*1000,move_error_mm=m['move_end_position_rms_m'][i]*1000,
                wrist_speed_rad_s=m['move_end_wrist_speed_rms_rad_s'][i]) for i in range(P) if clean[i] and not r['flags']['precise_and_settled'][i]],
            summary_path=str(folder/'summary.json'))
        rows.append(record)
        if gain['id']==292:baseline_values[folder.parent.name]=v[:,sl].numpy().copy()
    validation.append(dict(run=folder.parent.name,nominal_model_properties_unchanged=True,
        source_hashes_valid=True,archived_runner_hash_valid=True,all_tensors_finite=True,
        controller_formula_max_error_nm=s['controller_formula_max_error_nm'],
        finger_net_vs_cube_filtered_force_max_difference_n=pad_delta,
        no_new_robot_contact_cases_after_open=True,
        uses_task_config_gains_without_runtime_override=snap.get('uses_task_config_gains_without_runtime_override',False)))

paired_names=[name for name in baseline_values if name!='candidate_A']
baseline_comparison={}
for name in paired_names:
    delta=np.abs(baseline_values[name]-baseline_values[paired_names[0]])
    baseline_comparison[name]=dict(pre_release_joint_position_rad=float(delta[:960,:,6:12].max()),
        pre_release_joint_velocity_rad_s=float(delta[:960,:,12:18].max()),
        pre_release_hand_position_error_m=float(delta[:960,:,:3].max()),
        whole_sequence_cube_speed_m_s=float(delta[:,:,30].max()))
result=dict(protocol='2 s close/hold, 2 s lift 50 mm, 1 s hold, 2 s horizontal carry 50 mm toward table center, 1 s hold, 3 s open/release; fixed initial hand orientation.',
    scope='Scripted handling from the existing 24 saved airborne grasp postures at 20/100/200 g, in the actual nominal UMI training scene. No trained policy, pickup, or stacked placement is evaluated.',
    unique_gain_settings=len({r['gain']['id'] for r in rows}),rows=rows,validation=validation,
    matched_batch_A_trajectory_max_abs_deltas=baseline_comparison,
    criteria=json.loads((ROOT/'candidate_A/worker_0/summary.json').read_text())['criteria'])
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
with (ROOT/'gain_results.csv').open('w',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['run','gain_id','translation_kp','translation_kd','rotation_kp','rotation_kd','sequence_complete','collision_free','precise_settled','endpoint_position_median_mm','wrist_speed_max_rad_s'])
    for r in rows:
        g=r['gain'];writer.writerow([r['run'],g['id'],g['kp'][0],g['kd'][0],g['kp'][3],g['kd'][3],r['counts']['released'],r['counts']['collision_free'],r['counts']['precise_and_settled'],r['median_move_endpoint_position_mm'],r['maximum_pre_release_wrist_speed_rad_s']])
for r in rows:print(r['run'],r['gain']['id'],r['counts'],'position median mm',round(r['median_move_endpoint_position_mm'],3),'wrist max',round(r['maximum_pre_release_wrist_speed_rad_s'],3),flush=True)
print('VALIDATED',len(validation),'runs;',result['unique_gain_settings'],'unique gains; batch-control differences',baseline_comparison,flush=True)
