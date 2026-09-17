"""Analyze saved A/B trajectories without changing the model or controller."""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
import yaml

torch.set_num_threads(1)
ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
INPUT=ROOT/'22_explicit_gain_sweep';OUT=ROOT/'24_gain_failure_diagnosis';OUT.mkdir(exist_ok=True)
REPO=Path('/data/kanth042/repos/UWLab-reset-from-defaults')
SOURCE=REPO/'source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/kinematics.py'
# Execute the unchanged pure numerical definitions with an explicitly local
# metadata resolver; importing Isaac Lab itself would require starting Kit.
tree=ast.parse(SOURCE.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
def local_path(path,download_dir=None):
    assert Path(path).is_file(),path
    return path
scope=dict(torch=torch,yaml=yaml,functools=functools,os=os,tempfile=tempfile,retrieve_file_path=local_path)
exec(compile(tree,str(SOURCE),'exec'),scope)
Jfun=scope['compute_jacobian_analytical']
audit=json.loads((ROOT/'15_table_reachability/model/export_audit.json').read_text())
names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
limits=np.array([[next(j for j in audit['articulation_tree'] if j['name']==name)[key] for name in names] for key in ['lower','upper']])
plan=json.loads((INPUT/'plan_expanded.json').read_text());cutoffs=plan['ranking_limits']
for path,digest in plan['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
rows=[];trace_data={}
def rms(x):return np.sqrt(np.mean(np.sum(x*x,axis=-1),axis=0))
for phase in ['corrected_holdout','grasp']:
    for folder in sorted((INPUT/phase).glob('worker_*')):
        summary=json.loads((folder/'summary.json').read_text());snap=json.loads((folder/'runtime_inputs.json').read_text())
        if not set(summary['gain_ids'])&{292,276}:continue
        f=torch.load(folder/'trajectories.pt',map_location='cpu',weights_only=True);v=f['values'].numpy();P=f['pose_count']
        for group,gain_id in enumerate(f['gain_ids']):
            if gain_id not in [292,276]:continue
            gain=summary['results'][group];a=v[:,group*P:(group+1)*P].astype(np.float64)
            q=a[:,:,6:12];qd=a[:,:,12:18];tau=a[:,:,18:24]
            Js=[]
            for start in range(0,len(q.reshape(-1,6)),4096):
                Js.append(Jfun(torch.from_numpy(q.reshape(-1,6)[start:start+4096].astype(np.float32)),device='cpu',usd_path=snap['robot_usd']).numpy())
            J=np.concatenate(Js).reshape(1200,P,6,6).astype(np.float64)
            task_v=np.einsum('tpij,tpj->tpi',J,qd)
            K=np.einsum('tpri,r,tprj->tpij',J,np.array(gain['kp']),J)
            D=np.einsum('tpri,r,tprj->tpij',J,np.array(gain['kd']),J)
            eig,vec=np.linalg.eigh(K)
            mode_projection=np.einsum('tpi,tpi->tp',vec[:,:,:,0],qd)**2
            velocity_energy=np.sum(qd*qd,axis=-1)
            least_mode_fraction=np.sum(mode_projection[1080:],axis=0)/np.sum(velocity_energy[1080:],axis=0).clip(1e-30)
            damping_power=np.einsum('tpi,tpij,tpj->tp',qd,D,qd)
            motion_direction_damping=damping_power/velocity_energy.clip(1e-30)
            alignment=np.degrees(np.arccos(np.clip(np.abs(np.sum(J[:,:,3:,3]*J[:,:,3:,5],axis=-1)),0,1)))
            margin=np.minimum(q-limits[0],limits[1]-q)
            hf=rms(.5*np.diff(qd[1080:,:,3:],axis=0));wrist=rms(qd[1080:,:,3:]);ee_lin=rms(task_v[1080:,:,:3]);ee_ang=rms(task_v[1080:,:,3:])
            # Spectrum over the two-second final hold, after removing the mean.
            x=qd[960:,:,3:]-qd[960:,:,3:].mean(axis=0,keepdims=True)
            power=np.sum(np.abs(np.fft.rfft(x,axis=0))**2,axis=-1);freq=np.fft.rfftfreq(len(x),1/120)
            dominant=freq[1+np.argmax(power[1:],axis=0)]
            high_band=np.sum(power[freq>=20],axis=0)/np.sum(power[1:],axis=0).clip(1e-30)
            passed=np.ones(P,bool)
            for key,threshold in cutoffs.items():passed&=np.array(gain['per_pose'][key])<=threshold
            assert int(passed.sum())==gain['passed']
            key=f'{phase}_{gain_id}'
            trace_data[key]=dict(values=a.astype(np.float32),jacobian=J.astype(np.float32),task_velocity=task_v.astype(np.float32),
                stiffness_eigenvalues=eig.astype(np.float32),least_stiffness_direction=vec[:,:,:,0].astype(np.float32),
                direction_damping=motion_direction_damping.astype(np.float32),alignment_deg=alignment.astype(np.float32),
                joint_limit_margin_rad=margin.astype(np.float32),pose_count=P)
            outrows=[]
            for i in range(P):
                bad=[k for k,l in cutoffs.items() if gain['per_pose'][k][i]>l]
                item=dict(phase=phase,gain_id=gain_id,pose_index=i,source_row=snap['poses'][i].get('source_row'),
                    grasp_posture_index=i//3 if phase=='grasp' else None,cube_mass_kg=snap['poses'][i].get('cube_mass_kg'),
                    passed=bool(passed[i]),failed_criteria=bad,
                    metrics={k:gain['per_pose'][k][i] for k in cutoffs},
                    final_wrist_speed_rms_rad_s=float(wrist[i]),final_half_step_difference_rms_rad_s=float(hf[i]),
                    final_ee_linear_speed_rms_m_s=float(ee_lin[i]),final_ee_angular_speed_rms_rad_s=float(ee_ang[i]),
                    final_dominant_wrist_frequency_hz=float(dominant[i]),final_wrist_spectral_fraction_20hz_up=float(high_band[i]),
                    torque_utilization_by_joint=np.max(np.abs(tau[:,i])/np.array([150]*3+[28]*3),axis=0).tolist(),
                    minimum_joint_limit_margin_rad=float(np.min(margin[:,i])),minimum_margin_by_joint_rad=np.min(margin[:,i],axis=0).tolist(),
                    steps_within_1mrad_of_joint_limit=int(np.sum(np.any(margin[:,i]<.001,axis=1))),
                    final_least_stiffness_direction_velocity_fraction=float(least_mode_fraction[i]),
                    minimum_wrist_alignment_angle_deg=float(np.min(alignment[:,i])),final_alignment_mean_deg=float(np.mean(alignment[1080:,i])),
                    final_minimum_joint_stiffness_eigenvalue_median=float(np.median(eig[1080:,i,0])),
                    final_motion_direction_damping_median=float(np.median(motion_direction_damping[1080:,i])),
                    final_joint_mean_velocity_rad_s=qd[1080:,i].mean(axis=0).tolist(),
                    final_joint_peak_to_peak_position_rad=np.ptp(q[960:,i],axis=0).tolist(),
                    initial_q=snap['poses'][i]['q'],input_folder=str(folder))
                rows.append(item);outrows.append(item)
            print(key,'failures',sum(not x['passed'] for x in outrows),'no clipping',max(max(x['torque_utilization_by_joint']) for x in outrows),flush=True)
            for x in outrows:
                if x['passed'] or (phase=='grasp' and x['pose_index']%3):continue
                print(x['pose_index'],'wrist',round(x['final_wrist_speed_rms_rad_s'],3),'f',x['final_dominant_wrist_frequency_hz'],'alignment',round(x['minimum_wrist_alignment_angle_deg'],3),'weak fraction',round(x['final_least_stiffness_direction_velocity_fraction'],3),'qmargin',round(x['minimum_joint_limit_margin_rad'],4),'lin mm/s',round(x['final_ee_linear_speed_rms_m_s']*1000,3),'ang',round(x['final_ee_angular_speed_rms_rad_s'],3),flush=True)
np.savez_compressed(OUT/'trajectory_analysis.npz',**{name+'__'+field:value for name,fields in trace_data.items() for field,value in fields.items()})
(OUT/'trajectory_metrics.json').write_text(json.dumps(dict(
    scope='Existing corrected A/B records only. Kinematic diagnostics are evaluated at recorded post-step joint states. No dynamics rerun or parameter change.',
    definitions=dict(joint_stiffness='J.T @ diag(Kp) @ J; local stiffness contribution from the task spring, excluding geometric terms from nonzero error.',
        joint_damping='J.T @ diag(Kd) @ J; exact velocity-feedback matrix at each recorded configuration.',
        weakest_direction='Eigenvector of the smallest J.T Kp J eigenvalue; fraction weighted by squared recorded joint speed over the final 1 s.',
        alignment='Acute angle between wrist-1 and wrist-3 axes from the analytical Jacobian.',
        high_frequency='RMS norm of half the consecutive-step difference in three wrist velocities over the final 1 s.',
        frequency='Largest non-DC wrist-velocity spectral bin during the final 2 s; resolution 0.5 Hz. Descriptive for a finite transient, not a fitted mode.'),
    kinematics_source=str(SOURCE),kinematics_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    input_hashes_match=True,rows=rows),indent=2,allow_nan=False)+'\n')
