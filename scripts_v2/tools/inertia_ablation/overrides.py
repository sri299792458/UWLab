"""Apply and verify per-case body inertial interventions in an isolated fixture."""
import torch


def apply_overrides(env, robot, cases, poses_per_case):
    view = robot.root_physx_view
    before_m, before_i, before_c = [x.clone() for x in (view.get_masses(), view.get_inertias(), view.get_coms())]
    target_m, target_i, target_c = [x.clone() for x in (before_m, before_i, before_c)]
    masks = {k:torch.zeros(before_m.shape, dtype=torch.bool, device=before_m.device) for k in ['mass','inertia','com']}
    for j, case in enumerate(cases):
        rows=slice(j*poses_per_case,(j+1)*poses_per_case)
        for name, props in case.get('body_properties',{}).items():
            bid=robot.body_names.index(name)
            if 'mass' in props:
                target_m[rows,bid]=props['mass'];masks['mass'][rows,bid]=True
            if 'inertia' in props:
                target_i[rows,bid]=torch.as_tensor(props['inertia'],device=target_i.device).flatten();masks['inertia'][rows,bid]=True
            if 'com' in props:
                target_c[rows,bid,:3]=torch.as_tensor(props['com'],device=target_c.device);masks['com'][rows,bid]=True
    for bid,name in enumerate(robot.body_names):
        if not any(mask[:,bid].any() for mask in masks.values()):continue
        link=env.sim.physics_sim_view.create_rigid_body_view('/World/envs/env_*/Robot/'+name)
        assert link.count==env.num_envs
        for field,values,setter in [('mass',target_m[:,bid:bid+1],link.set_masses),('com',target_c[:,bid],link.set_coms),('inertia',target_i[:,bid],link.set_inertias)]:
            rows=torch.nonzero(masks[field][:,bid],as_tuple=True)[0].cpu().to(torch.int32)
            if len(rows):setter(values.contiguous(),rows)
    actual_m,actual_i,actual_c=[x.clone() for x in (view.get_masses(),view.get_inertias(),view.get_coms())]
    assert torch.allclose(actual_m,target_m,rtol=1e-6,atol=1e-9),float((actual_m-target_m).abs().max())
    assert torch.allclose(actual_i,target_i,rtol=1e-5,atol=1e-9),float((actual_i-target_i).abs().max())
    assert torch.allclose(actual_c[:,:,:3],target_c[:,:,:3],rtol=1e-6,atol=1e-8),float((actual_c[:,:,:3]-target_c[:,:,:3]).abs().max())
    assert torch.equal(actual_m[~masks['mass']],before_m[~masks['mass']])
    assert torch.equal(actual_i[~masks['inertia']],before_i[~masks['inertia']])
    assert torch.equal(actual_c[:,:,:3][~masks['com']],before_c[:,:,:3][~masks['com']])
    report=dict(body_names=robot.body_names,original_masses_kg=before_m[0].cpu().tolist(),
        original_inertias_kg_m2=before_i[0].cpu().tolist(),original_com_poses=before_c[0].cpu().tolist(),
        cases=[dict(id=c['id'],label=c['label'],protocol=c['protocol'],
            requested_body_properties=c.get('body_properties',{}),
            masses_kg=actual_m[j*poses_per_case].cpu().tolist(),
            inertias_kg_m2=actual_i[j*poses_per_case].cpu().tolist(),
            com_poses=actual_c[j*poses_per_case].cpu().tolist()) for j,c in enumerate(cases)],
        verification=dict(maximum_mass_error_kg=float((actual_m-target_m).abs().max()),
            maximum_inertia_error_kg_m2=float((actual_i-target_i).abs().max()),
            maximum_com_position_error_m=float((actual_c[:,:,:3]-target_c[:,:,:3]).abs().max()),
            untouched_masses_inertias_com_positions_exact=True))
    return report
