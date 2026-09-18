"""Render the actual saved 4.96 mm collision and a measured cross-section.

Uses exported collision hulls and source-joint FK. No fitting, model changes,
simulation, hardware commands, or mapping. Rendering does not scale the overlap.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import trimesh

from reachability_geometry import ROOT, SourceCollisionChecker, batch_fk, exact_pair_gap, fcl, pose_matrix, set_transform


root=ROOT/'20_dense_reachability'
out=root/'contact_5mm';out.mkdir(exist_ok=True)
failure=json.loads((root/'raw_sphere_gpu_verification.json').read_text())['hull_rejected_but_sphere_accepted'][0]
checker=SourceCollisionChecker(ROOT/'15_table_reachability/model')
fk=batch_fk(checker.audit,np.array([failure['q']]))
names=['right_inner_finger_shape_0','upper_arm_link_shape_0']
ids=[next(i for i,s in enumerate(checker.shapes) if s['name']==name) for name in names]
for i in ids:set_transform(checker.objects[i],fk[checker.shapes[i]['body']][0])
distance=fcl.DistanceResult()
gap=fcl.distance(checker.objects[ids[0]],checker.objects[ids[1]],
                 fcl.DistanceRequest(enable_nearest_points=True,enable_signed_distance=True),distance)
point_a,point_b=np.asarray(distance.nearest_points)
delta=point_b-point_a;depth=np.linalg.norm(delta);center=(point_a+point_b)/2
assert abs(depth+gap)<1e-10
translation_tests=[]
for scale in [1.,1.02]:
    shifted=fk[checker.shapes[ids[0]]['body']][0].copy()
    shifted[:3,3]+=scale*delta
    set_transform(checker.objects[ids[0]],shifted)
    translated_gap=exact_pair_gap(checker.objects[ids[0]],checker.objects[ids[1]],signed=True)
    translation_tests.append(dict(scale=scale,gap_m=translated_gap))
assert abs(translation_tests[0]['gap_m'])<1e-7
assert abs(translation_tests[1]['gap_m']-.02*depth)<1e-7

# The cross-section contains both FCL witness points and the finger's width
# direction. Its x axis is the direction in which the finger separates.
e1=delta/depth
finger_axis=fk['right_inner_finger'][0,:3,2]
e2=finger_axis-e1*np.dot(finger_axis,e1);e2/=np.linalg.norm(e2)
e3=np.cross(e1,e2)
basis=np.column_stack([e1,e2,e3])
meshes={}
sections={}
all_shapes=[]
for shape in checker.shapes:
    mesh=trimesh.load(shape['mesh'],force='mesh',process=False)
    T=fk[shape['body']][0]
    vertices=np.asarray(mesh.vertices)@T[:3,:3].T+T[:3,3]
    all_shapes.append(dict(name=shape['name'],body=shape['body'],vertices_base_m=vertices,faces=np.asarray(mesh.faces)))
    if shape['name'] not in names:continue
    transformed=trimesh.Trimesh(vertices=vertices,faces=mesh.faces,process=False)
    segments=trimesh.intersections.mesh_plane(transformed,plane_normal=e3,plane_origin=center)
    points=(segments.reshape(-1,3)-center)@basis*1000
    assert np.abs(points[:,2]).max()<1e-6
    unique=np.unique(np.round(points[:,:2],decimals=8),axis=0)
    polygon=unique[ConvexHull(unique).vertices]
    sections[shape['name']]=polygon
    meshes[shape['name']]=transformed
overlap=Polygon(sections[names[0]]).intersection(Polygon(sections[names[1]]))
assert overlap.geom_type=='Polygon' and overlap.area>0
witnesses=(np.array([point_a,point_b])-center)@basis*1000
assert np.allclose(witnesses[:,1:],0,atol=1e-8)
assert np.allclose(witnesses[:,0],[-depth*500,depth*500],atol=1e-8)

colors={names[0]:'#2764a8',names[1]:'#b77827'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':16,
                     'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12})
fig=plt.figure(figsize=(12,6.4),facecolor='white')
ax=fig.add_axes([.01,.13,.47,.77],projection='3d')
ax.set_title('Saved pose · transparent hull view',pad=14)
world_T_base=pose_matrix(checker.audit['lab_constants']['ROBOT_POS'],checker.audit['lab_constants']['ROBOT_ROT'])
world_points=[]
for shape in all_shapes:
    points=shape['vertices_base_m']@world_T_base[:3,:3].T+world_T_base[:3,3]
    world_points.append(points)
    color=colors.get(shape['name'],'#bec5cd')
    opacity=1. if shape['name']==names[0] else (.28 if shape['name']==names[1] else .42)
    collection=Poly3DCollection(points[shape['faces']],facecolors=color,
                                edgecolors='none',alpha=opacity,linewidths=0,rasterized=True)
    ax.add_collection3d(collection)
all_points=np.concatenate(world_points)
mid=(all_points.max(axis=0)+all_points.min(axis=0))/2
span=max(np.ptp(all_points,axis=0))*.53
ax.set_xlim(mid[0]-span,mid[0]+span);ax.set_ylim(mid[1]-span,mid[1]+span);ax.set_zlim(mid[2]-span,mid[2]+span)
ax.set_box_aspect((1,1,1),zoom=1.55);ax.set_proj_type('ortho');ax.view_init(elev=24,azim=-112)
ax.set_axis_off()
contact_world=center@world_T_base[:3,:3].T+world_T_base[:3,3]
projected=proj3d.proj_transform(*contact_world,ax.get_proj())
ax.annotate('Contact',xy=projected[:2],xycoords='data',xytext=(.66,.20),textcoords='axes fraction',
            color='#a8242b',fontsize=14,ha='center',arrowprops=dict(arrowstyle='->',color='#a8242b',lw=1.5))
from matplotlib.patches import Circle
locator=Circle(projected[:2],radius=.007,fill=False,edgecolor='#c82f38',linewidth=2.2,
               transform=ax.transData,zorder=100,clip_on=False)
fig.add_artist(locator)

section=fig.add_axes([.56,.16,.40,.72])
section.set_title('Section across the finger width',pad=14)
for name in names[::-1]:
    section.add_patch(PolygonPatch(sections[name],facecolor=colors[name],edgecolor=colors[name],alpha=.20,linewidth=1.8))
    p=sections[name]
    section.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color=colors[name],lw=1.7)
section.add_patch(PolygonPatch(np.asarray(overlap.exterior.coords),facecolor='#ce3842',edgecolor='none',alpha=.80,zorder=4))
section.scatter(witnesses[:,0],witnesses[:,1],s=27,c='#20252b',zorder=7)
dimension_y=-12
for x in witnesses[:,0]:section.plot([x,x],[0,dimension_y],color='#20252b',lw=.9,zorder=6)
section.annotate('',xy=(witnesses[1,0],dimension_y),xytext=(witnesses[0,0],dimension_y),
                 arrowprops=dict(arrowstyle='<->',lw=1.4,color='#20252b',shrinkA=0,shrinkB=0),zorder=8)
section.text(0,dimension_y-3.4,f'{depth*1000:.2f} mm',ha='center',va='top',color='#20252b',fontsize=17,zorder=9,
             bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
section.set_xlim(-25,25);section.set_ylim(-25,25);section.set_aspect('equal')
section.set_xticks([-20,-10,0,10,20]);section.set_yticks([-20,-10,0,10,20])
section.set_xlabel('Along separation direction (mm)',labelpad=9)
section.set_ylabel('Across finger width (mm)',labelpad=5)
section.grid(color='#d9dee3',linewidth=.65,alpha=.7);section.set_axisbelow(True)
for spine in section.spines.values():spine.set_color('#b9c1c9')
fig.legend(handles=[PolygonPatch([[0,0],[1,0],[1,1]],facecolor=colors[names[0]],label='Right TPU finger hull'),
                    PolygonPatch([[0,0],[1,0],[1,1]],facecolor=colors[names[1]],label='Upper-arm hull'),
                    PolygonPatch([[0,0],[1,0],[1,1]],facecolor='#ce3842',label='Hull overlap')],
           loc='upper center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=14,handlelength=1.2)
fig.savefig(out/'missed_contact.png',dpi=180,facecolor='white')
fig.savefig(out/'missed_contact.pdf',facecolor='white')
plt.close(fig)

report=dict(row=failure['row'],joint_names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'],
    joint_positions_rad=failure['q'],signed_fcl_gap_m=gap,penetration_depth_m=depth,
    witness_points_base_m=[point_a.tolist(),point_b.tolist()],
    finger_separating_translation_base_m=delta.tolist(),translation_validation=translation_tests,
    section_origin_base_m=center.tolist(),section_basis_columns_base=basis.tolist(),
    section_polygons_mm={k:v.tolist() for k,v in sections.items()},
    overlap_section_polygon_mm=np.asarray(overlap.exterior.coords).tolist(),
    geometry_hashes={str(Path(checker.shapes[i]['mesh'])):hashlib.sha256(Path(checker.shapes[i]['mesh']).read_bytes()).hexdigest() for i in ids},
    meaning='4.96 mm is the FCL penetration depth for these two collision hulls at the saved joint pose. Translating the finger hull alone by the measured vector brings them into contact. This translation is a geometry check, not an articulated robot motion.',
    view='The section is computed by intersecting the original transformed collision meshes with a plane containing both FCL witness points. No overlap scaling or hand-drawn collision shapes.')
(out/'measurement.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(image=str(out/'missed_contact.png'),penetration_mm=depth*1000,
                     overlap_section_area_mm2=overlap.area,translation_validation=translation_tests),indent=2))
