"""Prepare crossed inertia/mass/COM interventions, never fitted gains."""
import argparse
import copy
import itertools
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--model',choices=['stock','umi','grasp'],required=True);args=p.parse_args()
R=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
inputs=R/'22_explicit_gain_sweep';output=R/'23_inertia_gap'
plan=json.loads((inputs/'plan_expanded.json').read_text())
default=copy.deepcopy(plan['gains'][0]);plan['gains']=[]
umi=json.loads((inputs/'corrected_screen/worker_0/runtime_inputs.json').read_text())
stock=json.loads((inputs/'stock_reference_screen/worker_0/runtime_inputs.json').read_text())
knuckles=['left_outer_knuckle','right_outer_knuckle'];base='robotiq_base_link'
def inertia(model,name):return model['inertias_kg_m2'][model['body_names'].index(name)]
def add(label,props,protocol,factors):
    plan['gains'].append(dict(default,id=len(plan['gains']),label=label,body_properties=props,protocol=protocol,factors=factors))
if args.model=='stock':
    for protocol in ['motion','relative_zero']:
        for k,b in itertools.product([False,True],repeat=2):
            props={}
            if k:props.update({n:dict(inertia=inertia(umi,n)) for n in knuckles})
            if b:props[base]=dict(inertia=inertia(umi,base))
            add(f'stock_lower_knuckle_{int(k)}_lower_housing_I_{int(b)}_{protocol}',props,protocol,dict(lower_knuckle_inertia=k,lower_housing_inertia=b))
else:
    native=json.loads((output/'stock_factorial/worker_0/inertial_intervention.json').read_text())
    bid=native['body_names'].index(base)
    stock_com=native['original_com_poses'][bid][:3]
    stock_mass=native['original_masses_kg'][bid]
    factors=itertools.product([False,True],repeat=4) if args.model=='umi' else ((k,b,False,False) for k,b in itertools.product([False,True],repeat=2))
    factors=list(factors)
    for protocol in ['motion','relative_zero']:
        for k,b,m,c in factors:
            props={}
            if k:props.update({n:dict(inertia=inertia(stock,n)) for n in knuckles})
            if b or m or c:
                props[base]={}
                if b:props[base]['inertia']=inertia(stock,base)
                if m:props[base]['mass']=stock_mass
                if c:props[base]['com']=stock_com
            add(f'umi_knuckle_{int(k)}_housingI_{int(b)}_mass_{int(m)}_com_{int(c)}_{protocol}',props,protocol,
                dict(fallback_knuckle_inertia=k,stock_housing_inertia=b,stock_housing_mass=m,stock_housing_com=c))
plan['experiment']=dict(model=args.model,
    scope='Default gains fixed. Each body-property intervention is applied to the named current-model body, with native readback assertions; no source asset edits. Housing coordinate axes match across stock and UMI relative to wrist_3_link within 3.1e-7 rad.',
    target_protocols=dict(motion='Existing 2 s hold, 6 s windowed relative-to-initial target motion, 2 s hold.',relative_zero='Every 10 Hz action latches the current wrist pose, reproducing zero-relative-target behavior; same starting hand and object state as the motion condition.'))
file=output/f'{args.model}_plan.json';file.write_text(json.dumps(plan,indent=2)+'\n')
print(file,len(plan['gains']),'cases')
