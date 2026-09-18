"""Outcome-blind common stimulus selection and attribute-specific permutations.

For each food-form group and nutrient, a permutation avoiding equal displayed
values exists iff no displayed value occupies more than half the selected set.
Maximize the common cardinality subject to those constraints for all nutrients;
then break cardinality ties with a fixed random objective, never outcomes.
"""
import argparse, csv, hashlib, json, platform, shutil, subprocess
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint, linear_sum_assignment
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ['protein_100g', 'fat_100g', 'carbs_100g', 'fibre_100g']
LABELS = ['Protein', 'Fat', 'Carbohydrate', 'Fibre']
SEED = 20260917

def allowed_pairs(values, rule):
    v=np.array([float(x) for x in values]);d=np.abs(v[:,None]-v[None,:])
    allowed=d>1e-8
    if rule=='conservative':
        scale=np.maximum(np.maximum(v[:,None],v[None,:]),1.0)
        allowed &= (d+1e-8>=.05*scale)&(d<=.5*scale+1e-8)
    return allowed

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True)
    ap.add_argument('--rule',choices=['different','conservative'],default='different');args = ap.parse_args()
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT/'outputs') and not out.exists(); out.mkdir(parents=True)
    source = ROOT/'data/prepared/image_agreement_nutrition.csv'
    manifest = ROOT/'outputs/20260916_刺激与错配准备_v03/stimulus_manifest.csv'
    # Deliberate allowlist: no human ratings, beta values, or model outputs.
    meta = {r['image_id']: {k:r[k] for k in ['image_id', *ATTRS]} for r in read(source)}
    columns = ['image_id','food_name','pairing_group','food_group','image_path','P_eligible']
    base = sorted([{k:r[k] for k in columns} for r in read(manifest) if r['P_eligible']=='True'], key=lambda r:int(r['image_id']))
    assert len(base)==493
    groups=defaultdict(list); excluded=[]
    for r in base:
        vals=meta[r['image_id']]
        if not all(vals[a] and np.isfinite(float(vals[a])) for a in ATTRS):
            excluded.append({**r,'reason':'missing_nutrient_metadata'}); continue
        for a in ATTRS:
            assert float(vals[a])>=0
            r[a]=vals[a]; r['display_'+a]=f'{float(vals[a]):.2f}'
        groups[r['pairing_group']].append(r)
    selected=[]; group_audit=[]; rng=np.random.default_rng(SEED)
    for name, rr in sorted(groups.items()):
        n=len(rr); constraints=[]
        for a in ATTRS:
            vals=np.array([r['display_'+a] for r in rr])
            for value in sorted(set(vals)):
                constraints.append(2*(vals==value).astype(float)-1)
        if args.rule=='different':
            c=LinearConstraint(np.array(constraints), -np.inf, 0);nv=n
        else:
            edges=[]
            for ai,a in enumerate(ATTRS):
                allowed=allowed_pairs([r['display_'+a] for r in rr],args.rule)
                edges.extend((ai,int(i),int(j)) for i,j in zip(*np.where(allowed)))
            nv=n+len(edges);matrix=lil_matrix((8*n,nv))
            for ai in range(4):
                for i in range(n):matrix[2*ai*n+i,i]=-1;matrix[(2*ai+1)*n+i,i]=-1
            for col,(ai,i,j) in enumerate(edges,n):matrix[2*ai*n+i,col]=1;matrix[(2*ai+1)*n+j,col]=1
            c=LinearConstraint(matrix.tocsr(),0,0)
        cardinality=np.r_[np.ones(n),np.zeros(nv-n)]
        res=milp(-cardinality,integrality=np.ones(nv),bounds=Bounds(0,1),constraints=c,options={'mip_rel_gap':0})
        assert res.success and res.mip_gap==0
        maximum=int(round(-res.fun))
        tie=milp(np.r_[rng.random(n),np.zeros(nv-n)],integrality=np.ones(nv),bounds=Bounds(0,1),
                 constraints=[c,LinearConstraint(cardinality[None,:],maximum,maximum)],options={'mip_rel_gap':0})
        assert tie.success and tie.mip_gap==0
        keep=np.rint(tie.x[:n]).astype(bool); assert keep.sum()==maximum
        assert np.all(np.array(constraints) @ keep <= 0)
        kept=[r for r,k in zip(rr,keep) if k]; selected.extend(kept)
        for r,k in zip(rr,keep):
            if not k: excluded.append({**{col:r[col] for col in columns},'reason':'joint_different_display_value_permutation_infeasible_at_full_group_size'})
        group_audit.append({'pairing_group':name,'complete_candidates':n,'maximal_common_count':maximum,
                            'kept_ids':' '.join(r['image_id'] for r in kept),
                            'removed_ids':' '.join(r['image_id'] for r,k in zip(rr,keep) if not k)})
    selected.sort(key=lambda r:int(r['image_id'])); ids=[r['image_id'] for r in selected]
    assert len(ids)==len(set(ids))==(466 if args.rule=='different' else 401)
    for r in selected: r['image_sha256']=sha(ROOT/r['image_path'])
    permutations=[]; diagnostics=[]
    for ai,a in enumerate(ATTRS):
        ar=[]
        for replica in range(1,6):
            seed=SEED+100*(ai+1)+replica; prng=np.random.default_rng(seed)
            for name in sorted(groups):
                rr=[r for r in selected if r['pairing_group']==name]
                if not rr: continue
                n=len(rr); values=np.array([r['display_'+a] for r in rr]); allowed=allowed_pairs(values,args.rule)
                ri,ci=linear_sum_assignment(np.where(allowed,prng.random((n,n)),1e9))
                assert allowed[ri,ci].all()
                for i,j in zip(ri,ci):
                    v,w=float(values[i]),float(values[j])
                    ar.append({'nutrient':a,'condition':f'P{replica}','image_id':rr[i]['image_id'],
                      'donor_image_id':rr[j]['image_id'],'pairing_group':name,'food_name':rr[i]['food_name'],
                      'donor_food_name':rr[j]['food_name'],'true_value':values[i],'presented_value':values[j],
                      'absolute_delta_g100g':abs(w-v),'symmetric_relative_delta':2*abs(w-v)/(w+v),
                      'zero_transition':(v==0)!=(w==0),'seed':seed})
        for replica in range(1,6):
            rows=[r for r in ar if r['condition']==f'P{replica}']
            assert len(rows)==len(ids) and set(r['image_id'] for r in rows)==set(ids)
            assert set(r['donor_image_id'] for r in rows)==set(ids)
            assert Counter(r['true_value'] for r in rows)==Counter(r['presented_value'] for r in rows)
            assert all(r['image_id']!=r['donor_image_id'] and r['true_value']!=r['presented_value'] for r in rows)
        delta=np.array([r['absolute_delta_g100g'] for r in ar])
        diversity=Counter(len({r['donor_image_id'] for r in ar if r['image_id']==i}) for i in ids)
        value_diversity=Counter(len({r['presented_value'] for r in ar if r['image_id']==i}) for i in ids)
        diagnostics.append({'nutrient':a,'pairs':len(ar),'delta_min_median_max':np.quantile(delta,[0,.5,1]).tolist(),
            'delta_quantiles_0_5_25_50_75_95_100':np.quantile(delta,[0,.05,.25,.5,.75,.95,1]).tolist(),
            'delta_below_0_1_count':int((delta<.1).sum()),'zero_transitions':sum(r['zero_transition'] for r in ar),
            'distinct_donors_per_image_histogram':dict(sorted(diversity.items())),
            'distinct_presented_values_per_image_histogram':dict(sorted(value_diversity.items()))})
        permutations.extend(ar)
    write(out/'stimulus_manifest.csv',selected); write(out/'excluded_images.csv',excluded)
    write(out/'group_feasibility.csv',group_audit); write(out/'P_permutations.csv',permutations)
    extreme=[]
    for a in ATTRS:
        rr=[r for r in permutations if r['nutrient']==a]
        unique={ (r['image_id'],r['donor_image_id']):r for r in rr }
        ordered=sorted(unique.values(),key=lambda r:r['absolute_delta_g100g'])
        for tag,rs in [('smallest',ordered[:10]),('largest',ordered[-10:])]:
            extreme.extend({**r,'review_set':tag} for r in rs)
    write(out/'extreme_pairs_review.csv',extreme)
    (out/'pair_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    freeze={'date':datetime.now().astimezone().isoformat(),'status':'metadata_prepared_effects_not_read',
      'base_count':493,'complete_candidates':489,'common_count':len(ids),'image_ids':ids,
      'selection_seed':SEED,'selection':'maximum_common_cardinality_then_seeded_tie_break',
      'nutrients':ATTRS,'templates':{a:f'{label}: {{value:.2f}} g per 100 g.' for a,label in zip(ATTRS,LABELS)},
      'allowed':'same_existing_food_form_group_and_different_two_decimal_value','rule':args.rule,
      'numeric_limits':('0.05 <= abs(T-P)/max(T,P,1g100g) <= 0.50; input-only operational choice, not human validated' if args.rule=='conservative' else 'no_further_absolute_or_ratio_limit; diagnostic_review_required_before_model_run'),
      'P_seeds':{a:[SEED+100*(i+1)+k for k in range(1,6)] for i,a in enumerate(ATTRS)},
      'new_inputs_per_prompt':len(ids)*4*6,'new_inputs_total_AB':len(ids)*4*6*2,
      'behavior_and_brain_data_read':False,'model_results_read':False}
    (out/'preparation.json').write_text(json.dumps(freeze,indent=2))
    shutil.copy2(__file__,out/'script_snapshot.py')
    audit={'status':'passed_metadata_constraints','created':datetime.now().astimezone().isoformat(),
      'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [source,manifest,Path(__file__)]},
      'output_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
      'validation':'optimal common cardinality; 20 complete bijections; different formatted values; exact marginals; metadata allowlist'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps({'common_count':len(ids),'diagnostics':diagnostics},indent=2))

if __name__=='__main__': main()
