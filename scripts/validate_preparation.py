"""Validate real split structure and synthetic cross-fitting invariants, not hypothesis effects."""
import csv
import argparse
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from frozen_readout import crossfit_v_conditions, mean_replica_mse

ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='outputs/20260916_准备验证_v01')
    ap.add_argument('--manifest',default='outputs/20260916_刺激与错配准备_v02/stimulus_manifest.csv')
    args=ap.parse_args()
    out=ROOT/args.output
    assert out.resolve().is_relative_to(ROOT/'outputs')
    assert not out.exists();out.mkdir()
    manifest=ROOT/args.manifest
    with manifest.open() as f: rows=list(csv.DictReader(f))
    mapping=ROOT/'data/external/behavior_to_singletrial_beta_29subjects.csv'
    with mapping.open(encoding='utf-8-sig') as f:
        trials=[{k:r[k] for k in ('subject_id','session_id','run_id','trial_id','image_id')} for r in csv.DictReader(f)]
    eligible={r['image_id']:r for r in rows if r['P_eligible']=='True'}
    gfold=defaultdict(set)
    for r in rows:gfold[r['food_group']].add(r['outer_fold'])
    assert all(len(v)==1 for v in gfold.values())
    grouped=defaultdict(list)
    for r in trials:
        if r['image_id'] in eligible:grouped[r['subject_id']].append(r)
    size_records=[];missing_run_train=[]
    for s,rs in grouped.items():
        counts=Counter()
        for fold in range(5):
            train=[r for r in rs if int(eligible[r['image_id']]['outer_fold'])!=fold]
            test=[r for r in rs if int(eligible[r['image_id']]['outer_fold'])==fold]
            train_runs={(r['session_id'],r['run_id']) for r in train}
            for r in test:
                if (r['session_id'],r['run_id']) not in train_runs:missing_run_train.append((s,fold,r['run_id']))
            size_records.append({'subject_id':s,'fold':fold,'type':'food','session':'all','train_trials':len(train),'test_trials':len(test)})
            for session in sorted({r['session_id'] for r in rs}):
                a=[r for r in train if r['session_id']!=session]
                b=[r for r in test if r['session_id']==session]
                size_records.append({'subject_id':s,'fold':fold,'type':'food_session','session':session,'train_trials':len(a),'test_trials':len(b)})
                for r in b:counts[(r['session_id'],r['run_id'],r['trial_id'])]+=1
        assert set(counts.values())=={1}
        assert len(counts)==len(rs)
    assert not missing_run_train
    # Synthetic two-stage input: perturbing every outer test label cannot change predictions.
    rng=np.random.default_rng(20260916)
    n=120;d=12;groups=np.repeat(np.arange(60),2)
    x=rng.normal(size=(n,d));y=x@rng.normal(size=d)+rng.normal(size=n)*0.2
    features={'V':x,'T':x+0.1,'P1':x-0.1}
    test=np.where(groups%5==0)[0];train=np.where(groups%5!=0)[0]
    q,trace,model=crossfit_v_conditions(features,y,groups,train,test)
    altered=y.copy();altered[test]+=10000
    q2,trace2,model2=crossfit_v_conditions(features,altered,groups,train,test)
    assert all(np.array_equal(q[c],q2[c]) for c in q)
    assert np.array_equal(model.mean,x[train].mean(0))
    assert np.array_equal(model.mean,model2.mean)
    for c in features:assert np.isfinite(q[c]).all()
    for t in trace:
        assert set(groups[t['fit_indices']]).isdisjoint(groups[t['held_indices']])
        assert set(groups[t['fit_indices']]).isdisjoint(groups[test])
    eq,_,_=crossfit_v_conditions({'V':x,'T':x.copy(),'P1':x.copy()},y,groups,train,test)
    assert np.array_equal(eq['V'],eq['T']) and np.array_equal(eq['V'],eq['P1'])
    assert mean_replica_mse([y+1,y-1],y)==1.0
    # H_-s is independent of the target subject's own ratings.
    human=rng.normal(size=(29,n));s=8
    h_minus=np.mean(np.delete(human,s,axis=0),axis=0)
    human[s]+=9999
    assert np.array_equal(h_minus,np.mean(np.delete(human,s,axis=0),axis=0))
    tests={'global_group_disjoint':True,'every_trial_tested_once_in_food_session':True,
           'all_test_runs_have_training_foods_primary':True,'outer_label_perturbation_no_effect':True,
           'crossfit_group_exclusion':True,'V_scaler_frozen':True,'identical_VTP_zero_gain':True,
           'mean_P_loss_not_mean_prediction':True,'target_subject_ratings_excluded':True}
    report={'tests':tests,'seed':20260916,'scope':'structural and deterministic synthetic checks; not full population interval coverage simulation',
            'P_subject_trials':{s:len(v) for s,v in grouped.items()},
            'split_sizes':{kind:{'train_min':min(r['train_trials'] for r in size_records if r['type']==kind),'test_min':min(r['test_trials'] for r in size_records if r['type']==kind)} for kind in ['food','food_session']},
            'provenance':{'manifest_sha256':digest(manifest),'mapping_sha256':digest(mapping),'script_sha256':digest(__file__),'readout_sha256':digest(ROOT/'scripts/frozen_readout.py'),'python':platform.python_version(),'numpy':np.__version__,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}}
    with (out/'split_sizes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(size_records[0]));w.writeheader();w.writerows(size_records)
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'tests':tests,'split_sizes':report['split_sizes']}))


if __name__=='__main__':main()
