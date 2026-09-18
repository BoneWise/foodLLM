"""Outcome-blind integration checks for the neural fold implementation."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from two_stage_estimator import neural_fold


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20260916)
    n=120;groups=np.repeat(np.arange(60),2).astype(str)
    x=rng.normal(size=(n,6));want=x[:,0]+0.2*x[:,1]
    human=want[None,:]+rng.normal(scale=0.2,size=(5,n));human[1,::11]=np.nan
    content=rng.normal(size=(n,6));probs=rng.dirichlet(np.ones(4),size=n)
    sessions=(np.arange(n)%3).astype(str);runs=np.array([f'{s}/r{j%2}' for j,s in enumerate(sessions)])
    beta=want+0.2*content[:,0]+rng.normal(scale=0.1,size=n)
    test=np.flatnonzero(np.arange(n)//2%5==0);train=np.setdiff1d(np.arange(n),test)
    features={'V':x,'T':x.copy(),'P1':x.copy(),'P2':x.copy()}
    def fit(h=human,b=beta,held=None):
        return neural_fold(features,h,0,b,content,probs,groups,train,test,runs,
                           session_ids=sessions,held_session=held,lambdas=(0.001,0.1,10.))
    result=fit()
    assert np.array_equal(result['G'],np.zeros(len(test)))
    assert np.array_equal(result['S'],np.zeros(len(test)))
    altered_h=human.copy();altered_h[:,test]+=10000
    altered_b=beta.copy();altered_b[test]+=10000
    changed=fit(altered_h,altered_b)
    for c in result['predictions']:
        assert np.array_equal(result['predictions'][c],changed['predictions'][c])
    own=human.copy();own[0]+=9999;own_changed=fit(own)
    for c in result['predictions']:
        assert np.array_equal(result['predictions'][c],own_changed['predictions'][c])
    seen=[]
    for session in ['0','1','2']:
        fold=fit(held=session);seen.extend(fold['test_indices'])
        assert np.all(sessions[fold['train_indices']]!=session)
        assert np.all(sessions[fold['test_indices']]==session)
        assert fold['run_levels']==[]
        other_beta=beta.copy();other_beta[sessions==session]+=7777
        perturbed=fit(b=other_beta,held=session)
        for c in fold['predictions']:
            assert np.array_equal(fold['predictions'][c],perturbed['predictions'][c])
    assert sorted(seen)==test.tolist()
    for trace in result['crossfit_traces']:
        assert set(groups[trace['fit_indices']]).isdisjoint(groups[trace['held_indices']])
        assert set(groups[trace['fit_indices']]).isdisjoint(groups[test])
    tests={'identical_conditions_zero_G_and_S':True,
        'outer_test_human_and_brain_outcomes_cannot_change_predictions':True,
        'target_subject_own_ratings_cannot_change_predictions':True,
        'food_session_train_test_disjoint_and_once_only':True,
        'held_session_brain_values_cannot_change_predictions':True,
        'upstream_crossfit_groups_excluded':True,
        'missing_non_target_ratings_handled':True}
    root=Path(__file__).resolve().parents[1]
    report={'tests':tests,'seed':20260916,'scope':'synthetic core integration only; no actual effects or interval coverage',
        'synthetic_lambda_grid':[0.001,0.1,10.],'python':platform.python_version(),'numpy':np.__version__,
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'script_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/two_stage_estimator.py','scripts/frozen_readout.py','scripts/validate_two_stage_core.py']}}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(tests))


if __name__=='__main__':
    main()
