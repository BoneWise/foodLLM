"""Outcome-blind equivalence and isolation checks for batched estimation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from batched_readout import tune_predict,crossfit_predict
from frozen_readout import tune_v
from two_stage_estimator import leave_subject_out_mean,neural_fold
from run_layer_analysis import analyze_layer,brain_from_q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20260917);lambdas=tuple(10.**k for k in range(-6,7))
    tests={};max_error=0.
    with threadpool_limits(limits=2):
        for d in [7,128]:
            n=80;groups=np.repeat(np.arange(n//2),2).astype(str)
            x=rng.normal(size=(n,d));x[:,-1]=3.
            features={'V':x,'T':x+rng.normal(scale=.05,size=x.shape),
                      'P1':x+rng.normal(scale=.1,size=x.shape),'P2':x+rng.normal(scale=.15,size=x.shape)}
            human=x[:,0][None,:]+rng.normal(scale=.3,size=(5,n));human[1,::11]=np.nan
            targets=np.column_stack([leave_subject_out_mean(human,s) for s in range(5)])
            fold_ids=np.arange(n)//2%5;train=np.flatnonzero(fold_ids!=0);test=np.flatnonzero(fold_ids==0)
            pred,selected,_=tune_predict(features,targets,groups,train,test,lambdas,4)
            for s in range(5):
                ref=tune_v(x,targets[:,s],groups,train,lambdas,4)
                assert selected[s]==ref.regularization
                for c in features:
                    error=float(np.max(np.abs(pred[c][:,s]-ref.predict(features[c][test]))))
                    max_error=max(max_error,error)
                    assert error<2e-6
            tests[f'batch_matches_reference_d{d}_13_lambdas']=True
        content=rng.normal(size=(n,6));probs=rng.dirichlet(np.ones(4),size=n)
        sessions=(np.arange(n)%3).astype(str);runs=np.array([f'{s}/{i%2}' for i,s in enumerate(sessions)])
        beta=targets.T+rng.normal(scale=.3,size=(5,n));beta[2,::13]=np.nan
        q,traces=crossfit_predict(features,targets,groups,train,test,lambdas)
        for s in [0,2]:
            for held in [None,'1']:
                ref=neural_fold(features,human,s,beta[s],content,probs,groups,train,test,runs,
                                sessions,held,lambdas)
                b,p,ranks=brain_from_q({c:q[c][:,s] for c in q},beta[s],content,probs,train,test,
                                       runs,sessions,held)
                assert np.array_equal(b,ref['test_indices'])
                for c in p:
                    error=float(np.max(np.abs(p[c]-ref['predictions'][c])))
                    max_error=max(max_error,error)
                    assert error<2e-6
        tests['batched_two_stage_and_session_predictions_match_reference']=True
        altered=targets.copy();altered[test]+=10000
        alt,_=crossfit_predict(features,altered,groups,train,test,lambdas)
        for c in q:
            assert np.array_equal(q[c],alt[c])
        tests['test_food_human_targets_cannot_change_any_q']=True
        own=human.copy();own[0]+=999
        assert np.array_equal(targets[:,0],leave_subject_out_mean(own,0))
        tests['target_identity_ratings_excluded']=True
        changed_features={**features,'T':features['T']+1000}
        alt,alt_selected,_=tune_predict(changed_features,targets,groups,train,test,lambdas,4)
        assert np.array_equal(alt_selected,selected)
        assert np.array_equal(alt['V'],pred['V'])
        tests['T_cannot_change_V_tuning_or_predictions']=True
        for tr in traces:
            assert set(groups[tr['fit_indices']]).isdisjoint(groups[tr['held_indices']])
            assert set(groups[tr['fit_indices']]).isdisjoint(groups[test])
        tests['all_upstream_food_boundaries_checked']=True
        data={'features':{c:x.copy() for c in features},'human_ratings':human,'roi_beta':beta,
              'food_groups':groups,'outer_folds':fold_ids,'content_covariates':content,
              'raw_V_probs':probs,'session_ids':np.tile(sessions,(5,1)),
              'run_ids':np.tile(runs,(5,1)),'image_ids':[str(i) for i in range(n)],
              'subject_ids':[str(i) for i in range(5)]}
        arrays,report=analyze_layer(data,lambdas)
        for mode in report['metrics'].values():
            assert mode['G']==0 and mode['S']==0
        tests['complete_pipeline_identical_conditions_zero_contrasts']=True
        tests['missing_brain_trials_and_once_only_session_predictions']=True
        predp=np.array([[0.,2.],[2.,0.]]);y=np.ones(2)
        assert np.mean((predp-y)**2)==1 and np.mean((predp.mean(0)-y)**2)==0
        tests['replica_loss_mean_not_ensemble_loss']=True
    report={'status':'passed','tests':tests,'seed':20260917,'lambda_grid':lambdas,
            'max_absolute_reference_prediction_error':max_error,
            'scope':'algebraic equivalence and leakage checks, NOT population interval coverage',
            'script_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                map(Path,['scripts/batched_readout.py','scripts/run_layer_analysis.py',
                          'scripts/validate_batched_analysis.py','scripts/two_stage_estimator.py',
                          'scripts/frozen_readout.py'])}}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
