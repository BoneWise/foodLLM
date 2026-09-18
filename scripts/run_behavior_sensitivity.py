"""Previously listed condition-specific readouts; post-estimate bias diagnostics.

Condition-specific results never replace frozen-V primary comparisons. Bias
decomposition is explicitly descriptive and was added after viewing primary
estimates. No outcome-dependent layer or prompt selection occurs.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from batched_readout import tune_predict
from feature_inputs import load_layer_inputs
from frozen_readout import fit_at_lambda
from two_stage_estimator import leave_subject_out_mean
from run_layer_analysis import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False);rows=[];diagnostics=[];reference_errors=[]
    start=time.time()
    with threadpool_limits(limits=2):
        for sample in config['samples']:
            for prompt in config['prompts']:
                for layer in config['layers']:
                    d=load_layer_inputs(config['matrix_root'],config['analysis_root'],prompt,layer,sample)
                    y=np.nanmean(d['human_ratings'],axis=0);groups=d['food_groups'];folds=d['outer_folds']
                    conditions=list(d['features']);pred=np.full((len(conditions),len(y)),np.nan)
                    lambdas={c:[] for c in conditions}
                    name=f'{sample}_{prompt}_layer{layer:02d}'
                    original=np.load(Path(config['output'])/(name+'.npz'))
                    meta=json.loads((Path(config['output'])/(name+'.json')).read_text())
                    for k,c in enumerate(conditions):
                        # V was already fitted exactly this way: reuse, never reselect it.
                        if c=='V':
                            pred[k]=original['behavior_predictions'][0]
                            lambdas[c]=[t['behavior_lambda'] for t in meta['fold_traces']]
                        else:
                            for fold in sorted(set(folds)):
                                train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                                p,chosen,_=tune_predict({'V':d['features'][c]},y,groups,train,test,config['lambdas'],4)
                                pred[k,test]=p['V'][:,0];lambdas[c].append(float(chosen[0]))
                        mse=float(np.mean((pred[k]-y)**2))
                        rows.append({'sample':sample,'prompt':prompt,'layer':layer,'condition':c,'MSE':mse})
                        p=original['behavior_predictions'][k];bias=float(np.mean(p-y))
                        original_mse=float(np.mean((p-y)**2))
                        centered=float(np.mean(((p-p.mean())-(y-y.mean()))**2))
                        assert abs(original_mse-bias**2-centered)<1e-10
                        diagnostics.append({'sample':sample,'prompt':prompt,'layer':layer,'condition':c,
                            'MSE':original_mse,'mean_prediction':float(p.mean()),'mean_target':float(y.mean()),
                            'bias':bias,'bias_squared':bias**2,'centered_MSE':centered,
                            'correlation':float(np.corrcoef(p,y)[0,1]),
                            'prediction_min':float(p.min()),'prediction_max':float(p.max())})
                    # Fixed endpoint layers and first outer fold: actual-data numerical check.
                    if sample=='primary' and layer in [1,32]:
                        trace=meta['fold_traces'][0];fold=trace['outer_fold']
                        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                        targets=[y,leave_subject_out_mean(d['human_ratings'],0)]
                        lams=[trace['behavior_lambda'],trace['neural_fits'][-1]['lambda_by_target'][0]]
                        for j,(target,lam) in enumerate(zip(targets,lams)):
                            fitted=fit_at_lambda(d['features']['V'][train],target[train],groups[train],lam)
                            for k,c in enumerate(conditions):
                                expected=(original['behavior_predictions'][k,test] if j==0 else
                                          original[f'q_fold{fold}_{c}'][test,0])
                                error=float(np.max(np.abs(fitted.predict(d['features'][c][test])-expected)))
                                reference_errors.append(error)
                                assert error<2e-5
                    np.savez_compressed(out/(name+'.npz'),condition_specific_predictions=pred)
                    (out/(name+'.json')).write_text(json.dumps({'conditions':conditions,'lambdas':lambdas},indent=2)+'\n')
                    print(name+' complete',flush=True)
    for filename,records in [('condition_specific_metrics.csv',rows),('frozen_transfer_bias_diagnostics.csv',diagnostics)]:
        with (out/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summaries=[]
    for sample in config['samples']:
        for prompt in config['prompts']:
            conditions=list(dict.fromkeys(r['condition'] for r in rows if r['sample']==sample))
            metrics={c:float(np.mean([r['MSE'] for r in rows if
                (r['sample'],r['prompt'],r['condition'])==(sample,prompt,c)])) for c in conditions}
            metrics['G']=metrics['V']-metrics['T']
            if 'P1' in metrics:
                metrics['P_mean']=float(np.mean([metrics[c] for c in conditions if c.startswith('P')]))
                metrics['S']=metrics['P_mean']-metrics['T']
            metrics['G_percent_of_V']=100*metrics['G']/metrics['V']
            summaries.append({'sample':sample,'prompt':prompt,**metrics})
    (out/'mean32_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    shutil.copy2(__file__,out/'run_behavior_sensitivity.py')
    (out/'audit.json').write_text(json.dumps({'status':'complete','elapsed_seconds':time.time()-start,
        'scope':'pre-existing secondary condition-specific readouts; post-estimate bias diagnostics',
        'not_a_replacement_for_primary':True,'intervals_or_p_values':False,
        'actual_data_reference_check_count':len(reference_errors),
        'actual_data_max_absolute_reference_error':max(reference_errors),
        'script_sha256':sha(__file__),'config_sha256':sha(args.config),
        'primary_analysis_audit_sha256':sha(Path(config['output'])/'audit.json'),
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
