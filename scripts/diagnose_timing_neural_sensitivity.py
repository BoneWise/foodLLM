"""Refit brain maps omitting timing-incomplete trials; reuse unchanged wanting q.

Human ratings, outer folds, and all upstream readout training remain unchanged.
Only affected participants' brain fits are recomputed. This diagnostic does
not replace the original analysis or set a general neural QC threshold.
"""
import argparse,csv,json,shutil,subprocess
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from feature_inputs import load_layer_inputs
from run_layer_analysis import brain_from_q
from prepare_roi import sha,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flags',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/layer_estimation_v01.json').read_text());parent=Path(cfg['output'])
    with Path(args.flags).open() as f:flags=list(csv.DictReader(f))
    records=[];count_records=[]
    with threadpool_limits(limits=2):
        for sample in cfg['samples']:
            for prompt in cfg['prompts']:
                data=load_layer_inputs(cfg['matrix_root'],cfg['analysis_root'],prompt,1,sample)
                beta=data['roi_beta'].copy();affected=set();n_omitted=0
                for f in flags:
                    if f['image_id'] not in data['image_ids']:continue
                    s=data['subject_ids'].index(f['subject_id']);i=data['image_ids'].index(f['image_id'])
                    assert data['session_ids'][s,i]==f['session_id']
                    assert data['run_ids'][s,i]==f['session_id']+'/'+f['run_id']
                    beta[s,i]=np.nan;affected.add(s);n_omitted+=1
                for layer in cfg['layers']:
                    name=f'{sample}_{prompt}_layer{layer:02d}'
                    original=np.load(parent/(name+'.npz'));meta=json.loads((parent/(name+'.json')).read_text())
                    predictions=original['brain_predictions'].copy();conditions=meta['conditions']
                    counts=np.zeros((2,len(affected),beta.shape[1]),int)
                    for sj,s in enumerate(sorted(affected)):
                        predictions[:,s,:,:]=np.nan
                        for fold in sorted(set(data['outer_folds'])):
                            train=np.flatnonzero(data['outer_folds']!=fold);test=np.flatnonzero(data['outer_folds']==fold)
                            q={c:original[f'q_fold{fold}_{c}'][:,s] for c in conditions}
                            for mode in range(2):
                                held=[None] if mode==0 else sorted(set(data['session_ids'][s,np.isfinite(beta[s])]))
                                for session in held:
                                    b,p,_=brain_from_q(q,beta[s],data['content_covariates'],data['raw_V_probs'],
                                        train,test,data['run_ids'][s],data['session_ids'][s],session)
                                    counts[mode,sj,b]+=1
                                    for k,c in enumerate(meta['brain_conditions']):predictions[mode,s,k,b]=p[c]
                        assert np.array_equal(counts[:,sj],np.broadcast_to(np.isfinite(beta[s]),counts[:,sj].shape))
                    errors=(predictions-beta[None,:,None,:])**2
                    mse=np.nanmean(errors,axis=-1).mean(axis=1)
                    for k,mode in enumerate(meta['brain_modes']):
                        m=dict(zip(meta['brain_conditions'],mse[k].tolist()))
                        m['G']=m['V']-m['T'];m['T_vs_baseline']=m['baseline']-m['T']
                        if len(conditions)>2:
                            m['P_mean']=float(np.mean([m[c] for c in conditions if c.startswith('P')]))
                            m['S']=m['P_mean']-m['T']
                        records.append({'sample':sample,'prompt':prompt,'layer':layer,'mode':mode,**m})
                    np.savez_compressed(out/(name+'.npz'),brain_predictions=predictions,brain_squared_errors=errors)
                    count_records.append({'task':name,'observed_trials':int(np.isfinite(beta).sum()),'omitted':n_omitted})
                print(f'Completed brain timing sensitivity: {sample} {prompt}',flush=True)
    keys=['sample','prompt','layer','mode','baseline','V','T','P1','P2','P3','P4','P5','G','S','P_mean','T_vs_baseline']
    with (out/'layer_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    summaries=[]
    for sample in cfg['samples']:
        for prompt in cfg['prompts']:
            for mode in ['food','food_session']:
                rr=[r for r in records if (r['sample'],r['prompt'],r['mode'])==(sample,prompt,mode)]
                m={key:float(np.mean([r[key] for r in rr])) for key in rr[0] if key not in ['sample','prompt','layer','mode']}
                m['G_percent']=100*m['G']/m['V'];m['T_vs_baseline_percent']=100*m['T_vs_baseline']/m['baseline']
                if 'S' in m:m['S_percent']=100*m['S']/m['P_mean']
                summaries.append({'sample':sample,'prompt':prompt,'mode':mode,**m})
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n');shutil.copy2(__file__,out/'script_snapshot.py')
    audit={'status':'complete','diagnostic_only':True,'upstream_q_reused_because_human_ratings_and_folds_unchanged':True,
        'affected_subjects':sorted({r['subject_id'] for r in flags}),'task_counts':count_records,
        'flags_sha256':sha(args.flags),'parent_audit_sha256':sha(parent/'audit.json'),'script_sha256':sha(__file__),
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'seed':None,
        'no_original_inputs_or_results_overwritten':True,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
