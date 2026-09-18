"""Frozen V-output transfer baselines, on the exact main and full-VT food folds."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from batched_readout import tune_predict
from feature_inputs import load_layer_inputs
from run_layer_analysis import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False);rows=[]
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    inputs={}
    with threadpool_limits(limits=2):
        for sample in cfg['samples']:
            for prompt in cfg['prompts']:
                d=load_layer_inputs(cfg['matrix_root'],cfg['analysis_root'],prompt,1,sample)
                probs={};n=len(d['image_ids']);y=np.nanmean(d['human_ratings'],axis=0)
                for c in d['features']:
                    root=Path(cfg['matrix_root'])/f'{prompt}_{c}'
                    with (root/'image_index.csv').open() as f:
                        mapping={r['image_id']:int(r['matrix_row']) for r in csv.DictReader(f)}
                    indices=[mapping[i] for i in d['image_ids']]
                    probs[c]=np.load(root/'rating_probs.npy')[indices]
                    inputs[str(root/'rating_probs.npy')]=sha(root/'rating_probs.npy')
                expected={c:p@np.arange(1,5) for c,p in probs.items()}
                raw=np.stack(list(expected.values()))
                mean_pred=np.full(n,np.nan);calibrated={};trace={}
                for kind,features in [('expected',{c:v[:,None] for c,v in expected.items()}),('probabilities',probs)]:
                    pred=np.full((len(features),n),np.nan);trace[kind]=[]
                    for fold in sorted(set(d['outer_folds'])):
                        train=np.flatnonzero(d['outer_folds']!=fold);test=np.flatnonzero(d['outer_folds']==fold)
                        mean_pred[test]=y[train].mean()
                        p,chosen,_=tune_predict(features,y,d['food_groups'],train,test,cfg['lambdas'],4)
                        for k,c in enumerate(features):
                            pred[k,test]=p[c][:,0]
                        trace[kind].append({'fold':int(fold),'lambda':float(chosen[0])})
                    calibrated[kind]=pred
                arrays={'target':y,'raw_expected':raw,'training_mean':mean_pred,**calibrated}
                conditions=list(probs)
                for kind in ['raw_expected',*calibrated]:
                    mse=((arrays[kind]-y)**2).mean(1)
                    m=dict(zip(conditions,mse.tolist()))
                    m['G']=m['V']-m['T']
                    if len(conditions)>2:
                        m['P_mean']=float(mse[2:].mean());m['S']=m['P_mean']-m['T']
                    rows.append({'sample':sample,'prompt':prompt,'baseline':kind,
                                 'training_mean_MSE':float(np.mean((mean_pred-y)**2)),**m})
                np.savez_compressed(out/f'{sample}_{prompt}.npz',**arrays)
                (out/f'{sample}_{prompt}.json').write_text(json.dumps({'conditions':conditions,
                    'image_ids':d['image_ids'],'fold_tuning':trace},indent=2)+'\n')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['sample','prompt','baseline','training_mean_MSE',
            'V','T','P1','P2','P3','P4','P5','P_mean','G','S']);w.writeheader();w.writerows(rows)
    (out/'audit.json').write_text(json.dumps({'status':'complete','started_utc':started,
        'scope':'descriptive_output_baselines_no_population_intervals',
        'config_sha256':sha(args.config),'probability_input_sha256':inputs,
        'script_sha256':sha(__file__),'comparison':'same food folds and sample; frozen V calibration',
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n')
    print('Completed output baselines for A/B and both samples.')


if __name__=='__main__':
    main()
