"""RSA v06.1: raw fixed patterns, Prompt A then B; descriptive results only."""
import argparse,json,shutil,subprocess,platform
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from prepare_roi import sha,write_csv
from extract_rsa_patterns import rows
from rsa_core import correlation_rdm,ranked_unit,score,validate

ROOT=Path(__file__).resolve().parents[1]
CONDS=['V','T','P1','P2','P3','P4','P5']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--patterns',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'outputs') and not out.exists();out.mkdir(parents=True)
    cfgpath=ROOT/json.loads((ROOT/'configs/project.json').read_text())['current_analysis_config'];cfg=json.loads(cfgpath.read_text())
    assert cfg['version']=='v06.1' and not cfg['neural_inference']['run_in_first_pass']
    assert cfg['model']['conditions']==CONDS
    patterns=ROOT/args.patterns;pa=json.loads((patterns/'audit.json').read_text());assert pa['status']=='complete'
    pix=json.loads((patterns/'index.json').read_text());ids=pix['image_ids'];subjects=pix['subject_ids'];n=len(ids);lower=np.tril_indices(n,-1)
    inputs={str(p):sha(p) for p in [cfgpath,patterns/'audit.json',patterns/'index.json',Path(__file__),ROOT/'scripts/rsa_core.py']}
    shutil.copy2(cfgpath,out/'config_snapshot.json');shutil.copy2(__file__,out/'script_snapshot.py');shutil.copy2(ROOT/'scripts/rsa_core.py',out/'rsa_core_snapshot.py')
    numerical=validate();(out/'numerical_validation.json').write_text(json.dumps(numerical,indent=2)+'\n')
    idxroot=ROOT/cfg['inputs']['analysis_index']; idx=json.loads((idxroot/'index.json').read_text());globalrows=[idx['image_ids'].index(i) for i in ids]
    h=np.load(idxroot/'human_ratings.npy')[:,globalrows];wanting=np.nanmean(h,axis=0)
    kcal=np.load(idxroot/'content_covariates.npy')[globalrows,idx['content_columns'].index('correct_kcal_100g')]
    for name in ['index.json','human_ratings.npy','content_covariates.npy']:inputs[str(idxroot/name)]=sha(idxroot/name)
    wdm=np.abs(wanting[:,None]-wanting[None,:]);kdm=np.abs(kcal[:,None]-kcal[None,:]);wrank=ranked_unit(wdm[lower])
    np.save(out/'wanting_RDM.npy',wdm);np.save(out/'kcal_RDM.npy',kdm)
    (out/'image_index.json').write_text(json.dumps({'image_ids':ids,'subject_ids':subjects,'layers':list(range(1,33)),'conditions':CONDS},indent=2)+'\n')
    ref={'kcal_to_wanting_rho':score(kdm[lower],wdm[lower])};brains=[];qc=[];masks={};brain_dir=out/'brain';brain_dir.mkdir()
    for sub in subjects:
        p=patterns/sub/'patterns.npy';assert sha(p)==pa['output_sha256'][str(p.relative_to(patterns))]
        tt=rows(patterns/sub/'trials.csv');ii=np.array([ids.index(r['image_id']) for r in tt]);assert len(set(ii))==len(ii)
        b=correlation_rdm(np.load(p));tri=np.tril_indices(len(ii),-1);v=b[tri];brank=ranked_unit(v)
        subdir=brain_dir/sub;subdir.mkdir();np.save(subdir/'RDM.npy',b);np.save(subdir/'global_image_rows.npy',ii)
        gr,gc=ii[tri[0]],ii[tri[1]];np.save(subdir/'pair_global_rows.npy',np.stack([gr,gc]))
        sess=np.array([r['session_id'] for r in tt]);run=np.array([r['run_identity'] for r in tt])
        ds=(sess[:,None]!=sess[None,:]).astype(np.uint8);dr=(run[:,None]!=run[None,:]).astype(np.uint8)
        np.save(subdir/'session_RDM.npy',ds);np.save(subdir/'run_RDM.npy',dr)
        rec={'subject_id':sub,'images':len(ii),'voxels':np.load(p,mmap_mode='r').shape[1],'pairs':len(v),'kcal_rho':score(kdm[gr,gc],v)}
        for name,d in [('session',ds),('run',dr)]:
            pair=d[tri].astype(bool);rec[name+'_rho']=score(d[tri],v)
            for label,sel in [('same',~pair),('different',pair)]:
                rec[f'{name}_{label}_pairs']=int(sel.sum());rec[f'{name}_{label}_mean_distance']=float(v[sel].mean())
        qc.append(rec);mk=tuple(ii.tolist());masks[mk]=(gr,gc)
        brains.append({'subject':sub,'rows':gr,'cols':gc,'ranks':brank,'maskkey':mk,'vector':v})
    write_csv(out/'acquisition_QC.csv',qc);(out/'references.json').write_text(json.dumps(ref,indent=2)+'\n')
    print(f'Brain RDMs: {len(brains)}; shared image-mask variants: {len(masks)}; numerical checks passed',flush=True)
    original_audit=ROOT/cfg['inputs']['model_matrices']/'audit.json';original=json.loads(original_audit.read_text());inputs[str(original_audit)]=sha(original_audit)
    expected={rel:info['sha256'] for group in original['groups'] for rel,info in group['files'].items()}
    brain_results=[];behavior_results=[];contrasts=[];group_results=[];validation={};prompt_scores={}
    for prompt in ['A','B']:
        if prompt=='B': assert validation['A']['all_scores_finite'] and validation['A']['reference_checks_passed']
        result=np.empty((len(subjects),32,7));behavior=np.empty((32,7));errors=[]
        pdir=out/prompt;pdir.mkdir()
        for ci,condition in enumerate(CONDS):
            source=ROOT/cfg['inputs']['model_matrices']/f'{prompt}_{condition}'
            for name in ['hidden.npy','image_index.csv']:
                digest=sha(source/name);assert digest==expected[f'{prompt}_{condition}/{name}'];inputs[str(source/name)]=digest
            meta=rows(source/'image_index.csv');lookup={r['image_id']:int(r['matrix_row']) for r in meta};assert len(lookup)==len(meta)
            take=np.array([lookup[i] for i in ids]);features=np.load(source/'hidden.npy',mmap_mode='r')
            modelrdm=np.lib.format.open_memmap(pdir/f'{condition}_RDM.npy',mode='w+',dtype='float64',shape=(32,n,n))
            for li in range(32):
                d=correlation_rdm(features[li,take]);modelrdm[li]=d
                behavior[li,ci]=ranked_unit(d[lower])@wrank
                ranks={mk:ranked_unit(d[rr,cc]) for mk,(rr,cc) in masks.items()}
                for si,b in enumerate(brains): result[si,li,ci]=float(ranks[b['maskkey']]@b['ranks'])
                if li in [0,15,31]:
                    # Independent direct SciPy check with identical stimuli/pairs, for every condition.
                    for si in [0,subjects.index('sub-016'),28]:
                        b=brains[si];reference=spearmanr(d[b['rows'],b['cols']],b['vector']).statistic
                        errors.append(abs(reference-result[si,li,ci]))
                    errors.append(abs(spearmanr(d[lower],wdm[lower]).statistic-behavior[li,ci]))
            modelrdm.flush();del modelrdm
            print(f'{prompt} {condition}: 32 layers computed, all 29 participants',flush=True)
        assert np.isfinite(result).all() and np.isfinite(behavior).all() and max(errors)<1e-12
        np.save(pdir/'neural_scores.npy',result);np.save(pdir/'behavior_scores.npy',behavior)
        pmean=result[:,:,2:].mean(axis=2);delta=np.stack([result[:,:,1]-result[:,:,0],result[:,:,1]-pmean],axis=2)
        np.save(pdir/'contrasts.npy',delta)
        for si,sub in enumerate(subjects):
            for li in range(32):
                for ci,c in enumerate(CONDS):brain_results.append({'prompt':prompt,'subject_id':sub,'layer':li+1,'condition':c,'rho':float(result[si,li,ci])})
                contrasts.append({'prompt':prompt,'subject_id':sub,'layer':li+1,'T_minus_V':float(delta[si,li,0]),'T_minus_mean_P':float(delta[si,li,1]),'P_score_mean':float(pmean[si,li]),'P_score_min':float(result[si,li,2:].min()),'P_score_max':float(result[si,li,2:].max())})
        for li in range(32):
            for ci,c in enumerate(CONDS):behavior_results.append({'prompt':prompt,'layer':li+1,'condition':c,'rho':float(behavior[li,ci])})
            mean=result[:,li].mean(axis=0)
            group_results.append({'prompt':prompt,'layer':li+1,'V':float(mean[0]),'T':float(mean[1]),'P_mean':float(mean[2:].mean()),'P_min_group':float(mean[2:].min()),'P_max_group':float(mean[2:].max()),'T_minus_V':float(delta[:,li,0].mean()),'T_minus_mean_P':float(delta[:,li,1].mean()),'participants_positive_TV':int((delta[:,li,0]>0).sum()),'participants_positive_TP':int((delta[:,li,1]>0).sum()),'wanting_V':float(behavior[li,0]),'wanting_T':float(behavior[li,1]),'wanting_P_mean':float(behavior[li,2:].mean())})
        validation[prompt]={'all_scores_finite':True,'reference_checks_passed':True,'max_scipy_difference':max(errors),'direct_reference_comparisons':len(errors),'shape':list(result.shape),'B_gate':'correctness_only_independent_of_effect'}
        (pdir/'validation.json').write_text(json.dumps(validation[prompt],indent=2)+'\n');prompt_scores[prompt]=result
        print(f'{prompt} complete; correctness gate passed, max independent difference {max(errors):.3g}',flush=True)
    for name,rr in [('neural_scores',brain_results),('behavior_scores',behavior_results),('contrasts',contrasts),('group_curves',group_results)]:write_csv(out/(name+'.csv'),rr)
    audit={'status':'complete','created_at':datetime.now().astimezone().isoformat(),'protocol':'v06.1','post_result_follow_up':True,'subjects':29,'images':493,'brain_trials':13765,'brain_RDMs':29,'model_RDMs':448,'prompt_order':['A','B'],'neural_score_count':len(brain_results),'primary_contrasts':['T-V','T-mean_P_scores'],'extra_brain_preprocessing':[],'pair_masks_identical_across_layers_conditions_prompts':True,'inference_run':False,'sensitivity_run':False,'subject_equal_weighting':True,'input_sha256':inputs,'validation':validation,'seed':None,'fixture_seed_only':2026091701,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print('RSA complete; no inferential tests or sensitivity analyses run.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
