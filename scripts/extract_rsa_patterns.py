"""Read-only provenance checks and raw vmPFC pattern extraction for RSA v06.1."""
import argparse, csv, json, re, shutil, platform, subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import nibabel as nib
import scipy
from scipy.io import loadmat
from prepare_roi import sha, write_csv

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    assert out.is_relative_to(ROOT/'outputs') and not out.exists()
    out.mkdir(parents=True)
    cfg_path=ROOT/json.loads((ROOT/'configs/project.json').read_text())['current_analysis_config']
    cfg=json.loads(cfg_path.read_text()); assert cfg['version']=='v06.1'
    shutil.copy2(cfg_path,out/'config_snapshot.json');shutil.copy2(__file__,out/'script_snapshot.py')
    ledger=rows(ROOT/cfg['inputs']['brain_eligibility']); mapping=rows(ROOT/cfg['inputs']['beta_mapping'])
    keys=['subject_id','session_id','run_id','trial_id','image_id']; key=lambda r:tuple(r[k] for k in keys)
    mapped={key(r):r for r in mapping}; assert len(mapped)==len(mapping)==15869
    idxdir=ROOT/cfg['inputs']['analysis_index']; index=json.loads((idxdir/'index.json').read_text())
    mask=np.load(idxdir/'primary_mask.npy'); ids=[i for i,p in zip(index['image_ids'],mask) if p]
    assert len(ids)==493; order={i:n for n,i in enumerate(ids)}
    selected=[{**mapped[key(r)],**r} for r in ledger if r['neural_analysis_eligible']=='True' and r['image_id'] in order]
    assert len(selected)==13765
    input_hash={}
    for p in [cfg_path,Path(__file__),idxdir/'index.json',idxdir/'primary_mask.npy',ROOT/cfg['inputs']['brain_eligibility'],ROOT/cfg['inputs']['beta_mapping']]: input_hash[str(p)]=sha(p)
    source=Path('/path/to/external_data/minimal_lsa/run_minimal_lsa.m'); archived=ROOT/'archive/source_code/run_minimal_lsa.m'
    assert sha(source)==sha(archived)
    source_text=source.read_text(); assert 'spm.spatial.smooth' not in source_text and 'spm_smooth' not in source_text
    assert 'desc-preproc_bold' in source_text
    input_hash[str(source)]=sha(source);shutil.copy2(source,out/'source_LSA_snapshot.m')
    old_path=ROOT/'outputs/20260916_实际SPM与beta复核_v01/audit.json'
    old=json.loads(old_path.read_text()); input_hash[str(old_path)]=sha(old_path)
    old_hashes={r['path']:r for r in old['external_input_hashes']}
    grouped=defaultdict(list)
    for r in selected: grouped[r['spm_mat_local']].append(r)
    provenance=[]
    for n,(path,rr) in enumerate(sorted(grouped.items())):
        digest=sha(path); assert digest==old_hashes[path]['sha256']
        s=loadmat(path,simplify_cells=True)['SPM']
        scans=[str(x).strip().rsplit(',',1) for x in np.atleast_1d(s['xY']['P'])]
        files={p[0] for p in scans};assert len(files)==1
        bold=Path(scans[0][0]); r=rr[0]
        stem=f'{r["subject_id"]}_{r["session_id"]}_task-food_run-{r["run_id"]}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold'
        assert bold.name==stem+'.nii' and bold.is_file()
        assert [int(p[1]) for p in scans]==list(range(1,len(scans)+1))
        conditions=s['Sess']['U']; conditions=[conditions] if isinstance(conditions,dict) else list(conditions)
        u={str(np.atleast_1d(x['name'])[0]):x for x in conditions}; names=list(np.atleast_1d(s['xX']['name']))
        for r in rr:
            col=int(re.search(r'beta_(\d+)',r['beta_file_local'])[1])-1
            target=f'target_trial_{int(r["trial_id"]):04d}'
            assert names[col]==f'Sn(1) {target}*bf(1)'
            assert Path(s['Vbeta'][col]['fname']).name==Path(r['beta_file_local']).name
            assert abs(float(np.atleast_1d(u[target]['ons'])[0])-float(r['onset_beta']))<1e-6
            assert u[target]['P']['name']=='none'
        provenance.append({'subject_id':rr[0]['subject_id'],'session_id':rr[0]['session_id'],'run_id':rr[0]['run_id'],
            'SPM_path':path,'SPM_sha256':digest,'input_BOLD':str(bold),'selected_trials':len(rr),'source_no_spatial_smoothing':True})
        if n%50==0: print(f'Unsmoothed source / SPM identity: {n+1}/{len(grouped)}',flush=True)
    write_csv(out/'run_provenance.csv',provenance)
    subject_summaries=[]; trial_records=[]; mean_error=0.; processed=0
    for sub in index['subject_ids']:
        rr=sorted([r for r in selected if r['subject_id']==sub],key=lambda r:order[r['image_id']])
        support_path=ROOT/cfg['inputs']['fixed_roi_support']/f'{sub}_fixed_support.nii.gz'
        support=nib.load(support_path); data=support.get_fdata(); assert set(np.unique(data))<={0,1}
        coords=np.argwhere(data>0); assert len(coords)>1
        # np.argwhere ordering is C order; bounding-box Boolean selection has identical order.
        lo=coords.min(0); hi=coords.max(0)+1; slices=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        local=data[slices]>0
        assert np.array_equal(np.argwhere(local)+lo,coords)
        arr=np.empty((len(rr),len(coords)),dtype=np.float32); meta=[]
        input_hash[str(support_path)]=sha(support_path)
        for j,r in enumerate(rr):
            image=nib.load(r['beta_file_local'])
            assert image.shape==support.shape and np.allclose(image.affine,support.affine,rtol=0,atol=1e-5)
            vals=np.asarray(image.dataobj[slices])[local]
            assert np.isfinite(vals).all() and float(np.var(vals.astype(np.float64)))>0, key(r)
            assert np.array_equal(vals,vals.astype(np.float32)), 'Float32 storage must be lossless'
            assert len(vals)==int(r['fixed_support_voxels'])
            error=abs(float(vals.astype(np.float32).mean())-float(r['roi_beta_mean']))
            assert error==0, (key(r),error);mean_error=max(mean_error,error)
            arr[j]=vals
            rec={k:r[k] for k in keys};rec.update(pattern_row=j,global_image_row=order[r['image_id']],run_identity=r['session_id']+'/'+r['run_id'],beta_file=r['beta_file_local'])
            meta.append(rec);trial_records.append({**rec,'voxel_count':len(vals),'pattern_variance':float(np.var(vals.astype(np.float64))),'mean_ledger_absolute_error':error})
            processed+=1
        subdir=out/sub;subdir.mkdir();np.save(subdir/'patterns.npy',arr);np.save(subdir/'voxel_ijk.npy',coords);np.save(subdir/'affine.npy',support.affine)
        write_csv(subdir/'trials.csv',meta)
        subject_summaries.append({'subject_id':sub,'images':len(rr),'voxels':len(coords),'min_pattern_variance':float(np.min(np.var(arr.astype(np.float64),axis=1))),'patterns_sha256':sha(subdir/'patterns.npy')})
        print(f'Patterns {sub}: {len(rr)} trials x {len(coords)} voxels; total {processed}',flush=True)
    write_csv(out/'trial_QC.csv',trial_records);write_csv(out/'subject_QC.csv',subject_summaries)
    (out/'index.json').write_text(json.dumps({'image_ids':ids,'subject_ids':index['subject_ids'],'voxel_order':'C_order_of_nonzero_fixed_support','pattern_dtype':'float32_lossless_source_values','extra_preprocessing':[]},indent=2)+'\n')
    audit={'status':'complete','created_at':datetime.now().astimezone().isoformat(),'subjects':len(subject_summaries),'images':493,'brain_trials':processed,'runs':len(provenance),'all_patterns_finite_nonconstant':True,'voxel_count_range':[min(s['voxels'] for s in subject_summaries),max(s['voxels'] for s in subject_summaries)],'mean_ledger_max_absolute_error':mean_error,'new_exclusions':0,'unsmoothed_provenance':'Actual SPM desc-preproc BOLD paths, unchanged SPM hashes from stage audit, and archived/current LSA source match; no added spatial smoothing in source. Not a forensic audit of all upstream BOLD bytes.','input_sha256':input_hash,'source_hash_matches_archive':True,'raw_patterns_no_extra_preprocessing':True,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'nibabel':nib.__version__},'seed':None,'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','subjects','images','brain_trials','runs','voxel_count_range','mean_ledger_max_absolute_error']},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
