"""Validate every attribute-aware cache and assemble layer-major feature matrices."""
import argparse,csv,json,hashlib,shutil,platform
from collections import defaultdict
from pathlib import Path
import numpy as np
from control_prompt_spec import render_task_text

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(r):return (r['prompt_id'],r['nutrient'],r['condition'],r['image_id'])

def main():
    ap=argparse.ArgumentParser()
    for name in ['features','manifest','config','output']:ap.add_argument('--'+name,required=True)
    a=ap.parse_args();src=Path(a.features);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((src/'audit.json').read_text());cfg=json.loads(Path(a.config).read_text());expected=json.loads(Path(a.manifest).read_text())['inputs']
    assert audit['status']=='complete' and audit['input_manifest_sha256']==sha(a.manifest) and audit['prompt_config_sha256']==sha(a.config)
    records=json.loads((src/'records.json').read_text());assert len(records)==len(expected)==audit['inputs']==1203
    assert len(audit['pilot_reference_exact_matches'])==15
    ex={identity(r):r for r in expected};got={identity(r):r for r in records};assert len(ex)==len(got)==len(records) and ex.keys()==got.keys()
    groups=defaultdict(list)
    for key,r in got.items():
        assert all(r[k]==v for k,v in ex[key].items())
        assert r['model_revision']==cfg['model_revision'] and not r['enable_thinking'] and r['readout_index']==r['token_count']-1
        assert r['candidate_token_ids']==[16,17,18,19] and r['norm_error']==0 and r['lm_head_error']==0
        rendered=(src/(Path(r['cache_path']).stem+'_rendered.txt')).read_text()
        assert hashlib.sha256(rendered.encode()).hexdigest()==r['prompt_sha256']
        assert render_task_text(cfg,r) in rendered
        groups[key[:3]].append(r)
    assert len(groups)==3
    entries=[];common=None;repeated_inputs={};repeat_checks=0
    for (prompt,nutrient,condition),rr in sorted(groups.items()):
        rr.sort(key=lambda r:int(r['image_id']));ids=[r['image_id'] for r in rr]
        if common is None:common=ids
        assert ids==common and len(ids)==401
        dest=out/f'{prompt}_{nutrient}_{condition}';dest.mkdir()
        matrix=np.lib.format.open_memmap(dest/'hidden.npy',mode='w+',dtype=np.float32,shape=(32,len(rr),4096))
        probs=np.empty((len(rr),4),dtype=np.float32);meta=[]
        for j,r in enumerate(rr):
            p=src/r['cache_path'];assert sha(p)==r['cache_sha256']
            with np.load(p,allow_pickle=False) as z:
                assert z['hidden'].shape==(32,4096) and z['hidden'].dtype==np.float32 and np.isfinite(z['hidden']).all()
                assert z['input_ids'].shape==(1,r['token_count']) and int(z['input_ids'][0,-1])==r['readout_token_id']
                assert hashlib.sha256(z['input_ids'].tobytes()).hexdigest()==r['input_ids_sha256']
                assert z['final_norm'].shape==(4096,) and np.isfinite(z['final_norm']).all()
                assert np.array_equal(z['rating_probs'],np.array(r['rating_probs'],dtype=np.float32))
                assert np.array_equal(z['candidate_logprobs'],np.array(r['candidate_logprobs'],dtype=np.float32))
                assert np.isfinite(z['rating_probs']).all() and abs(z['rating_probs'].sum()-1)<1e-6
                assert abs(float(z['rating_probs']@np.arange(1,5))-r['expected_rating'])<1e-6
                signature=(r['prompt_id'],r['nutrient'],r['image_id'],r['inserted_line'])
                payload=(r['input_ids_sha256'],hashlib.sha256(z['hidden'].tobytes()).hexdigest(),hashlib.sha256(z['rating_probs'].tobytes()).hexdigest())
                if signature in repeated_inputs:
                    assert repeated_inputs[signature]==payload,'Identical model input has different features or probabilities'
                    repeat_checks+=1
                else:repeated_inputs[signature]=payload
                matrix[:,j]=z['hidden'];probs[j]=z['rating_probs'];assert np.array_equal(matrix[:,j],z['hidden'])
            meta.append({'matrix_row':j,**{k:r[k] for k in ['image_id','prompt_id','nutrient','condition','inserted_line','text_id','assignment_seed','cache_path','cache_sha256']}})
        matrix.flush();del matrix;np.save(dest/'rating_probs.npy',probs)
        with (dest/'image_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
        reopened=np.load(dest/'hidden.npy',mmap_mode='r')
        for j in [0,len(rr)//2,len(rr)-1]:
            with np.load(src/rr[j]['cache_path']) as z:assert np.array_equal(reopened[:,j],z['hidden'])
        del reopened
        entries.append({'prompt':prompt,'nutrient':nutrient,'condition':condition,'images':401,'files':{str(p.relative_to(out)):sha(p) for p in dest.iterdir()}})
        print(json.dumps({'verified_built':dest.name,'images':401}),flush=True)
    shutil.copy2(__file__,out/'script_snapshot.py')
    result={'status':'complete_verified','verified_inputs':len(records),'common_image_ids':common,'groups':entries,
      'scientific_layers':32,'matrix_axes':['layer','image','hidden_feature'],'cache_copy':'float32_lossless',
      'source_audit_sha256':sha(src/'audit.json'),'manifest_sha256':sha(a.manifest),'config_sha256':sha(a.config),
      'script_sha256':sha(__file__),'validation':'all cache hashes, identities, raw candidate probabilities, model revision, token index; exact persistence checks',
      'human_brain_targets_loaded':False,'versions':{'python':platform.python_version(),'numpy':np.__version__}}
    result['identical_input_payload_exact_checks']=repeat_checks
    (out/'audit.json').write_text(json.dumps(result,indent=2));print('All attribute feature matrices verified',flush=True)

if __name__=='__main__':main()
