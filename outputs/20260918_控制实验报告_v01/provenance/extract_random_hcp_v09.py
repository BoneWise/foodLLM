"""Extract unchanged valuation betas for 40 HCP parcels on the common 401 images."""
import json, shutil, platform, subprocess, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
import scipy
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits
from prepare_roi import sha, write_csv
from extract_rsa_patterns import rows
from rsa_core import correlation_rdm, validate

ROOT=Path(__file__).resolve().parents[1]

def main():
    cfgp=ROOT/'configs/controls_v09_execution_v01.json';cfg=json.loads(cfgp.read_text())
    atlas=ROOT/cfg['atlas_preparation'];aa=json.loads((atlas/'audit.json').read_text())
    qa=json.loads((atlas/'visual_QA.json').read_text());assert qa['status']=='passed'
    out=ROOT/cfg['patterns_output'];assert not out.exists();out.mkdir(parents=True)
    shutil.copy2(__file__,out/'script_snapshot.py')
    old=ROOT/'outputs/20260917_RSA原始模式_v01';oa=json.loads((old/'audit.json').read_text())
    cache=ROOT/'outputs/20260917_四营养A_RSA_v01';idx=json.loads((cache/'index.json').read_text())
    table=rows(atlas/'labels.csv');names=[r['ROI'] for r in table]
    ims=idx['image_ids'];image_index={x:i for i,x in enumerate(ims)}
    outidx=dict(image_ids=ims,subject_ids=idx['subject_ids'],ROIs=names,parcel_table=table,pattern_dtype='float32_lossless',extra_preprocessing=[])
    (out/'index.json').write_text(json.dumps(outidx,indent=2))
    inp={str(p):sha(p) for p in [cfgp,atlas/'audit.json',atlas/'visual_QA.json',atlas/'labels.csv',cache/'index.json',old/'audit.json',Path(__file__),ROOT/'scripts/rsa_core.py']}
    # Revalidate masks/SPM so prior beta-event correspondence remains applicable.
    for p,h in aa['input_sha256'].items():
        assert sha(p)==h;inp[p]=h
    quality=[];verification=validate();errors=[]
    for sub in idx['subject_ids']:
        tp=old/sub/'trials.csv';assert sha(tp)==oa['output_sha256'][str(Path(sub)/'trials.csv')]
        tt=[r for r in rows(tp) if r['image_id'] in image_index]
        assert len({r['image_id'] for r in tt})==len(tt)
        gi=np.array([image_index[r['image_id']] for r in tt]);assert np.array_equal(gi,np.sort(gi))
        assert np.array_equal(gi,np.load(cache/'vmPFC/brain'/sub/'global_image_rows.npy'))
        inp[str(tp)]=sha(tp)
        pp=old/sub/'patterns.npy';cp=old/sub/'voxel_ijk.npy'
        for p in [pp,cp]:assert sha(p)==oa['output_sha256'][str(p.relative_to(old))];inp[str(p)]=sha(p)
        oldpatterns=np.load(pp,mmap_mode='r');oldcoords=np.load(cp)
        ref=nib.load(tt[0]['beta_file']);shape=ref.shape
        oldflat=np.ravel_multi_index(tuple(oldcoords.T),shape)
        selectors={};supports={}
        for name in names:
            p=atlas/'fixed_supports'/sub/(name+'.nii.gz')
            assert sha(p)==aa['output_sha256'][str(p.relative_to(atlas))];inp[str(p)]=sha(p)
            im=nib.load(p);assert im.shape==shape and np.allclose(im.affine,ref.affine,rtol=0,atol=1e-5)
            mask=np.asarray(im.dataobj)>0;sel=np.flatnonzero(mask.ravel())
            assert len(sel)>=cfg['technical_minimum_voxels'],(sub,name,len(sel))
            selectors[name]=sel;supports[name]=mask
        arrays={name:np.empty((len(tt),len(sel)),np.float32) for name,sel in selectors.items()}
        payload=hashlib.sha256()
        for j,r in enumerate(tt):
            im=nib.load(r['beta_file']);assert im.shape==shape and np.allclose(im.affine,ref.affine,rtol=0,atol=1e-5)
            flat=np.asanyarray(im.dataobj).ravel()
            assert np.array_equal(flat[oldflat],oldpatterns[int(r['pattern_row'])]),'Prior vmPFC payload differs'
            for name,sel in selectors.items():
                v=flat[sel];assert np.isfinite(v).all() and np.var(v.astype(float))>0,(sub,name,r['image_id'])
                assert np.array_equal(v,v.astype(np.float32));arrays[name][j]=v
                payload.update(v.astype(np.float32).tobytes())
            if (j+1)%100==0:print(f'{sub}: {j+1}/{len(tt)} betas',flush=True)
        st=out/'trials'/sub;st.mkdir(parents=True)
        mapped=[dict(r,source_pattern_row=r['pattern_row'],pattern_row=j,global_image_row=int(gi[j])) for j,r in enumerate(tt)]
        write_csv(st/'trials.csv',mapped)
        session=np.array([r['session_id'] for r in tt]);run=np.array([r['run_identity'] for r in tt])
        np.save(st/'session_RDM.npy',(session[:,None]!=session[None,:]).astype(np.uint8))
        np.save(st/'run_RDM.npy',(run[:,None]!=run[None,:]).astype(np.uint8))
        for name in names:
            sd=out/name/sub;sd.mkdir(parents=True)
            arr=arrays[name];rdm=correlation_rdm(arr)
            e=float(np.max(np.abs(rdm-squareform(pdist(arr.astype(float),'correlation')))));assert e<1e-12
            errors.append(e)
            np.save(sd/'patterns.npy',arr);np.save(sd/'RDM.npy',rdm);np.save(sd/'global_image_rows.npy',gi)
            np.save(sd/'voxel_ijk.npy',np.argwhere(supports[name]));np.save(sd/'affine.npy',ref.affine)
            quality.append(dict(subject_id=sub,ROI=name,images=len(tt),pairs=len(tt)*(len(tt)-1)//2,voxels=arr.shape[1],
              finite=True,min_pattern_variance=float(np.var(arr.astype(float),axis=1).min()),RDM_sd=float(rdm[np.tril_indices(len(tt),-1)].std()),
              payload_sha256_all_40_ROIs=payload.hexdigest(),independent_distance_max_error=e))
        print(sub,'40 parcel patterns and RDMs complete',flush=True)
    write_csv(out/'pattern_QC.csv',quality)
    audit=dict(status='complete',created=datetime.now().astimezone().isoformat(),ROIs=names,participants=29,common_images=401,
      trials_per_ROI=sum(r['images'] for r in quality if r['ROI']==names[0]),new_exclusions=[],raw_trial_beta=True,
      extra_preprocessing=[],beta_phase='valuation',all_prior_vmPFC_payloads_exactly_reproduced=True,
      independent_distance_max_error=max(errors),method_validation=verification,seed=None,
      input_sha256=inp,external_beta_hash_scope='extracted_40_ROI_payload_and_exact_legacy_vmPFC_payload_check_not_full_beta_volumes',
      git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,nibabel=nib.__version__),
      output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('HCP extraction complete',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
