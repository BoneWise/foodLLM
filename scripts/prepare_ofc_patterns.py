"""Build Suzuki-label anatomical masks and extract unchanged single-trial beta patterns."""
import argparse,json,shutil,subprocess,platform
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
from prepare_roi import sha,write_csv
from extract_rsa_patterns import rows

ROOT=Path(__file__).resolve().parents[1]
LABELS={'lOFC':[2111,2112,2211,2212,2321,2322],'mOFC':[2611,2612]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--atlas-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    src=ROOT/a.atlas_root;out=(ROOT/a.output).resolve();assert out.is_relative_to(ROOT/'outputs') and not out.exists()
    qa=json.loads((src/'visual_QA.json').read_text());assert qa['status']=='passed'
    out.mkdir(parents=True);shutil.copy2(__file__,out/'script_snapshot.py')
    path=src/'registration/AAL_original_in_MNI152NLin2009cAsym.nii.gz';atlas=nib.load(path);labels=atlas.get_fdata()
    source_labels={line.split()[1]:int(line.split()[2]) for line in (src/'source/ROI_MNI_V4.txt').read_text().splitlines() if line.strip()}
    assert set(LABELS['lOFC'])=={source_labels[f'Frontal_{part}_Orb_{side}'] for part in ['Sup','Mid','Inf'] for side in ['L','R']}
    assert LABELS['mOFC']==[source_labels['Frontal_Med_Orb_L'],source_labels['Frontal_Med_Orb_R']]
    old=ROOT/'outputs/20260917_RSA原始模式_v01';old_a=json.loads((old/'audit.json').read_text());idx=json.loads((old/'index.json').read_text())
    mp=ROOT/'data/external/behavior_to_singletrial_beta_29subjects.csv';mapping=rows(mp)
    assert sha(mp)==old_a['input_sha256'][str(mp)]
    ref=nib.load(mapping[0]['beta_file_local']);assert ref.shape==atlas.shape and np.allclose(ref.affine,atlas.affine,rtol=0,atol=1e-5)
    prior=ROOT/'outputs/20260916_vmPFC准备_v01/audit.json';prior_a=json.loads(prior.read_text())
    inp={str(p):sha(p) for p in [path,src/'visual_QA.json',src/'source/ROI_MNI_V4.txt',old/'audit.json',old/'index.json',mp,prior,Path(__file__)]}
    anatomical={k:np.isin(labels,v) for k,v in LABELS.items()};assert not np.any(anatomical['lOFC']&anatomical['mOFC'])
    old_sphere=nib.load(ROOT/'outputs/20260916_vmPFC准备_v01/vmPFC_sphere.nii.gz').get_fdata()>.5
    coverage=[];overlap=[];runqc=[]
    for roi,mask in anatomical.items():
        rd=out/roi;rd.mkdir();nib.save(nib.Nifti1Image(mask.astype(np.uint8),ref.affine),rd/'anatomical_mask.nii.gz')
        overlap.append({'ROI':roi,'anatomical_voxels':int(mask.sum()),'overlap_old_vmPFC_voxels':int((mask&old_sphere).sum()),'fraction_old_vmPFC':float((mask&old_sphere).sum()/old_sphere.sum())})
    for sub in idx['subject_ids']:
        runrows=[r for r in prior_a['provenance']['run_input_hashes'] if '/'+sub+'/' in r['mask_path']]
        expected_runs={r['spm_mat_local'] for r in mapping if r['subject_id']==sub};assert {r['spm_path'] for r in runrows}==expected_runs
        support=np.ones(ref.shape,dtype=bool)
        for r in runrows:
            assert sha(r['mask_path'])==r['mask_sha256'] and sha(r['spm_path'])==r['spm_sha256']
            im=nib.load(r['mask_path']);assert im.shape==ref.shape and np.allclose(im.affine,ref.affine,rtol=0,atol=1e-5)
            valid=np.asarray(im.dataobj)>0;support&=valid
            for roi,mask in anatomical.items():runqc.append({'subject_id':sub,'mask_path':r['mask_path'],'ROI':roi,'covered_voxels':int((mask&valid).sum()),'coverage':float((mask&valid).sum()/mask.sum())})
            inp[r['mask_path']]=r['mask_sha256'];inp[r['spm_path']]=r['spm_sha256']
        tt=rows(old/sub/'trials.csv');assert sha(old/sub/'trials.csv')==old_a['output_sha256'][str(Path(sub)/'trials.csv')]
        inp[str(old/sub/'trials.csv')]=sha(old/sub/'trials.csv')
        supports={roi:mask&support for roi,mask in anatomical.items()}
        old_coords=np.load(old/sub/'voxel_ijk.npy');old_support=np.zeros(ref.shape,dtype=bool);old_support[tuple(old_coords.T)]=True
        old_patterns=np.load(old/sub/'patterns.npy',mmap_mode='r')
        union=np.logical_or.reduce([*supports.values(),old_support])
        coords=np.argwhere(union);lo=coords.min(0);hi=coords.max(0)+1;sl=tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))
        arrays={roi:np.empty((len(tt),int(mask.sum())),dtype=np.float32) for roi,mask in supports.items()}
        for roi,mask in supports.items():assert mask.sum()>1
        beta_payload=__import__('hashlib').sha256()
        for j,r in enumerate(tt):
            im=nib.load(r['beta_file']);assert im.shape==ref.shape and np.allclose(im.affine,ref.affine,rtol=0,atol=1e-5)
            block=np.asarray(im.dataobj[sl]);assert np.array_equal(block,block.astype(np.float32),equal_nan=True)
            assert np.array_equal(block[old_support[sl]],old_patterns[j]),'Existing vmPFC payload changed'
            for roi,mask in supports.items():
                v=block[mask[sl]];assert np.isfinite(v).all() and np.var(v.astype(float))>0,(sub,roi,j)
                arrays[roi][j]=v;beta_payload.update(v.astype(np.float32).tobytes())
        for roi,mask in supports.items():
            sd=out/roi/sub;sd.mkdir();arr=arrays[roi];xyz=np.argwhere(mask)
            np.save(sd/'patterns.npy',arr);np.save(sd/'voxel_ijk.npy',xyz);np.save(sd/'affine.npy',ref.affine)
            nib.save(nib.Nifti1Image(mask.astype(np.uint8),ref.affine),sd/'fixed_support.nii.gz');write_csv(sd/'trials.csv',tt)
            left=np.isin(labels,[v for v in LABELS[roi] if v%2==1]);right=np.isin(labels,[v for v in LABELS[roi] if v%2==0])
            coverage.append({'subject_id':sub,'ROI':roi,'images':len(tt),'runs':len(runrows),
              'anatomical_voxels':int(anatomical[roi].sum()),'fixed_voxels':int(mask.sum()),'coverage':float(mask.sum()/anatomical[roi].sum()),
              'left_voxels':int((mask&left).sum()),'right_voxels':int((mask&right).sum()),
              'left_coverage':float((mask&left).sum()/left.sum()),'right_coverage':float((mask&right).sum()/right.sum()),
              'min_pattern_variance':float(np.var(arr.astype(float),axis=1).min()),'all_finite':bool(np.isfinite(arr).all()),
              'raw_payload_sha256_both_ROIs':beta_payload.hexdigest()})
        print(f'{sub}: '+', '.join(f'{r}={int(m.sum())} voxels' for r,m in supports.items()),flush=True)
    write_csv(out/'coverage.csv',coverage);write_csv(out/'run_coverage.csv',runqc);write_csv(out/'ROI_overlap.csv',overlap)
    for roi in LABELS:(out/roi/'index.json').write_text(json.dumps(idx,indent=2))
    audit={'status':'complete','date':datetime.now().astimezone().isoformat(),'subject_count':len(idx['subject_ids']),
      'image_count':len(idx['image_ids']),'ROIs':LABELS,'source_space':'original_AAL_SPM_single_subject',
      'target_space':'MNI152NLin2009cAsym','fixed_support':'ROI_intersect_all_existing_subject_run_masks',
      'raw_beta_patterns':True,'extra_preprocessing':[],'new_exclusions':[],'source_unsmoothed_audit':str(old/'audit.json'),
      'input_sha256':inp,'seed':None,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'nibabel':nib.__version__},
      'output_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('OFC raw patterns complete',flush=True)

if __name__=='__main__':main()
