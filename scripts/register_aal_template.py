"""Public-template registration only; contains no participant data or effects."""
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
os.environ['ANTS_RANDOM_SEED']='20260917'
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import ants
import numpy as np

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    fixed=ants.image_read(str(src/'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'))
    mask=ants.image_read(str(src/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'))>.5
    moving=ants.image_read(str(src/'single_subj_T1.nii'))
    # Both supplied T1s contain extracranial tissue. Constrain the similarity
    # metric to the independent target brain mask in every registration stage.
    params=dict(type_of_transform='SyNRA',initial_transform='Identity',random_seed=20260917,
      aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=1.0,
      aff_iterations=(1000,500,250,100),aff_shrink_factors=(6,4,2,1),aff_smoothing_sigmas=(3,2,1,0),
      syn_metric='CC',syn_sampling=4,reg_iterations=(100,70,50,20),grad_step=.15,
      flow_sigma=3,total_sigma=0,singleprecision=False,verbose=True)
    params['mask_all_stages']=True
    result=ants.registration(fixed=fixed,moving=moving,mask=mask,outprefix=str(out/'colin_to_2009_'),**params)
    ants.image_write(result['warpedmovout'],str(out/'warped_source_T1w.nii.gz'))
    atlas=ants.image_read(str(src/'ROI_MNI_V4.nii'))
    warped=ants.apply_transforms(fixed=fixed,moving=atlas,transformlist=result['fwdtransforms'],interpolator='nearestNeighbor')
    ants.image_write(warped,str(out/'AAL_original_in_MNI152NLin2009cAsym.nii.gz'))
    assert set(np.unique(warped.numpy())).issubset(set(np.unique(atlas.numpy())))
    before=ants.resample_image_to_target(moving,fixed,interp_type='linear')
    valid=mask.numpy()>0
    corr=lambda x:float(np.corrcoef(fixed.numpy()[valid],x.numpy()[valid])[0,1])
    audit={'status':'computed_pending_visual_QA','parameters':params,'source_space':'SPM_canonical_single_subject_Colin27',
      'target_space':'MNI152NLin2009cAsym','source_atlas_interpolation':'nearestNeighbor',
      'fwdtransforms':[Path(p).name for p in result['fwdtransforms']],
      'invtransforms':[Path(p).name for p in result['invtransforms']],
      'T1_pearson_before':corr(before),'T1_pearson_after':corr(result['warpedmovout']),
      'ants_version':ants.__version__,'python':platform.python_version(),'threads':1,
      'input_sha256':{p.name:sha(p) for p in src.iterdir() if p.is_file()},
      'script_sha256':sha(__file__),'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'requirements.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    print(json.dumps({k:audit[k] for k in ['status','T1_pearson_before','T1_pearson_after']}),flush=True)

if __name__=='__main__':main()
