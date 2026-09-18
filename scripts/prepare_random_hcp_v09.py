"""Verify author labels, resample within 2009cAsym, and audit input-only coverage."""
import json, platform, shutil, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_roi import sha, write_csv
from extract_rsa_patterns import rows

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT/'outputs/20260918_随机HCP图谱核查_v01'
    assert not (out/'audit.json').exists()
    cfgp = ROOT/'configs/controls_v09_execution_v01.json'
    cfg = json.loads(cfgp.read_text())
    out.mkdir(parents=True,exist_ok=False)
    src = ROOT/'outputs/20260917_HCP图谱准备_v01/source'
    manifest = json.loads((src/'source_manifest.json').read_text())
    for r in manifest['files']: assert sha(ROOT/r['local_path']) == r['sha256']
    ap = src/'HCPex_2mm.nii'
    refp = ROOT/'outputs/20260917_OFC图谱准备_v01/source/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'
    atlas, ref = nib.load(ap), nib.load(refp)
    # Author guide pp. 1,4,10 identifies ICBM152 2009c asymmetric; target is same template.
    existing_grid = ROOT/'outputs/20260917_HCP图谱准备_v01/HCPex_v1.1_on_beta_grid.nii.gz'
    target = nib.load(existing_grid)
    assert target.shape==ref.shape and np.allclose(target.affine,ref.affine,rtol=0,atol=1e-5)
    labels = target.get_fdata()
    assert np.array_equal(labels, labels.astype(np.int16))
    targetp = out/'HCPex_v1.1_on_beta_grid.nii.gz'
    nib.save(nib.Nifti1Image(labels.astype(np.int16), ref.affine), targetp)
    label_rows = [line.split() for line in (src/'HCPex_2mm.txt').read_text().splitlines() if line.strip()]
    lookup = {(r[1], r[2]):int(r[0]) for r in label_rows}
    assert len(lookup)==426 and all(r[0]==r[3] for r in label_rows)
    full_lookup = {int(r[0]):r[1] for line in (src/'HCPex_LookUpTable.txt').read_text().splitlines() if (r:=line.split()) and not r[0].startswith('#')}
    masks={}; table=[]
    for group, parcels in cfg['HCP_groups'].items():
        for parcel in parcels:
            for hemi in ['L','R']:
                value=lookup[(hemi,parcel)];name=hemi+'_'+parcel
                assert full_lookup[value].endswith('_'+hemi)
                mask=labels==value; xyz=nib.affines.apply_affine(ref.affine,np.argwhere(mask))
                assert mask.sum()>2 and (xyz[:,0].mean()<0 if hemi=='L' else xyz[:,0].mean()>0)
                masks[name]=mask
                md=out/'masks';md.mkdir(exist_ok=True)
                nib.save(nib.Nifti1Image(mask.astype(np.uint8),ref.affine),md/(name+'.nii.gz'))
                table.append(dict(ROI=name,parcel=parcel,hemisphere=hemi,group=group,label=value,
                  long_name=full_lookup[value],source_voxels=int((atlas.get_fdata()==value).sum()),
                  target_voxels=int(mask.sum()),center_x=float(xyz[:,0].mean()),center_y=float(xyz[:,1].mean()),center_z=float(xyz[:,2].mean())))
    assert len(masks)==40
    priorp=ROOT/'outputs/20260916_vmPFC准备_v01/audit.json';prior=json.loads(priorp.read_text())
    indexp=ROOT/'outputs/20260917_四营养A_RSA_v01/index.json';idx=json.loads(indexp.read_text())
    inp={str(p):sha(p) for p in [existing_grid,ap,refp,cfgp,priorp,indexp,src/'source_manifest.json',Path(__file__)]}
    coverage=[];runcoverage=[]
    for sub in idx['subject_ids']:
        rr=[r for r in prior['provenance']['run_input_hashes'] if '/'+sub+'/' in r['mask_path']]
        assert rr
        support=np.ones(ref.shape,bool)
        for r in rr:
            for pkey,hkey in [('mask_path','mask_sha256'),('spm_path','spm_sha256')]:
                assert sha(r[pkey])==r[hkey];inp[r[pkey]]=r[hkey]
            im=nib.load(r['mask_path']);assert im.shape==ref.shape and np.allclose(im.affine,ref.affine,rtol=0,atol=1e-5)
            valid=np.asarray(im.dataobj)>0;support &= valid
            for name,mask in masks.items():runcoverage.append(dict(subject_id=sub,ROI=name,run_mask=r['mask_path'],voxels=int((mask&valid).sum()),coverage=float((mask&valid).sum()/mask.sum())))
        sd=out/'fixed_supports'/sub;sd.mkdir(parents=True)
        for name,mask in masks.items():
            valid=mask&support;n=int(valid.sum())
            nib.save(nib.Nifti1Image(valid.astype(np.uint8),ref.affine),sd/(name+'.nii.gz'))
            coverage.append(dict(subject_id=sub,ROI=name,anatomical_voxels=int(mask.sum()),fixed_voxels=n,coverage=n/int(mask.sum()),technical_minimum_3_voxels=n>=3,runs=len(rr)))
        print(sub, 'coverage checked',flush=True)
    overlap=[]
    legacy_paths={'vmPFC':ROOT/'outputs/20260916_vmPFC准备_v01/vmPFC_sphere.nii.gz',
      'lOFC':ROOT/'outputs/20260917_OFC原始模式_v01/lOFC/anatomical_mask.nii.gz',
      'mOFC':ROOT/'outputs/20260917_OFC原始模式_v01/mOFC/anatomical_mask.nii.gz'}
    for legacy,lp in legacy_paths.items():
        oldmask=nib.load(lp).get_fdata()>.5;inp[str(lp)]=sha(lp)
        for name,mask in masks.items():
            overlap.append(dict(ROI=name,legacy_ROI=legacy,overlap_voxels=int((mask&oldmask).sum()),fraction_random_parcel=float((mask&oldmask).sum()/mask.sum())))
    write_csv(out/'legacy_overlap.csv',overlap)
    write_csv(out/'labels.csv',table);write_csv(out/'coverage.csv',coverage);write_csv(out/'run_coverage.csv',runcoverage)
    # One three-plane image per parcel; both hemispheres visible in coronal/axial views.
    bg=ref.get_fdata();figdir=out/'overlay';figdir.mkdir()
    for parcel in [p for g in cfg['HCP_groups'].values() for p in g]:
        union=masks['L_'+parcel]|masks['R_'+parcel]
        center=np.round(np.argwhere(masks['L_'+parcel]).mean(0)).astype(int)
        fig,axs=plt.subplots(1,3,figsize=(10,3.5))
        for axis,ax in enumerate(axs):
            sl=tuple(int(center[axis]) if d==axis else slice(None) for d in range(3))
            ax.imshow(bg[sl].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(bg,99))
            overlay=np.ma.masked_where(~union[sl].T,union[sl].T)
            ax.imshow(overlay,origin='lower',cmap='autumn',vmin=0,vmax=1,alpha=.65)
            world=nib.affines.apply_affine(ref.affine,center)[axis]
            ax.set_title(f'{"xyz"[axis]} = {world:.1f} mm');ax.axis('off')
        fig.suptitle(f'HCPex v1.1 | {parcel} L+R | target 2009cAsym (neurological)')
        fig.tight_layout();fig.savefig(figdir/(parcel+'.png'),dpi=120);plt.close(fig)
    shutil.copy2(__file__,out/'script_snapshot.py')
    audit=dict(status='atlas_and_input_coverage_complete_visual_review_pending',created=datetime.now().astimezone().isoformat(),
      atlas='HCPex_v1.1_author_2mm',source_commit=manifest['commit'],source_space='ICBM152_2009c_asymmetric',target_space='MNI152NLin2009cAsym',
      space_evidence='source/UserGuide_1Jan2023.pdf pages 1,4,10; existing target template and beta metadata',
      resampling='reuse_verified_v08_grid_no_new_resampling',source_shape=list(atlas.shape),target_shape=list(ref.shape),
      source_affine=atlas.affine.tolist(),target_affine=ref.affine.tolist(),parcels=40,participants=29,
      technical_invalid=[r for r in coverage if not r['technical_minimum_3_voxels']],minimum_coverage=min(r['coverage'] for r in coverage),
      seed=None,input_sha256=inp,git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      versions=dict(python=platform.python_version(),numpy=np.__version__,nibabel=nib.__version__),
      output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:audit[k] for k in ['technical_invalid','minimum_coverage']}))

if __name__=='__main__':main()
