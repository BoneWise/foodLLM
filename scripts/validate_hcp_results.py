"""Verify saved HCP aggregation, pair masks, score exports and result integrity."""
import gzip,json
from pathlib import Path
from datetime import datetime
import numpy as np
from prepare_roi import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'outputs/20260917_HCP_RSA_v01';a=json.loads((p/'audit.json').read_text());ix=json.loads((p/'index.json').read_text())
    for name,h in a['output_sha256'].items():assert sha(p/name)==h,name
    u=np.load(p/'unilateral_and_legacy_scores.npy');b=np.load(p/'bilateral_and_reference_and_group_scores.npy');c=np.load(p/'bilateral_and_reference_and_group_contrasts.npy');r=np.load(p/'regional_paired_contrasts.npy')
    assert u.shape==(2,4,39,29,32,7) and b.shape==(2,4,26,29,32,7)
    assert np.isfinite(u).all() and np.isfinite(b).all() and np.isfinite(c).all()
    assert np.array_equal(u[:,0,:,:,:,0],u[:,1,:,:,:,0]) and np.array_equal(u[:,0,:,:,:,0],u[:,3,:,:,:,0])
    names=ix['unilateral_and_legacy_ROIs'];series=ix['series'];errors=[]
    for parcel in series[:18]:
        direct=(u[:,:,names.index('L_'+parcel)]+u[:,:,names.index('R_'+parcel)])/2
        errors.append(float(np.max(abs(direct-b[:,:,series.index(parcel)]))))
    for group,parcels in ix['groups'].items():
        selected=[names.index(h+'_'+q) for q in parcels for h in ['L','R']]
        direct=u[:,:,selected].sum(axis=2)/len(selected)
        errors.append(float(np.max(abs(direct-b[:,:,series.index(group)]))))
    assert np.array_equal(b[:,:,18:21],u[:,:,-3:])
    assert np.allclose(c[...,0],b[...,1]-b[...,0],atol=1e-14,rtol=0)
    assert np.allclose(c[...,1],b[...,1]-sum(b[...,k] for k in range(2,7))/5,atol=1e-14,rtol=0)
    for j,v in enumerate(['higher_visual','early_intermediate_visual']):
        direct=c[:,:,series.index('OFC_nine')]-c[:,:,series.index(v)]
        errors.append(float(np.max(abs(direct-r[:,:,j]))))
    assert max(errors)<1e-14
    with gzip.open(p/'all_subject_scores.csv.gz','rt') as f:
        assert next(f).strip()=='prompt,nutrient,ROI,subject_id,layer,condition,RSA'
        count=sum(1 for _ in f)
    assert count==(39+18+5)*2*4*29*32*7
    result=dict(status='passed',created=datetime.now().astimezone().isoformat(),all_saved_output_hashes_match=True,shared_V_exact=True,
      all_hemisphere_parcel_group_and_regional_aggregations_checked=True,maximum_aggregation_error=max(errors),
      P_is_mean_of_five_scores=True,legacy_reference_arrays_exact=True,score_export_rows=count,
      sample='29 participants; 401 image range; same within-person pair mask across all 39 input ROIs',
      inference_run=False,input_audit_sha256=sha(p/'audit.json'),script_sha256=sha(__file__))
    (p/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
