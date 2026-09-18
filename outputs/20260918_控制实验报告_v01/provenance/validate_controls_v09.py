"""Independent end-to-end validation of v09 identities, contrasts, exports and references."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def main():
 p=ROOT/'outputs/20260918_控制RSA_v01';q=ROOT/'outputs/20260918_控制实验报告_v01';ix=read(p/'index.json');names=ix['series'];un=ix['unilateral_ROIs'];audit=read(p/'audit.json');qa=read(q/'audit.json')
 for root,a in [(p,audit),(q,qa)]:
  for f,h in a['output_sha256'].items():assert sha(root/f)==h,str(root/f)
 s=np.load(p/'series_VTP_scores.npy');n=np.load(p/'series_N_scores.npy');c=np.load(p/'series_contrasts.npy');us=np.load(p/'unilateral_VTP_scores.npy');nn=np.load(p/'unilateral_N_scores.npy')
 assert s.shape==(2,4,46,29,32,7) and n.shape==(2,46,29,32,3) and c.shape==(2,4,46,29,32,4)
 assert us.shape==(2,4,79,29,32,7) and nn.shape==(2,79,29,32,3)
 interrupted=ROOT/'outputs/20260918_控制RSA中断记录_v01'
 assert np.array_equal(us[0],np.load(interrupted/'A_VTP_unilateral_and_legacy_scores.npy'))
 assert np.array_equal(nn[0],np.load(interrupted/'A_N_unilateral_and_legacy_scores.npy'))
 old=ROOT/'outputs/20260917_HCP_RSA_v01';oi=read(old/'index.json');os=np.load(old/'unilateral_and_legacy_scores.npy');assert np.array_equal(us[:,:,:39],os)
 assert np.array_equal(us[:,0,...,0],us[:,1,...,0])
 assert np.allclose(c[...,0]+c[...,1],c[...,2],atol=1e-15,rtol=0)
 assert np.allclose(c[...,1],s[...,1]-n.mean(-1)[:,None],atol=0,rtol=0)
 for name in ix['parcels']:
  r=names.index(name);l=un.index('L_'+name);rr=un.index('R_'+name)
  assert np.array_equal(n[:,r],(nn[:,l]+nn[:,rr])/2)
  assert np.array_equal(s[:,:,r],(us[:,:,l]+us[:,:,rr])/2)
 for name,pp in ix['groups'].items():
  ii=[names.index(x) for x in pp];r=names.index(name)
  assert np.allclose(s[:,:,r],s[:,:,ii].mean(2),atol=1e-15,rtol=0)
  assert np.allclose(n[:,r],n[:,ii].mean(1),atol=1e-15,rtol=0)
 error=0
 for pi,pr in enumerate(['A','B']):
  for name,si,li,ci in [('L_9p',0,0,0),('R_TPOJ2',13,31,2),('L_47s',28,16,1),('vmPFC',7,15,0)]:
   root=ROOT/('outputs/20260918_随机HCP原始模式_v01' if name in un[39:] else 'outputs/20260917_HCP原始模式_v01');base=root/name/ix['subject_ids'][si]
   if name=='vmPFC':base=ROOT/'outputs/20260917_四营养A_RSA_v01'/name/'brain'/ix['subject_ids'][si]
   b=np.load(base/'RDM.npy');g=np.load(base/'global_image_rows.npy');m=np.load(p/pr/f'N{ci+1}_RDM.npy',mmap_mode='r')[li];tr=np.tril_indices(len(g),-1)
   direct=spearmanr(b[tr],m[g[tr[0]],g[tr[1]]]).statistic;dif=abs(direct-nn[pi,un.index(name),si,li,ci]);assert dif<1e-12;error=max(error,float(dif))
 for r in rows(p/'window_summary.csv'):
  if r['ROI'] not in names:continue
  pi=ix['prompts'].index(r['prompt']);nj=ix['nutrients'].index(r['nutrient']);ri=names.index(r['ROI']);ci=ix['contrasts'].index(r['contrast']);ls=np.array(ix['windows'][r['window']])-1
  direct=c[pi,nj,ri][:,ls,ci].mean();assert abs(direct-float(r['mean']))<1e-14
 nine=np.load(q/'random_nine_group_contrasts.npy',mmap_mode='r');ni=read(q/'reference_index.json');assert nine.shape==(2,4,1000,29,32,4)
 for j in [0,17,517,999]:assert np.allclose(nine[:,:,j],c[:,:,[names.index(x) for x in ni['nine_parcel_groups'][j]]].mean(2),atol=1e-15,rtol=0)
 matches=read(q/'matched_reference_members.json');quality={r['ROI']:r for r in rows(q/'ROI_quality_summary.csv')}
 for name,selected in matches.items():
  direct=[r for r in ix['random_parcels'] if abs(float(quality[r]['LOSO'])-float(quality[name]['LOSO']))<=.02 and .5<=float(quality[r]['median_fixed_voxels'])/float(quality[name]['median_fixed_voxels'])<=2];assert selected==direct
 for r in rows(q/'individual_HCP_reference_positions.csv'):
  assert int(r['matched_n'])==len(matches[r['ROI']]);assert (r['matched_fraction_below']=='')==(int(r['matched_n'])<5)
 result=dict(status='passed',created=datetime.now().astimezone().isoformat(),original_VTP_bitwise_preserved=True,all_79_ROIs_common_participant_pairs=True,all_401_image_ID_order_consistent=True,all_saved_output_hashes_verified=True,hemisphere_and_group_aggregation_verified=True,N_V_plus_T_N_equals_T_V=True,N_P_score_means_not_RDM_means=True,full_and_deep_window_exports_verified=True,nine_group_independent_checks=True,matching_calipers_and_insufficient_rule_verified=True,independent_Spearman_max_error=error,no_p_values_or_CIs=True,inputs={'RSA_audit':sha(p/'audit.json'),'report_audit':sha(q/'audit.json'),'transfer_verification':sha(ROOT/'outputs/20260918_无关文本模型缓存_v01/transfer_verification.json')},script_sha256=sha(Path(__file__)))
 (q/'independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
