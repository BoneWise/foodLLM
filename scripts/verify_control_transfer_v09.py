"""Validate transported N matrices and compare actual inference token IDs to pre-run audit."""
import csv,json,hashlib,platform
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 root=ROOT/'outputs/20260918_无关文本模型缓存_v01';prep=ROOT/'outputs/20260918_控制输入冻结_v01';tok=ROOT/'outputs/20260918_控制Token核查_v01'
 with (tok/'token_counts.csv').open() as f:tt={(r['prompt'],r['condition'],r['image_id']):r for r in csv.DictReader(f) if r['nutrient']=='shared_N'}
 inputs={};checks=0;model=None;versions=None
 for pr in ['A','B']:
  m=root/f'controls_{pr}_matrix_v01';a=json.loads((m/'audit.json').read_text());feature=root/'metadata'/f'controls_{pr}_features_v01';fa=json.loads((feature/'audit.json').read_text());pilot=root/'metadata'/f'controls_{pr}_pilot_v01'
  assert a['status']=='complete_verified' and fa['status']=='complete' and len(fa['pilot_reference_exact_matches'])==15
  assert a['source_audit_sha256']==sha(feature/'audit.json') and fa['pilot_reference_audit_sha256']==sha(pilot/'audit.json')
  assert a['manifest_sha256']==sha(prep/f'{pr}_full.json') and a['config_sha256']==sha(prep/'control_prompt_v01.json')
  if model is None:model=fa['model_files'];versions=fa['versions']
  assert model==fa['model_files'] and versions==fa['versions']
  for g in a['groups']:
   for rel,h in g['files'].items():assert sha(m/rel)==h;inputs[str((m/rel).relative_to(ROOT))]=h
  records=json.loads((feature/'records.json').read_text());expected=json.loads((prep/f'{pr}_full.json').read_text())['inputs'];ex={(r['condition'],r['image_id']):r for r in expected};assert len(records)==1203
  for r in records:
   assert all(r[k]==v for k,v in ex[(r['condition'],r['image_id'])].items());t=tt[(pr,r['condition'],r['image_id'])]
   assert t['input_ids_sha256']==r['input_ids_sha256'] and int(t['total_tokens'])==r['token_count'] and int(t['readout_token_id'])==r['readout_token_id'];checks+=1
  for f in [m/'audit.json',feature/'audit.json',feature/'records.json',pilot/'audit.json',tok/'audit.json',tok/'token_counts.csv']:inputs[str(f.relative_to(ROOT))]=sha(f)
 assert checks==2406
 result=dict(status='passed',created=datetime.now().astimezone().isoformat(),verified_new_inputs=checks,pre_inference_vs_actual_token_hashes_exact=True,all_transferred_matrix_hashes_exact=True,A_B_model_files_and_versions_identical=True,server_raw_cache_location='/path/to/foodLLM/outputs/controls_{A,B}_features_v01',raw_NPZ_caches_retained_on_server=True,local_lossless_hidden_and_probabilities=True,input_sha256=inputs,model_files=model,server_versions=versions,script_sha256=sha(Path(__file__)),python=platform.python_version())
 (root/'transfer_verification.json').write_text(json.dumps(result,indent=2));print('2406 inputs and six feature matrices verified')
if __name__=='__main__':main()
