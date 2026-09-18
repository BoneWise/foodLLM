"""Pre-inference N token audit plus the actual stored V/T/P token distributions."""
import argparse,json,hashlib,csv,platform
from pathlib import Path
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor
import numpy as np
from control_prompt_spec import render_task_text

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    inp=root/'inputs/controls_v09_v01';out=root/'outputs/controls_v09_token_audit_v01';out.mkdir(exist_ok=False)
    cfg=json.loads((inp/'control_prompt_v01.json').read_text());processor=AutoProcessor.from_pretrained(root/'models/Qwen3.5-9B-c202236',local_files_only=True,min_pixels=cfg['image_min_pixels'],max_pixels=cfg['image_max_pixels'])
    records=[];inputs={str(inp/'control_prompt_v01.json'):sha(inp/'control_prompt_v01.json')}
    all_old=json.loads((root/'outputs/20260916_Qwen_AB_features_v03/records.json').read_text())
    inputs[str(root/'outputs/20260916_Qwen_AB_features_v03/records.json')]=sha(root/'outputs/20260916_Qwen_AB_features_v03/records.json')
    for pr in ['A','B']:
        rows=json.loads((inp/f'{pr}_full.json').read_text())['inputs'];ids={r['image_id'] for r in rows}
        for r in rows:
            prompt=render_task_text(cfg,r);line=r['inserted_line'];lineids=processor.tokenizer.encode(line,add_special_tokens=False)
            rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert sha(r['image_path'])==r['image_sha256']
            with Image.open(r['image_path']) as im:x=processor(text=[rendered],images=[im.convert('RGB')],return_tensors='np')
            records.append(dict(prompt=pr,nutrient='shared_N',condition=r['condition'],image_id=r['image_id'],line_tokens=len(lineids),total_tokens=int(x['input_ids'].shape[1]),readout_token_id=int(x['input_ids'][0,-1]),input_ids_sha256=hashlib.sha256(x['input_ids'].tobytes()).hexdigest(),source='new_pre_inference_processor'))
        for r in all_old:
            if r['prompt_id']==pr and r['condition']=='V' and r['image_id'] in ids:
                records.append(dict(prompt=pr,nutrient='shared_V',condition='V',image_id=r['image_id'],line_tokens=0,total_tokens=r['token_count'],readout_token_id=r['readout_token_id'],input_ids_sha256=r['input_ids_sha256'],source='existing_actual_forward_record'))
        pp=root/f'outputs/multi_nutrient_{pr}_features_v01/records.json';inputs[str(pp)]=sha(pp)
        for r in json.loads(pp.read_text()):
            assert r['image_id'] in ids
            records.append(dict(prompt=pr,nutrient=r['nutrient'],condition=r['condition'],image_id=r['image_id'],line_tokens=len(r['nutrition_token_ids']),total_tokens=r['token_count'],readout_token_id=r['readout_token_id'],input_ids_sha256=r['input_ids_sha256'],source='existing_actual_forward_record'))
        print(pr,'token audit complete',flush=True)
    assert len(records)==22456
    with (out/'token_counts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for pr,n,c in sorted(set((r['prompt'],r['nutrient'],r['condition']) for r in records)):
        rr=[r for r in records if (r['prompt'],r['nutrient'],r['condition'])==(pr,n,c)]
        summary.append(dict(prompt=pr,nutrient=n,condition=c,images=len(rr),**{field+'_min_median_max':np.quantile([r[field] for r in rr],[0,.5,1]).tolist() for field in ['line_tokens','total_tokens']}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'audit.json').write_text(json.dumps(dict(status='passed_before_N_features',created=datetime.now().astimezone().isoformat(),rows=len(records),input_sha256=inputs,script_sha256=sha(__file__),output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},versions=dict(python=platform.python_version(),numpy=np.__version__),scope='N actual processor input; old V/T/P actual forward records; no new model features'),indent=2))

if __name__=='__main__':main()
