"""Download only the approved pinned Qwen checkpoint into a supplied model directory."""
import argparse
import json
import time
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'

if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);args=ap.parse_args()
    start=time.time()
    path=snapshot_download('Qwen/Qwen3.5-9B',revision=REVISION,local_dir=args.model_dir,max_workers=4,
                           allow_patterns=['*.json','*.jinja','*.safetensors','*.txt','README.md'])
    files=[{'name':str(p.relative_to(path)),'bytes':p.stat().st_size} for p in sorted(Path(path).rglob('*')) if p.is_file() and '.cache' not in p.parts]
    record={'repo':'Qwen/Qwen3.5-9B','revision':REVISION,'seconds':time.time()-start,'files':files,'bytes':sum(x['bytes'] for x in files)}
    (Path(path)/'foodllm_download_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'complete','revision':REVISION,'bytes':record['bytes'],'seconds':record['seconds']}),flush=True)
