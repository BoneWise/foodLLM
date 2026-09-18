"""Build layer-major, memory-mapped analysis inputs from verified model caches."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['features','verification','output']:
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    source=Path(args.features);out=Path(args.output)
    validation=json.loads(Path(args.verification).read_text())
    assert validation['status']=='passed' and validation['verified_inputs']==7202
    assert validation['provenance']['extraction_audit_sha256']==digest(source/'audit.json')
    out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list)
    for row in json.loads((source/'records.json').read_text()):
        groups[(row['prompt_id'],row['condition'])].append(row)
    entries=[]
    for (prompt,condition),rows in sorted(groups.items()):
        rows.sort(key=lambda r:int(r['image_id']))
        dest=out/f'{prompt}_{condition}';dest.mkdir()
        matrix=np.lib.format.open_memmap(dest/'hidden.npy',mode='w+',dtype=np.float32,
                                       shape=(32,len(rows),4096))
        probs=np.empty((len(rows),4),dtype=np.float32)
        records=[]
        for index,row in enumerate(rows):
            cache=source/row['cache_path']
            assert digest(cache)==row['cache_sha256']
            with np.load(cache,allow_pickle=False) as z:
                matrix[:,index,:]=z['hidden']
                assert np.array_equal(matrix[:,index,:],z['hidden'])
                probs[index]=z['rating_probs']
            records.append({'matrix_row':index,'image_id':row['image_id'],
                'primary_eligible':row['primary_eligible'],'prompt_id':prompt,'condition':condition,
                'true_kcal_100g':row['true_kcal_100g'],
                'presented_kcal_100g':row['presented_kcal_100g'],
                'donor_image_id':row['donor_image_id'],'source_cache':row['cache_path'],
                'source_cache_sha256':row['cache_sha256']})
        matrix.flush();del matrix
        np.save(dest/'rating_probs.npy',probs,allow_pickle=False)
        with (dest/'image_index.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        # Independently re-open and compare edge/interior slices after persistence.
        reopened=np.load(dest/'hidden.npy',mmap_mode='r',allow_pickle=False)
        for j in sorted({0,len(rows)//2,len(rows)-1}):
            with np.load(source/rows[j]['cache_path'],allow_pickle=False) as z:
                assert np.array_equal(reopened[:,j,:],z['hidden'])
        del reopened
        entries.append({'prompt_id':prompt,'condition':condition,'images':len(rows),
            'hidden_shape':[32,len(rows),4096],
            'files':{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':digest(p)}
                     for p in sorted(dest.iterdir())}})
        print(json.dumps({'built':f'{prompt}_{condition}','images':len(rows)}),flush=True)
    assert sum(e['images'] for e in entries)==7202
    report={'status':'complete','matrix_axes':['layer','image','feature'],
        'layer_numbers':list(range(1,33)),'dtype':'float32_exact_copy_of_verified_cache',
        'groups':entries,'total_image_condition_prompt_rows':7202,
        'human_or_brain_targets_loaded':False,'final_norm_not_an_analysis_layer':True,
        'verification_sha256':digest(args.verification),'script_sha256':digest(__file__),
        'source_audit_sha256':digest(source/'audit.json'),'numpy_version':np.__version__}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
