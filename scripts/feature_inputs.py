"""Align model and human/ROI matrices by image ID, never by assumed row position."""
import csv
import json
from pathlib import Path

import numpy as np


def load_layer_inputs(matrix_root, analysis_root, prompt_id, layer, sample='primary'):
    if prompt_id not in ('A','B') or layer not in range(1,33):
        raise ValueError('Use a fixed prompt A/B and a scientific layer 1..32')
    if sample not in ('primary','full_VT'):
        raise ValueError('Use primary 493 or full_VT 568 sample')
    matrix_root=Path(matrix_root);analysis_root=Path(analysis_root)
    assert json.loads((matrix_root/'audit.json').read_text())['status']=='complete'
    assert json.loads((analysis_root/'audit.json').read_text())['status']=='complete'
    index=json.loads((analysis_root/'index.json').read_text())
    all_ids=index['image_ids']
    primary=np.load(analysis_root/'primary_mask.npy',allow_pickle=False)
    positions=np.flatnonzero(primary) if sample=='primary' else np.arange(len(all_ids))
    ids=[all_ids[i] for i in positions]
    conditions=['V','T']+([f'P{k}' for k in range(1,6)] if sample=='primary' else [])
    features={};raw_V_probs=None
    for condition in conditions:
        directory=matrix_root/f'{prompt_id}_{condition}'
        with (directory/'image_index.csv').open() as f:
            rows=list(csv.DictReader(f))
        mapping={r['image_id']:int(r['matrix_row']) for r in rows}
        assert len(mapping)==len(rows)
        assert set(ids).issubset(mapping)
        assert set(mapping)==set(all_ids if condition in ('V','T') else [i for i,m in zip(all_ids,primary) if m])
        select=np.array([mapping[i] for i in ids])
        matrix=np.load(directory/'hidden.npy',mmap_mode='r',allow_pickle=False)
        assert matrix.shape==(32,len(rows),4096)
        features[condition]=np.array(matrix[layer-1,select,:],copy=True)
        if condition=='V':
            raw_V_probs=np.load(directory/'rating_probs.npy',allow_pickle=False)[select]
    subject_image={name:np.load(analysis_root/(name+'.npy'),allow_pickle=False)[:,positions]
                   for name in ['human_ratings','roi_beta','session_ids','run_ids']}
    return {'features':features,'image_ids':ids,'subject_ids':index['subject_ids'],
        'food_groups':np.asarray(index['food_groups'])[positions],
        'outer_folds':np.load(analysis_root/'outer_folds.npy',allow_pickle=False)[positions],
        'content_covariates':np.load(analysis_root/'content_covariates.npy',allow_pickle=False)[positions],
        'raw_V_probs':raw_V_probs,'prompt_id':prompt_id,'layer':layer,'sample':sample,
        **subject_image}
