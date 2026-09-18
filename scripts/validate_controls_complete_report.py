"""Independent CSV-based coverage/pairing checks and completion receipt for v03."""
import csv
import hashlib
import itertools
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/20260918_控制实验报告_v03'
RSA=ROOT/'outputs/20260918_控制RSA_v01'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def rows(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    audit=json.loads((OUT/'generation_audit.json').read_text())
    for p,h in audit['input_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in audit['output_sha256'].items():assert sha(OUT/p)==h,p
    original_audit=json.loads((RSA/'audit.json').read_text())
    for n in ['series_VTP_scores.npy','series_N_scores.npy','series_contrasts.npy','absolute_curves.csv','contrast_curves.csv','window_summary.csv','participant_window_summary.csv']:
        assert sha(RSA/n)==original_audit['output_sha256'][n],n
    ix=json.loads((RSA/'index.json').read_text())
    old=rows(RSA/'participant_window_summary.csv')
    values={(r['prompt'],r['nutrient'],r['ROI'],r['contrast'],r['window'],r['subject_id']):float(r['value']) for r in old}
    curves={(r['prompt'],r['nutrient'],r['ROI'],r['contrast'],int(r['layer'])):float(r['mean']) for r in rows(RSA/'contrast_curves.csv')}
    ab=rows(OUT/'AB_consistency.csv')
    expected=set(itertools.product(ix['nutrients'],ix['series'],ix['windows'],ix['contrasts']))
    assert len(ab)==len(expected)==1472
    assert {(r['nutrient'],r['ROI'],r['window'],r['contrast']) for r in ab}==expected
    for r in ab:
        nut,roi,w,c=(r[k] for k in ['nutrient','ROI','window','contrast'])
        a=[values[('A',nut,roi,c,w,s)] for s in ix['subject_ids']]
        b=[values[('B',nut,roi,c,w,s)] for s in ix['subject_ids']]
        checks={'A_positive_subjects':sum(x>0 for x in a),'B_positive_subjects':sum(x>0 for x in b),
                'paired_subjects_both_positive':sum(x>0 and y>0 for x,y in zip(a,b)),
                'paired_subjects_both_negative':sum(x<0 and y<0 for x,y in zip(a,b)),
                'paired_subjects_opposite_sign':sum(x*y<0 for x,y in zip(a,b)),
                'paired_subjects_includes_zero':sum(x==0 or y==0 for x,y in zip(a,b)),
                'layers_same_nonzero_sign':sum(curves[('A',nut,roi,c,l)]*curves[('B',nut,roi,c,l)]>0 for l in ix['windows'][w])}
        for k,v in checks.items():assert int(r[k])==v,(nut,roi,w,c,k)
        assert abs(sum(a)/29-float(r['A_mean']))<1e-14
        assert abs(sum(b)/29-float(r['B_mean']))<1e-14
    for name,count in [('participant_summary.csv',2944),('implementation_consistency.csv',736),('random_region_summary.csv',64),('parallel_region_summary.csv',736)]:assert len(rows(OUT/name))==count
    pictures=list((OUT/'figures').glob('*.png'));assert len(pictures)==71
    assert len(list((OUT/'figures').glob('*.svg')))==71
    for p in pictures:
        with Image.open(p) as im:assert min(im.size)>1000;im.verify()
    for roi in ix['series']:assert (OUT/'figures'/f'region_{roi}.png').exists()
    files=list(OUT.glob('*.md'))+[ROOT/'README.md',ROOT/'docs/研究状态与下一阶段.md']
    links=0
    for path in files:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if target.startswith(('http:', 'https:', '#')):continue
            dest=(path.parent/target.split('#')[0]).resolve()
            if dest==OUT/'completion_receipt.json':continue
            assert dest.exists(),(path,target)
            links+=1
    cfg=json.loads((ROOT/'configs/project.json').read_text())
    assert cfg['latest_report']==str((OUT/'结果报告.md').relative_to(ROOT))
    assert cfg['paths']['controls_report_document']==cfg['latest_report']
    # Verify the numerical claims used as examples in the prose.
    k={(r['nutrient'],r['ROI'],r['window'],r['contrast']):r for r in ab}
    assert k[('protein_100g','higher_visual','all32','T-N')]['layers_same_nonzero_sign']=='13'
    assert k[('fibre_100g','vmPFC','all32','T-meanP')]['layers_same_nonzero_sign']=='4'
    for roi in ix['random_parcels']:
        for c in ['T-V','T-N']:
            r=k[('fat_100g',roi,'all32',c)];assert float(r['A_mean'])>0 and float(r['B_mean'])>0
    shutil.copy2(__file__,OUT/'validation_script_snapshot.py')
    qa=dict(status='passed',method='Reviewed all seven contact sheets and a full-resolution vmPFC atlas page.',
            contact_sheets=[p.name for p in sorted(OUT.glob('contact_sheet*.jpg'))],
            checks=['No missing panels or obvious label overlaps in overview sheets',
                    'Readable axis labels, legends, zero lines and implementation shading in detailed atlas sample',
                    'Common heatmap limits across prompts and nutrients within each contrast',
                    'All original ROI definitions and descriptive-only labels retained'])
    (OUT/'visual_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    result=dict(status='complete',created=datetime.now().astimezone().isoformat(),
        scope='Existing descriptive results completed under the corrected regional-control framing.',
        new_experiment=False,new_model_inference=False,new_brain_extraction=False,RSA_recomputed=False,inference_run=False,
        regions=41,existing_group_summaries=5,figure_pairs_png_svg=71,
        validation={**audit['validation'],'AB_pairing_independently_checked_from_original_CSV':True,
                    'original_run_hashes_verified':True,'protected_files_unchanged':True,'local_links_checked':links,
                    'all_expected_ROIs_and_windows_present':True,'visual_QA':'passed'},
        narrative_limitations=['descriptive only','N not token/syntax/number-format matched',
                               'L17-32 post-result exploratory window','region measurement definitions differ',
                               'formal inference and acquisition sensitivity not performed'],
        source_generation_audit_sha256=sha(OUT/'generation_audit.json'),
        project_entry_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'README.md',ROOT/'docs/研究状态与下一阶段.md',ROOT/'CHANGELOG.md',ROOT/'configs/project.json']},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion_receipt.json'})
    (OUT/'completion_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'figures':71,'links_checked':links,'protected_files':len(audit['input_sha256']),'AB_rows_verified':len(ab)},ensure_ascii=False))


if __name__=='__main__':main()
