"""Freeze input-only unrelated strings and random HCP parcels before new effects."""
import json, platform, shutil, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from prepare_roi import sha, write_csv

ROOT=Path(__file__).resolve().parents[1]
WORDS='anchor bracket cable compass hinge bolt screw nut washer rivet clamp latch lever pulley piston axle gear spring wheel spoke valve nozzle pipe beam column arch pillar brick tile plank panel railing fence gate tower bridge tunnel antenna radar satellite telescope prism lens mirror magnet battery motor switch socket plug circuit wire keyboard cursor printer stapler folder envelope ribbon zipper buckle button needle thread spool'.split()

def main():
    out=ROOT/'outputs/20260918_控制输入冻结_v01';out.mkdir(parents=True,exist_ok=False)
    # These words are object nouns, not an affect-free or semantically empty vocabulary.
    # 'nut' has a food sense: remove it before constructing any input.
    words=sorted(set(WORDS)-{'nut'})
    oldcfgp=ROOT/'configs/hcp_regional_execution_v01.json';oldcfg=json.loads(oldcfgp.read_text())
    ixp=ROOT/'outputs/20260917_HCP_RSA_v01/index.json';ix=json.loads(ixp.read_text());ids=ix['image_ids'];assert len(ids)==401
    labelp=ROOT/'outputs/20260917_HCP图谱准备_v01/source/HCPex_2mm.txt'
    labels=[s.split() for s in labelp.read_text().splitlines() if s.strip()]
    cortical={r[2] for r in labels if r[1]=='L' and int(r[0])<=180}
    excluded=[p for g in oldcfg['HCP_groups'].values() for p in g]
    pool=sorted(cortical-set(excluded));assert len(pool)==162
    selected=np.random.default_rng(2026091802).choice(pool,20,replace=False).tolist()
    rng=np.random.default_rng(2026091801);strings=[]
    while len(strings)<401:
        line='Reference words: '+', '.join(rng.choice(words,4,replace=False).tolist())+'.'
        if line not in strings:strings.append(line)
    assignments=[]
    for k in range(1,4):
        perm=np.random.default_rng(2026091810+k).permutation(401)
        assignments.extend(dict(image_id=i,condition=f'N{k}',text_id=int(j),inserted_line=strings[j],seed=2026091810+k) for i,j in zip(ids,perm))
    write_csv(out/'N_assignments.csv',assignments)
    (out/'N_vocabulary.json').write_text(json.dumps(dict(words=words,word_count_per_line=4,lines=strings,review='common concrete object nouns; direct food/nutrition terms excluded before inference; incidental semantic associations remain possible'),indent=2))
    prompt=json.loads((ROOT/'configs/multi_nutrient_prompt_v01.json').read_text())
    prompt.update(version='unrelated_text_v09_v01',unrelated_template='Reference words: {four_comma_separated_nouns}.',N_conditions=['N1','N2','N3'])
    (out/'control_prompt_v01.json').write_text(json.dumps(prompt,indent=2))
    oldmanp=ROOT/'outputs/20260917_四营养模型输入_v01/A_full.json'
    oldrows=json.loads(oldmanp.read_text())['inputs'];byid={r['image_id']:r for r in oldrows}
    for pr in ['A','B']:
        rr=[dict(prompt_id=pr,nutrient='unrelated_text',condition=a['condition'],image_id=a['image_id'],image_path=byid[a['image_id']]['image_path'],image_sha256=byid[a['image_id']]['image_sha256'],inserted_line=a['inserted_line'],text_id=a['text_id'],assignment_seed=a['seed']) for a in assignments]
        (out/f'{pr}_full.json').write_text(json.dumps(dict(inputs=rr),indent=2))
        pilot=[r for r in rr if r['image_id'] in [ids[i] for i in [0,100,200,300,400]]]
        assert len(pilot)==15
        (out/f'{pr}_pilot.json').write_text(json.dumps(dict(inputs=pilot),indent=2))
    # Same nine-parcel aggregation as OFC, conditional on this fixed 20-parcel panel.
    rng=np.random.default_rng(2026091803);groups=[];seen=set()
    while len(groups)<1000:
        g=tuple(sorted(rng.choice(20,9,replace=False).tolist()))
        if g not in seen:seen.add(g);groups.append([selected[i] for i in g])
    (out/'random_parcels.json').write_text(json.dumps(dict(candidate_pool=pool,excluded=excluded,selected=selected,seed=2026091802,draw='20 without replacement, no effect-based replacements',nine_parcel_reference_groups=groups,group_seed=2026091803,groups_are_dependent=True),indent=2))
    cfg=dict(version='v09_execution_v01',date='2026-09-18',status='frozen_before_new_control_effects',user_authorization='User approved recommended minimal controls and autonomous execution through report on 2026-09-18',
      input_preparation=str(out.relative_to(ROOT)),source_HCP_execution=str(oldcfgp.relative_to(ROOT)),common_images=401,candidate_participants=29,nutrients=ix['nutrients'],prompts=['A','B'],conditions_N=['N1','N2','N3'],N0_included=False,
      N_template=prompt['unrelated_template'],N_generation_seed=2026091801,N_permutation_seeds=[2026091811,2026091812,2026091813],N_shared_across_nutrients=True,N_assignments_identical_A_B=True,
      HCP_groups={'random_reference':selected},random_ROI_seed=2026091802,atlas_preparation='outputs/20260918_随机HCP图谱核查_v01',patterns_output='outputs/20260918_随机HCP原始模式_v01',RSA_output='outputs/20260918_控制RSA_v01',
      technical_minimum_voxels=3,coverage_exclusion=False,invalid_rule='keep sampled list; record invalid subject-parcel; do not replace; paired comparisons use common valid subjects and food pairs; if masks differ recompute derived old RSA on intersection, preserve original',
      model_revision=prompt['model_revision'],model_dtype='bfloat16',thinking=False,readout='same_last_prompt_token_32_post_block_pre_final_norm',inference_seed=20260916,
      contrasts=['N-V','T-N','T-V','T-meanP'],N_aggregation='mean_of_three_RSA_scores_not_RDM_or_hidden_mean',P_aggregation='unchanged_mean_five_RSA_scores',
      layer_windows={'all32':list(range(1,33)),'deep17_32':list(range(17,33))},deep_window_status='post_v08_1_exploratory_not_confirmatory',
      matching=dict(role='descriptive_sensitivity_not_noise_correction',primary_unit='individual_HCP_bilateral_parcel',LOSO_absolute_caliper=.02,median_fixed_voxel_ratio_range=[.5,2.0],minimum_comparable_references=5,no_caliper_tuning=True,legacy_ROIs='annotated_continuity_only_no_same_unit_percentile',other_QC=['coverage','session_RDM','run_RDM'],insufficient_rule='report_not_enough_comparable_regions_no_rank_claim'),
      nine_parcel_reference='1000 frozen unique nine-parcel subsets within selected20, equal hemisphere then equal parcel scores; dependent descriptive groups, not a whole-cortex null distribution',
      inference_run=False,extra_brain_preprocessing=[],report_all_individual_parcels=True,retain_original_VTP=True,
      implementation_choices='assistant defaults under approved scope: vocabulary/template, seeds, four words, 1000 group subsets, .02 LOSO and factor2 voxel calipers and minimum5; not universal methodological cutoffs')
    cp=ROOT/'configs/controls_v09_execution_v01.json';assert not cp.exists();cp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,out/'script_snapshot.py')
    audit=dict(status='inputs_frozen_before_new_control_effects',created=datetime.now().astimezone().isoformat(),new_model_inputs=2406,new_model_RDMs=192,random_bilateral_parcels=20,
      input_sha256={str(p):sha(p) for p in [oldcfgp,ixp,labelp,oldmanp,Path(__file__),cp]},output_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()},
      seed_record='see execution config and per-assignment table',git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__),new_brain_or_model_effects_read=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('Fixed random parcels:',selected);print('N examples:',strings[:3])

if __name__=='__main__':main()
