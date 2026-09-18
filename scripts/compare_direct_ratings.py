"""Describe verified Qwen direct answers against existing human ratings; no fitting."""
import json
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_layer_analysis import sha

OUT=Path('outputs/20260916_Qwen直接评分与人类比较_v01')


def main():
    OUT.mkdir(exist_ok=False)
    project=json.loads(Path('configs/project.json').read_text())
    root=Path(project['paths']['analysis_tables'])
    source=Path('outputs/20260916_Qwen_AB_features_v03/records.json')
    records=json.loads(source.read_text())
    idx=json.loads((root/'index.json').read_text())
    h=np.load(root/'human_ratings.npy');primary=np.load(root/'primary_mask.npy')
    lookup={(r['prompt_id'],r['condition'],r['image_id']):r for r in records}
    assert len(lookup)==len(records)==7202 and np.isfinite(h).sum()==15869
    result={'scope':'post_estimate_descriptive_direct_output_comparison_no_fitting_or_inference',
            'human_target':'image_mean_across_available_retained_participants',
            'expected_rating':'sum k times probability conditional on answers 1..4',
            'hard_rating':'verified full-vocabulary argmax answer',
            'image_weighting':'equal','human_trial_distribution_weighting':'observed_trials',
            'P_aggregation':'mean_of_five_separately_computed_metrics_not_mean_prediction',
            'samples':{}}
    plot_values={};agreement_rows=[];max_baseline_error=0.
    for sample,mask in [('primary',primary),('full_VT',np.ones(568,dtype=bool))]:
        ids=np.array(idx['image_ids'])[mask].tolist();human=h[:,mask];y=np.nanmean(human,axis=0)
        obs=np.isfinite(human)
        block={'images':len(ids),'human_ratings':int(obs.sum()),'human_mean_image_weighted':float(y.mean()),
               'human_image_sd':float(y.std()),'human_image_min':float(y.min()),'human_image_max':float(y.max()),
               'human_trial_rating_counts':[int((human==k).sum()) for k in range(1,5)],'prompts':{}}
        for prompt in ['A','B']:
            old=np.load(Path('outputs/20260916_输出基线_v01')/f'{sample}_{prompt}.npz')
            oldmeta=json.loads((Path('outputs/20260916_输出基线_v01')/f'{sample}_{prompt}.json').read_text())
            assert ids==oldmeta['image_ids'] and np.array_equal(y,old['target'])
            conditions=['V','T']+([f'P{k}' for k in range(1,6)] if sample=='primary' else [])
            pb={};expected={};hard_values={}
            for ci,c in enumerate(conditions):
                rr=[lookup[prompt,c,i] for i in ids]
                probs=np.array([r['rating_probs'] for r in rr]);e=probs@np.arange(1,5)
                hard=np.array([r['hard_rating'] for r in rr])
                assert np.allclose(probs.sum(1),1,atol=1e-6)
                assert np.array_equal(hard,probs.argmax(1)+1)
                assert all(r['global_argmax_is_rating'] and r['global_argmax_token']==str(r['hard_rating']) for r in rr)
                assert np.allclose(e,[r['expected_rating'] for r in rr],rtol=0,atol=1e-12)
                diff=e-y;bias=float(diff.mean());mse=float(np.mean(diff**2))
                max_baseline_error=max(max_baseline_error,float(np.max(np.abs(e-old['raw_expected'][ci]))))
                pb[c]={'expected_mean':float(e.mean()),'expected_sd':float(e.std()),'bias':bias,
                    'pearson_r':float(pearsonr(e,y).statistic),'spearman_rho':float(spearmanr(e,y).statistic),
                    'mse':mse,'rmse':float(np.sqrt(mse)),'mae':float(np.mean(np.abs(diff))),
                    'bias_squared':bias**2,'centered_mse':float(np.mean((diff-bias)**2)),
                    'hard_rating_counts':[int((hard==k).sum()) for k in range(1,5)],
                    'hard_mean':float(hard.mean()),'hard_pearson_r':float(pearsonr(hard,y).statistic),
                    'exact_match_to_individual_trials':float((human==hard)[obs].mean()),
                    'constant_four_exact_match':float((human==4)[obs].mean())}
                assert abs(mse-pb[c]['bias_squared']-pb[c]['centered_mse'])<1e-12
                expected[c]=e;hard_values[c]=hard
                if sample=='primary':plot_values[prompt,c]=(y,e,hard)
                agreement_rows.extend({'sample':sample,'prompt':prompt,'condition':c,'image_id':i,
                    'human_mean':float(yy),'qwen_expected':float(ee),'qwen_hard':int(hh)}
                    for i,yy,ee,hh in zip(ids,y,e,hard))
            pb['T_vs_V']={'mean_expected_change':float(np.mean(expected['T']-expected['V'])),
                         'hard_answer_change_fraction':float(np.mean(hard_values['T']!=hard_values['V']))}
            if sample=='primary':
                pp=[pb[f'P{k}'] for k in range(1,6)]
                pb['P_mean_metrics']={k:float(np.mean([r[k] for r in pp])) for k in ['expected_mean','pearson_r','spearman_rho','mse','rmse','mae']}
                pb['T_vs_P']={'mean_per_replica_expected_pearson':float(np.mean([pearsonr(expected['T'],expected[f'P{k}']).statistic for k in range(1,6)])),
                    'mean_per_replica_absolute_expected_difference':float(np.mean([np.mean(np.abs(expected['T']-expected[f'P{k}'])) for k in range(1,6)]))}
            pb['existing_food_holdout_calibration']={
                'training_mean_mse':float(np.mean((old['training_mean']-y)**2)),
                'V_trained_expected_output_V_mse':float(np.mean((old['expected'][0]-y)**2)),
                'V_trained_expected_output_T_mse':float(np.mean((old['expected'][1]-y)**2))}
            block['prompts'][prompt]=pb
        result['samples'][sample]=block
    assert max_baseline_error<1e-12
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'image_comparisons.json').write_text(json.dumps(agreement_rows,ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,7.7),sharex=True,sharey=True)
    for ri,p in enumerate(['A','B']):
        for ci,c in enumerate(['V','T','P1']):
            y,e,_=plot_values[p,c];ax=axes[ri,ci];m=result['samples']['primary']['prompts'][p][c]
            ax.scatter(y,e,s=13,alpha=.5,color='#176B87' if p=='A' else '#D56B36',edgecolors='none')
            ax.plot([1,4],[1,4],ls='--',color='#777',lw=1)
            ax.set_title(f"{p} | {c} | r = {m['pearson_r']:.3f}, rho = {m['spearman_rho']:.3f}")
            ax.set(xlim=(1,4),ylim=(1,4),xticks=[1,2,3,4],yticks=[1,2,3,4]);ax.set_aspect('equal')
            if ci==0:ax.set_ylabel('Qwen expected rating')
            if ri==1:ax.set_xlabel('Human mean rating')
    fig.suptitle('Direct Qwen ratings vs human wanting | 493 images',fontsize=15)
    fig.text(.5,.015,'A: primary prompt | B: wording robustness | P1: illustrative mismatch replica | Dashed line: equal scores',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(OUT/'direct_ratings.png',dpi=170);fig.savefig(OUT/'direct_ratings.pdf');plt.close(fig)
    main=result['samples']['primary'];a=main['prompts']['A'];b=main['prompts']['B']
    lines=['# Qwen直接评分与人类真实评分','',
        '这是对已提取输出的描述性比较，无新训练、无选层、无显著性检验。主要样本493图，29人共13,767条评分；每图以现有参与者的平均wanting为人类目标。两条脑试次剔除不改变行为样本。',
        '', '## 评分层级','',
        '- 人类评分是参与者自身1–4分wanting；每张图可有多个人的不同答案。',
        '- Qwen直接答案是1/2/3/4中的最高概率答案；期望评分是四个候选概率加权平均，不等于实际生成的小数答案。',
        '- 本比较使用直接输出。此前32层隐藏表征经人类评分训练得到的wanting读出，是另一种分析。',
        '',f"人类图片均分的平均值为 **{main['human_mean_image_weighted']:.3f}**。",'',
        '| 提示 | 条件 | Qwen期望均分 | Pearson r | Spearman排序相关 | RMSE（分） |',
        '|---|---|---:|---:|---:|---:|']
    for p in ['A','B']:
        for c in ['V','T','P_mean_metrics']:
            m=main['prompts'][p][c];lines.append(f"| {p} | {c} | {m['expected_mean']:.3f} | {m['pearson_r']:.3f} | {m['spearman_rho']:.3f} | {m['rmse']:.3f} |")
    lines += ['', 'V仅看图；T正确能量密度；P为合理错配。P行是5套分别计算指标后的平均，未先平均预测。相关是图片间描述性对应；不能把它当成个体评分一致率或脑对齐指标。',
        '', '## 一致之处与差异','',
        f"1. 只看图时保留部分人类偏好排序：主A的r={a['V']['pearson_r']:.3f}、排序相关={a['V']['spearman_rho']:.3f}；B亦为正。",
        f"2. 打分整体偏高且集中。A/V的1、2、3、4分数量为{a['V']['hard_rating_counts']}，其中4分占{100*a['V']['hard_rating_counts'][3]/493:.1f}%；人类个体评分中4分占{100*a['V']['constant_four_exact_match']:.1f}%。二者分别按图片／实际试次统计。",
        f"3. A/V硬答案与每条人类评分完全相同的比例为{100*a['V']['exact_match_to_individual_trials']:.1f}%，始终答4为{100*a['V']['constant_four_exact_match']:.1f}%。这不等于‘比随机高或低’，也不替代群体均值相关。",
        f"4. A加正确热量后，均分从{a['V']['expected_mean']:.3f}降到{a['T']['expected_mean']:.3f}，但r从{a['V']['pearson_r']:.3f}降到{a['T']['pearson_r']:.3f}。原始MSE降低包含整体高估减弱；去掉均值偏差后的误差从{a['V']['centered_mse']:.3f}升到{a['T']['centered_mse']:.3f}，不支持图片偏好对应改善。",
        f"5. 正确和错配营养条件相似。A的T与各P期望评分相关的平均为{a['T_vs_P']['mean_per_replica_expected_pearson']:.4f}，逐图绝对差平均仅{a['T_vs_P']['mean_per_replica_absolute_expected_difference']:.4f}分。仅这些结果尚不能证明数字完全无效，也不能区分营养文字、数字及格式各自贡献。",
        f"6. 措辞会改变营养条件反应：B的T均分为{b['T']['expected_mean']:.3f}、r={b['T']['pearson_r']:.3f}，保留部分排序但弱于B/V。不能把A的近零相关概括成所有prompt下都无相关。",'',
        '## 与校准结果的联系','',
        f"既有食品组留出中，仅用V训练线性校准期望输出后，A/V的MSE为{a['existing_food_holdout_calibration']['V_trained_expected_output_V_mse']:.3f}，优于训练集均值基线的{a['existing_food_holdout_calibration']['training_mean_mse']:.3f}。说明原始输出含有可用的群体偏好信息，但原始1–4分尺度未与人类对齐。冻结这一V校准后用于T，MSE为{a['existing_food_holdout_calibration']['V_trained_expected_output_T_mse']:.3f}。没有本轮额外拟合。",'',
        '## 边界','',
        '人类只看图片；T/P输入的营养标签没有对应的人类营养操纵评分。因此只能判断模型条件输出与现有人类图片偏好的接近程度，不能推出人类看到标签后的变化。描述性相关没有处理食品组依赖的区间估计；不报告p值或确认性结论。',
        '', '全568图V/T补充、每套P与逐图数据见metrics.json和image_comparisons.json。完整原始行为、提示、特征和正式分析入口均未改变。','',
        '![直接评分散点](direct_ratings.png)','']
    (OUT/'比较摘要.md').write_text('\n'.join(lines))
    shutil.copy2(__file__,OUT/'script_snapshot.py')
    inputs=[source,root/'index.json',root/'human_ratings.npy',root/'primary_mask.npy',Path('configs/project.json')]
    inputs+=list(Path('outputs/20260916_输出基线_v01').glob('*.npz'))+list(Path('outputs/20260916_输出基线_v01').glob('*.json'))
    audit={'status':'complete','records':7202,'seed':None,'randomness':'none_no_fit',
        'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'validation':{'existing_raw_baseline_prediction_max_error':max_baseline_error,'IDs_matched':True,
                      'argmax_and_probability_consistency':True,'MSE_bias_decomposition':True},
        'input_sha256':{str(p):sha(p) for p in inputs},'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
