"""Audit completed predictions and create descriptive tables and scientific plots."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_layer_analysis import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True)
    ap.add_argument('--baselines',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();root=Path(args.input);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((root/'audit.json').read_text())
    assert audit['status']=='complete' and len(audit['completed'])==128
    cfg=json.loads((root/'execution_config.json').read_text())
    mr=Path(cfg['matrix_root']);ma=json.loads((mr/'audit.json').read_text())
    checked=0
    for group in ma['groups']:
        for name,entry in group['files'].items():
            assert sha(mr/name)==entry['sha256'];checked+=1
    with (root/'layer_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    summaries=[];rank_deficient=0;largest_error=0.;subject_rows=[]
    for name in audit['completed']:
        meta=json.loads((root/(name+'.json')).read_text());a=np.load(root/(name+'.npz'))
        for suffix in ['.json','.npz']:
            assert sha(root/(name+suffix))==audit['output_sha256'][name+suffix]
        # Independently reconstruct the published metrics from saved held-out errors.
        mh=a['behavior_squared_errors'].mean(1)
        mb=np.nanmean(a['brain_squared_errors'],axis=-1).mean(1)
        for c,m in zip(meta['conditions'],mh):
            largest_error=max(largest_error,abs(float(m)-meta['metrics']['behavior'][c]))
        for k,mode in enumerate(meta['brain_modes']):
            for c,m in zip(meta['brain_conditions'],mb[k]):
                largest_error=max(largest_error,abs(float(m)-meta['metrics'][mode][c]))
            subject_mse=a['brain_subject_mse'][k]
            assert np.allclose(subject_mse,np.nanmean(a['brain_squared_errors'][k],axis=-1))
            for s,subject in enumerate(meta['subject_ids']):
                m=dict(zip(meta['brain_conditions'],subject_mse[s]))
                subject_rows.append({'sample':meta['sample'],'prompt':meta['prompt'],'layer':meta['layer'],
                    'mode':mode,'subject':subject,'G':m['V']-m['T'],
                    'S':float(np.mean([v for c,v in m.items() if c.startswith('P')]))-m['T']
                         if len(meta['conditions'])>2 else '',
                    'T_vs_baseline':m['baseline']-m['T']})
        rank_deficient+=meta['rank_deficient_brain_fits']
        assert a['test_counts'].max()==1
        assert np.all(a['test_counts'].sum(axis=(1,2))==meta['observed_trials'])
    assert largest_error<1e-12
    for sample in cfg['samples']:
        for prompt in cfg['prompts']:
            for endpoint in ['behavior','food','food_session']:
                subset=[r for r in rows if (r['sample'],r['prompt'],r['endpoint'])==(sample,prompt,endpoint)]
                assert sorted(int(r['layer']) for r in subset)==list(range(1,33))
                summary={'sample':sample,'prompt':prompt,'endpoint':endpoint}
                for key in ['baseline','V','T','P_mean','G','S','T_vs_baseline']:
                    if subset[0][key]!='':
                        summary[key]=float(np.mean([float(r[key]) for r in subset]))
                summary['G_percent_of_V']=100*summary['G']/summary['V']
                summary['positive_G_layers']=sum(float(r['G'])>0 for r in subset)
                if 'S' in summary:
                    summary['S_percent_of_P']=100*summary['S']/summary['P_mean']
                    summary['positive_S_layers']=sum(float(r['S'])>0 for r in subset)
                if 'baseline' in summary:
                    summary['T_vs_baseline_percent']=100*summary['T_vs_baseline']/summary['baseline']
                summaries.append(summary)
    (out/'mean32_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (out/'subject_layer_contrasts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(subject_rows[0]));w.writeheader();w.writerows(subject_rows)
    with (Path(args.baselines)/'metrics.csv').open() as f:
        output_baselines=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),sharex=True)
    colors={'A':'#176B87','B':'#D56B36'}
    for j,(contrast,reference,label) in enumerate([('G','V','T improvement over V'),('S','P_mean','T improvement over mean P loss')]):
        for i,endpoint in enumerate(['behavior','food']):
            ax=axes[i,j]
            for prompt in cfg['prompts']:
                subset=sorted([r for r in rows if r['sample']=='primary' and r['prompt']==prompt
                               and r['endpoint']==endpoint],key=lambda r:int(r['layer']))
                ax.plot([int(r['layer']) for r in subset],[100*float(r[contrast])/float(r[reference]) for r in subset],
                        color=colors[prompt],lw=2,label=f'{prompt}: '+('primary' if prompt=='A' else 'prompt robustness'))
                if endpoint=='food':
                    session=sorted([r for r in rows if r['sample']=='primary' and r['prompt']==prompt
                                    and r['endpoint']=='food_session'],key=lambda r:int(r['layer']))
                    ax.plot([int(r['layer']) for r in session],[100*float(r[contrast])/float(r[reference]) for r in session],
                            color=colors[prompt],lw=1.4,ls='--',alpha=.85,label=f'{prompt}: food x session')
            ax.axhline(0,color='#888888',lw=.8);ax.grid(alpha=.12)
            ax.set_title(('Human wanting' if i==0 else 'vmPFC')+' | '+label,fontsize=11)
            ax.set_ylabel('Held-out MSE reduction (%)')
            ax.set_xlim(1,32);ax.set_xticks([1,8,16,24,32])
            if i: ax.set_xlabel('Decoder layer')
            ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Correct energy-density information: layerwise effect estimates',fontsize=15,y=.99)
    fig.text(.5,.018,'493 images | Positive = lower error for T | Descriptive estimates; no confidence intervals or significance tests',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(out/'layerwise_effects.png',dpi=180)
    fig.savefig(out/'layerwise_effects.pdf')
    plt.close(fig)
    text=['# 逐层交叉验证结果：描述性摘要','',
          '本结果使用既定 v05 样本、折号与模型，未按效果改变提示、选层或更换主分析。尚未完成完整重拟合 bootstrap 与区间覆盖验证，以下不提供显著性结论。',
          '', '## 493 图主样本：32 层等权平均','',
          'G = MSE(V) − MSE(T)；S = mean[MSE(P1)…MSE(P5)] − MSE(T)。正数表示 T 误差更低。行为误差为评分²，脑误差为 ROI β²，不能直接比较绝对数值。','',
          '| Prompt | 终点 | G | 相对 V | S | 相对 P | T 比共同脑基线改善 |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['sample']=='primary':
            base=(f"{r['T_vs_baseline_percent']:+.3f}%" if 'baseline' in r else '—')
            text.append(f"| {r['prompt']} | {r['endpoint']} | {r['G']:+.6g} | {r['G_percent_of_V']:+.3f}% | {r['S']:+.6g} | {r['S_percent_of_P']:+.3f}% | {base} |")
    text += ['', '百分比使用跨层平均误差作分母；图中百分比按每层误差作分母。food 为食品组留出；food_session 为食品组与 session 同时留出。脑误差先在每人全部测试试次汇总，再对 29 人等权平均。',
             '', '## 568 图 V/T：独立重拟合补充','',
             '| Prompt | 终点 | G | 相对 V |','|---|---|---:|---:|']
    for r in summaries:
        if r['sample']=='full_VT':
            text.append(f"| {r['prompt']} | {r['endpoint']} | {r['G']:+.6g} | {r['G_percent_of_V']:+.3f}% |")
    text += ['', '## 主样本输出基线','',
             '| Prompt | 输出读出 | V MSE | T MSE | G | S |','|---|---|---:|---:|---:|---:|']
    for r in output_baselines:
        if r['sample']=='primary':
            text.append(f"| {r['prompt']} | {r['baseline']} | {float(r['V']):.6f} | {float(r['T']):.6f} | {float(r['G']):+.6f} | {float(r['S']):+.6f} |")
    text += ['', 'raw_expected 为原始期望评分；expected / probabilities 为仅在 V 训练的标量／四概率 ridge 校准，固定应用于其他条件。所有模型使用相同食品折。',
             '', '## 验证与边界','',
             f'- 128 个提示×层×样本任务完成；32 层不是 32 个独立样本。矩阵输入 {checked} 个文件哈希与原审计完全一致。',
             '- 保存的逐图／逐人误差重新汇总，与输出表一致；两个脑验证模式中，每个有效试次恰好得到一次留出预测。',
             f'- 低维脑拟合中秩不足计数：{rank_deficient}；未因此筛人或筛层。',
             '- A 为主提示；B 只做措辞敏感性检查。5 套 P 为固定实现平均损失，不是额外被试或独立重复实验。',
             '- 现有人类只看图，未接受营养标签操作；模型输入效应不能直接说明人类营养整合机制。',
             '- ROI 几何入口不等于功能可靠性证明；食品×session 检查不等于跨新人泛化。',
             '- 本次描述性估计先于推断覆盖验证；后续推断参数需说明设定于这些估计可见之后，不能倒写为盲态预注册。',
             '', '![全层描述性曲线](layerwise_effects.png)','']
    if cfg.get('neural_exclusion_policy'):
        policy=json.loads(Path(cfg['neural_exclusion_policy']).read_text())
        counts=policy['expected_brain_counts']
        text[3:3]=['',
            '当前脑数据按用户决定剔除 sub-002 的 ses-01/04/48（图443）与 ses-02/04/48（图43），原因是估值事件结束超过采集结束。',
            f"脑分析主样本 {counts['primary']:,} 条、全样本 {counts['full']:,} 条；29人、493/568图片及食品折不变。行为仍为13,767/15,869条评分。",
            '该决定发生在原结果和敏感性结果可见之后。行为与上游wanting读出复用原结果；脑结果采用已核验的剔除后重拟合，旧结果保留。','']
    (out/'结果摘要.md').write_text('\n'.join(text))
    validation={'status':'passed','input_matrix_files_hash_verified':checked,
        'completed_layer_tasks':128,'metric_reconstruction_max_error':largest_error,
        'heldout_once_per_observed_trial':True,'rank_deficient_brain_fits':rank_deficient,
        'inference_complete':False,'intervals_or_p_values_reported':False}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    shutil.copy2(__file__,out/'summarize_layer_analysis.py')
    (out/'audit.json').write_text(json.dumps({'status':'complete','matplotlib':matplotlib.__version__,
        'analysis_audit_sha256':sha(root/'audit.json'),
        'baseline_audit_sha256':sha(Path(args.baselines)/'audit.json'),
        'script_sha256':sha(__file__),
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
