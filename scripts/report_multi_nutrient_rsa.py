"""Report all four attributes and all ROIs without layer/attribute selection."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_rsa_patterns import rows
from prepare_roi import sha,write_csv

ROOT=Path(__file__).resolve().parents[1]
ROIS=['lOFC','mOFC','vmPFC'];ATTRS=['protein_100g','fat_100g','carbs_100g','fibre_100g'];NAMES=['Protein','Fat','Carbohydrate','Fibre']

def main():
    ap=argparse.ArgumentParser()
    for k in ['A','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--B');ap.add_argument('--preview-A',action='store_true')
    args=ap.parse_args();out=(ROOT/args.output).resolve();assert out.is_relative_to(ROOT/'outputs') and not out.exists()
    assert bool(args.B)!=args.preview_A,'Provide B for a final report, or --preview-A for a clearly marked preview'
    roots={'A':ROOT/args.A}
    if args.B:roots['B']=ROOT/args.B
    audits={p:json.loads((r/'audit.json').read_text()) for p,r in roots.items()}
    assert all(a['status']=='complete' and a['images']==401 for a in audits.values())
    ia=json.loads((roots['A']/'index.json').read_text())
    if args.B:
        ib=json.loads((roots['B']/'index.json').read_text());assert ia['image_ids']==ib['image_ids'] and ia['subject_ids']==ib['subject_ids']
    out.mkdir(parents=True);curves=sum([rows(r/'group_curves.csv') for r in roots.values()],[]);beh=sum([rows(r/'behavior_scores.csv') for r in roots.values()],[])
    layers=np.arange(1,33);colors={'V':'#2765a8','T':'#dc6c24','P_mean':'#656b73'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for prompt in roots:
        for mode in ['alignment','contrasts']:
            fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True)
            for ri,roi in enumerate(ROIS):
                for ai,a in enumerate(ATTRS):
                    rr=sorted([r for r in curves if r['ROI']==roi and r['nutrient']==a and r['prompt']==prompt],key=lambda r:int(r['layer']));get=lambda k:np.array([float(r[k]) for r in rr])*1000
                    ax=axes[ri,ai]
                    if mode=='alignment':
                        for c,col in colors.items():ax.plot(layers,get(c),color=col,label=c.replace('_mean',' mean'),lw=1.6)
                        ax.fill_between(layers,get('P_min'),get('P_max'),color=colors['P_mean'],alpha=.15,label='P range, not CI')
                    else:
                        ax.plot(layers,get('T_minus_V'),color='#31926e',label='T minus V',lw=1.6)
                        ax.plot(layers,get('T_minus_mean_P'),color='#8b4f9d',label='T minus mean P',lw=1.6)
                    ax.axhline(0,color='#999',lw=.7);ax.grid(alpha=.15);ax.set_xlim(1,32);ax.set_title(roi+' / '+NAMES[ai])
                    if ai==0:ax.set_ylabel('Spearman rho x 1,000')
                    if ri==2:ax.set_xlabel('Decoder layer')
                    if ri==0 and ai==0:ax.legend(fontsize=7,loc='best')
            fig.suptitle(f'Prompt {prompt} | 401 common images / 29 participants | raw RSA, descriptive only',fontsize=13)
            fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/f'{prompt}_{mode}.png',dpi=155);fig.savefig(out/f'{prompt}_{mode}.pdf');plt.close(fig)
        for ci,title in [(0,'T minus V'),(1,'T minus mean P')]:
            deltas={a:np.load(roots[prompt]/a/'contrasts.npy')[:,:,:,ci]*1000 for a in ATTRS}
            limit=max(float(np.abs(d).max()) for d in deltas.values())
            fig,axes=plt.subplots(3,4,figsize=(16,10),sharex=True,sharey=True,layout='constrained')
            for ri,roi in enumerate(ROIS):
                for ai,a in enumerate(ATTRS):
                    ax=axes[ri,ai];im=ax.imshow(deltas[a][ri],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,origin='upper',extent=(.5,32.5,29.5,.5))
                    ax.set_title(roi+' / '+NAMES[ai]);ax.set_xticks([1,8,16,24,32]);ax.set_yticks([1,8,15,22,29],[ia['subject_ids'][j-1] for j in [1,8,15,22,29]])
                    if ri==2:ax.set_xlabel('Decoder layer')
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Individual delta rho x 1,000')
            fig.suptitle(f'Prompt {prompt} | {title} | all 29 participants; shared color scale',fontsize=13)
            fig.savefig(out/f'{prompt}_participant_delta_{ci}.png',dpi=155);fig.savefig(out/f'{prompt}_participant_delta_{ci}.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True)
        for ai,a in enumerate(ATTRS):
            ar=np.load(roots[prompt]/a/'behavior_scores.npy');ax=axes[ai]
            for c,vec in [('V',ar[:,0]),('T',ar[:,1]),('P_mean',ar[:,2:].mean(1))]:ax.plot(layers,vec,color=colors[c],label=c.replace('_mean',' mean'),lw=1.6)
            ax.fill_between(layers,ar[:,2:].min(1),ar[:,2:].max(1),color=colors['P_mean'],alpha=.15)
            ax.set_title(NAMES[ai]);ax.set_xlabel('Decoder layer');ax.grid(alpha=.15);ax.axhline(0,color='#999',lw=.7)
        axes[0].set_ylabel('Model - group wanting Spearman rho');axes[0].legend(fontsize=8)
        fig.suptitle(f'Prompt {prompt} | Behavioral reference, 401 images | shaded P range is not CI',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'{prompt}_wanting.png',dpi=155);fig.savefig(out/f'{prompt}_wanting.pdf');plt.close(fig)
    summary=[];participant_summary=[];wanting_table=[]
    for prompt in roots:
        for a in ATTRS:
            score=np.load(roots[prompt]/a/'neural_scores.npy');delta=np.load(roots[prompt]/a/'contrasts.npy');behavior=np.load(roots[prompt]/a/'behavior_scores.npy')
            wanting_table.append(f"| {prompt} | {NAMES[ATTRS.index(a)]} | {behavior[:,0].mean():.6f} | {behavior[:,1].mean():.6f} | {behavior[:,2:].mean():.6f} |")
            for ri,roi in enumerate(ROIS):
                for ci,contrast in enumerate(['T_minus_V','T_minus_mean_P']):
                    individual=delta[ri,:,:,ci].mean(1)
                    participant_summary.append({'prompt':prompt,'nutrient':a,'ROI':roi,'contrast':contrast,'positive_participants':int((individual>0).sum()),'participants':len(individual),'median':float(np.median(individual)),'min':float(individual.min()),'max':float(individual.max())})
                summary.append({'prompt':prompt,'nutrient':a,'ROI':roi,'mean_V':float(score[ri,:,:,0].mean()),'mean_T':float(score[ri,:,:,1].mean()),'mean_P':float(score[ri,:,:,2:].mean()),
                    'T_minus_V':float(delta[ri,:,:,0].mean()),'T_minus_mean_P':float(delta[ri,:,:,1].mean()),
                    'TV_positive_layers':int((delta[ri,:,:,0].mean(0)>0).sum()),'TP_positive_layers':int((delta[ri,:,:,1].mean(0)>0).sum()),
                    'wanting_mean_V':float(behavior[:,0].mean()),'wanting_mean_T':float(behavior[:,1].mean()),'wanting_mean_P':float(behavior[:,2:].mean())})
    write_csv(out/'descriptive_summary.csv',summary);write_csv(out/'group_curves.csv',curves)
    write_csv(out/'participant_layer_mean_descriptive.csv',participant_summary)
    if args.preview_A:
        (out/'A阶段快照.md').write_text('# A阶段描述性快照\n\nA的全部输入和RSA已验证；B尚未完成。以下不是完整A/B结论，不含显著性检验，不用于挑选属性、ROI或层。\n\n'+''.join(f'![{name}]({name}.png)\n\n' for name in ['A_alignment','A_contrasts','A_participant_delta_0','A_participant_delta_1','A_wanting']))
        shutil.copy2(__file__,out/'script_snapshot.py')
        (out/'report_audit.json').write_text(json.dumps({'status':'A_preview_generated_B_pending','input_sha256':{str(roots['A']/'audit.json'):sha(roots['A']/'audit.json')},'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2))
        return
    table=[]
    for a,name in zip(ATTRS,NAMES):
        for roi in ROIS:
            x=next(r for r in summary if r['nutrient']==a and r['ROI']==roi and r['prompt']=='A');y=next(r for r in summary if r['nutrient']==a and r['ROI']==roi and r['prompt']=='B')
            table.append(f"| {name} | {roi} | {x['T_minus_V']:+.6f} | {x['T_minus_mean_P']:+.6f} | {y['T_minus_V']:+.6f} | {y['T_minus_mean_P']:+.6f} |")
    ref=rows(roots['A']/'nutrient_references.csv');qc=rows(roots['A']/'acquisition_QC.csv');reference_table=[];qc_table=[]
    for a,name in zip(ATTRS,NAMES):
        means=[float(np.mean([float(r['nutrient_brain_rho']) for r in ref if r['nutrient']==a and r['ROI']==roi])) for roi in ROIS]
        bw=float(next(r['nutrient_wanting_rho'] for r in ref if r['nutrient']==a))
        reference_table.append('| '+name+' | '+' | '.join(f'{x:+.6f}' for x in [*means,bw])+' |')
    for roi in ROIS:
        means=[float(np.mean([float(r[k]) for r in qc if r['ROI']==roi])) for k in ['session_rho','run_rho','wanting_rho']]
        qc_table.append('| '+roi+' | '+' | '.join(f'{x:+.6f}' for x in means)+' |')
    text='''# 四种单营养属性：401图、三个ROI的描述性RSA

日期：2026-09-17。A主要、B稳健性，Qwen3.5-9B、thinking关闭、固定32层pre-answer token。**本报告不包含显著性检验、置信带或最佳层筛选。**

所有条件中的人类脑数据完全相同。这里的变化是模型输入改变后的对齐分数变化，不是人脑看到营养标签后的反应变化。

## 全部结果的量级

下表是跨32层均值，只帮助读数量级，不是新的主检验；必须同时看完整曲线。

| 属性 | ROI | A：T−V | A：T−P | B：T−V | B：T−P |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

## 完整逐层曲线

神经图的纵轴以ρ×1,000显示，每个panel独立缩放。P阴影是5套分数的范围，不是置信区间。三ROI来自同一批参与者，不能把一个区域的正向曲线与另一区域的负向曲线直接当作显著区域差异。

![A对齐](A_alignment.png)

![A两对比](A_contrasts.png)

![B对齐](B_alignment.png)

![B两对比](B_contrasts.png)

## 逐人差值

全部29人的两种差值都保留；每张图内使用统一色标，无裁切。行顺序对应已保存的subject_id索引。

![A逐人T−V](A_participant_delta_0.png)

![A逐人T−P](A_participant_delta_1.png)

![B逐人T−V](B_participant_delta_0.png)

![B逐人T−P](B_participant_delta_1.png)

## Wanting行为锚点

使用401图的人类可用评分均值绝对差RDM，所有模型条件共用。该群体均值RDM不提供29份独立的行为RSA重复。

![A行为](A_wanting.png)

![B行为](B_wanting.png)

## 数据、验证和解释边界

- 同一401图、同一参与者有效pair、A/B、三ROI、四属性及全部条件一致。仅V复用旧特征；每种新条件一次只提供一个g/100 g字段。
- 19,248个新增输入均经过缓存身份、模型revision、图片哈希、提示、答案token、32层形状、finite值和评分概率核验。矩阵是BF16输出转float32后的逐值复制。A完成技术验证后才运行B，不看A效应筛B。
- 模型和脑都使用1−Pearson RDM，RDM间Spearman；各P分别算RSA后平均。不把food pairs当独立样本。每ROI保持raw trial β，无新脑预处理、删人或删试次。
- 脑RDM取既有完整RDM的401图有效子矩阵；子集后重新求秩。没有把493图的RSA分数直接挪作401图结果。
- 各营养解释RDM使用原数据库每100 g值的绝对差；提示显示两位小数。Session/run仅作QC，输出保留；没有在看到效果后自动消除采集结构。
- 401图是在既有食品形态类别与保守错配范围下四属性共同可配对的集合，仅覆盖25/40候选形态组。选择只用元数据，但存在食品范围限制，不能直接泛化到全部568图。
- P为同类食品内的一对一数值置换，保留边际分布；数值强度规则是助手运行前的操作性选择，未做人类可信度验证，不是四属性生物效应等强的设计。因此不做“最强营养”的确认性排名。
- 所有新实验沿用同一批已分析的脑数据，且研究者已见过kcal/旧行为结果；这是透明的后续扩展，不是独立盲态复制。mOFC与旧vmPFC存在部分重叠。

正式推断、采集结构敏感性及跨新食品泛化仍未执行。T−V或T−P为正的层数仅是描述，不能替代校正后的participant统计。
'''
    text+='\n## 固定参考RDM与扫描结构\n\n以下均在同一401图范围、同一参与者有效pair上计算，A/B共用。单营养RDM相关低或负，不等于已经否定营养信息与脑表征的关系；模型分数超过该简单参考也不能单独证明复杂整合机制。\n\n| 原始属性RDM | lOFC | mOFC | vmPFC | 群体wanting |\n|---|---:|---:|---:|---:|\n'+'\n'.join(reference_table)+'\n\n| 脑ROI | Session相关 | Session＋run相关 | Wanting相关 |\n|---|---:|---:|---:|\n'+'\n'.join(qc_table)+'\n'
    text+='\n## Wanting数量级汇总\n\n跨32层平均仅供阅读数量级，不能替代完整曲线；同一群体wanting RDM不作为29个独立重复。\n\n| 提示 | 属性 | V | T | 平均P |\n|---|---|---:|---:|---:|\n'+'\n'.join(wanting_table)+'\n\n逐人跨层差值的正向人数、中位数和范围见[完整逐人描述](participant_layer_mean_descriptive.csv)。正向人数与正向层数都不构成显著性检验。\n'
    (out/'结果摘要.md').write_text(text);shutil.copy2(__file__,out/'script_snapshot.py')
    (out/'report_audit.json').write_text(json.dumps({'status':'generated_pending_visual_QA','input_sha256':{str(r/'audit.json'):sha(r/'audit.json') for r in roots.values()},'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2))

if __name__=='__main__':main()
