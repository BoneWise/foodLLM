"""Descriptive OFC report with unchanged historical vmPFC reference."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_rsa_patterns import rows
from prepare_roi import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=(ROOT/a.output).resolve()
    assert out.is_relative_to(ROOT/'outputs') and not out.exists();out.mkdir(parents=True)
    data=ROOT/'outputs/20260917_OFC_kcal_RSA_v01';old=ROOT/'outputs/20260917_raw_RSA_v01';patterns=ROOT/'outputs/20260917_OFC原始模式_v01'
    audit=json.loads((data/'audit.json').read_text());assert audit['status']=='complete'
    values=rows(data/'group_curves.csv')
    values += [{**r,'ROI':'vmPFC','P_min':r['P_min_group'],'P_max':r['P_max_group']} for r in rows(old/'group_curves.csv')]
    colors={'V':'#2765a8','T':'#dc6c24','P_mean':'#656b73'};ROIS=['lOFC','mOFC','vmPFC'];layers=np.arange(1,33)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for prompt in ['A','B']:
        fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
        for ri,roi in enumerate(ROIS):
            rr=sorted([r for r in values if r['ROI']==roi and r['prompt']==prompt],key=lambda r:int(r['layer']))
            get=lambda k:np.array([float(r[k]) for r in rr])*1000
            for c in ['V','T','P_mean']:axes[ri,0].plot(layers,get(c),label=c.replace('_mean',' mean'),color=colors[c],lw=1.8)
            axes[ri,0].fill_between(layers,get('P_min'),get('P_max'),color=colors['P_mean'],alpha=.15,label='P range (not CI)')
            axes[ri,1].plot(layers,get('T_minus_V'),color='#31926e',lw=1.8)
            axes[ri,2].plot(layers,get('T_minus_mean_P'),color='#8b4f9d',lw=1.8)
            for ci,title in enumerate(['Alignment','T minus V','T minus mean P']):
                ax=axes[ri,ci];ax.axhline(0,color='#999',lw=.8);ax.set_title(roi+' | '+title);ax.set_ylabel('Spearman rho x 1,000');ax.grid(alpha=.15);ax.set_xlim(1,32)
            axes[ri,0].legend(fontsize=8,loc='best')
        for ax in axes[-1]:ax.set_xlabel('Decoder layer')
        fig.suptitle(f'Kcal / Prompt {prompt} | 493 images, 29 participants | raw RSA, descriptive only',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/f'{prompt}_layer_curves.png',dpi=160);fig.savefig(out/f'{prompt}_layer_curves.pdf');plt.close(fig)
    cov=rows(patterns/'coverage.csv');qc=rows(data/'acquisition_QC.csv')
    qc += [{**r,'ROI':'vmPFC'} for r in rows(old/'acquisition_QC.csv')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ri,roi in enumerate(ROIS):
        rr=[r for r in qc if r['ROI']==roi]
        for k,label,col in [('session_rho','Session','#de945a'),('run_rho','Session + run','#6c9bc0')]:
            offset=-.12 if label=='Session' else .12;x=np.full(len(rr),ri)+offset
            axes[1].scatter(x,[float(r[k]) for r in rr],s=16,alpha=.5,color=col,label=label if ri==0 else None)
            axes[1].scatter(ri+offset,np.mean([float(r[k]) for r in rr]),s=65,color=col,edgecolor='black',marker='D')
        if roi!='vmPFC':
            cc=[r for r in cov if r['ROI']==roi]
            axes[0].scatter(np.full(len(cc),ri),[100*float(r['coverage']) for r in cc],s=22,alpha=.6,label=roi)
        axes[2].scatter(np.full(len(rr),ri)-.1,[float(r['kcal_rho']) for r in rr],s=18,alpha=.55,color='#2765a8',label='Kcal RDM' if ri==0 else None)
        if 'wanting_rho' in rr[0]:axes[2].scatter(np.full(len(rr),ri)+.1,[float(r['wanting_rho']) for r in rr],s=18,alpha=.55,color='#dc6c24',label='Wanting (OFC only)' if ri==0 else None)
    axes[0].set_xticks([0,1],ROIS[:2]);axes[0].set_ylabel('Fixed valid voxels / anatomical ROI (%)');axes[0].set_title('Coverage, all 29 retained');axes[0].set_ylim(80,101)
    axes[1].set_xticks(range(3),ROIS);axes[1].set_ylabel('Brain RDM correlation');axes[1].set_title('Acquisition structure');axes[1].legend()
    axes[2].set_xticks(range(3),ROIS);axes[2].set_ylabel('Brain RDM correlation');axes[2].set_title('Attribute references');axes[2].legend()
    for ax in axes[1:]:ax.axhline(0,color='#999',lw=.8)
    fig.tight_layout();fig.savefig(out/'coverage_and_acquisition_QC.png',dpi=160);fig.savefig(out/'coverage_and_acquisition_QC.pdf');plt.close(fig)
    summary=json.loads((data/'summary.json').read_text());table=[]
    for roi in ['lOFC','mOFC']:
        for prompt in ['A','B']:
            x=summary[roi][prompt];table.append(f"| {roi} | {prompt} | {x['cross_layer_T_minus_V']:+.6f} | {x['cross_layer_T_minus_P']:+.6f} | {x['positive_group_layers_TP']}/32 |")
    report='''# 原版AAL lOFC/mOFC：kcal RSA结果

日期：2026-09-17。493图、29人，A主要、B稳健性；描述性扩展，未运行推断或敏感性。

## 主要发现

**加入Suzuki定义的解剖OFC区域后，仍未出现正确kcal信息稳定优于错配信息的清晰描述性模式。**mOFC的T−平均P在A/B均为28/32层负；lOFC的该差值随提示和层变化。不要将非显著（本轮根本未检验）写成等效或没有作用。

| ROI | Prompt | T−V跨层均值 | T−平均P跨层均值 | T−P为正的层数 |
|---|---|---:|---:|---:|
'''+ '\n'.join(table)+'''

跨层均值只表示量级，层数不是独立重复或显著性检验。完整逐人分数和32层曲线均保留。

![Prompt A](A_layer_curves.png)

![Prompt B](B_layer_curves.png)

## 区域和数据对应

- 按[Suzuki等原文](https://www.nature.com/articles/s41593-017-0008-x)补充图S2a的标签组合，lOFC取原版AAL双侧Frontal_Sup_Orb、Frontal_Mid_Orb、Frontal_Inf_Orb；mOFC取双侧Frontal_Med_Orb，不加入Rectus。
- 图谱和标签来自[FieldTrip发布的原版AAL](https://www.fieldtriptoolbox.org/template/atlas/)，固定下载提交及哈希。SPM单人T1与AAL具有相同网格；通过有记录的ANTs配准映射到MNI152NLin2009cAsym，用最近邻插值保留标签。不是把不同模板直接当作同一空间。
- 模板叠加检查通过；解剖lOFC为7,433体素，mOFC为1,673体素。每人分别与全部既有run有效mask求交，lOFC固定6,560–7,081体素（88.3%–95.3%），mOFC固定1,431–1,662体素（85.5%–99.3%）。这些是覆盖检查，不等于可靠性已经达标。
- 每ROI保留相同13,765条raw valuation单试次β；既有两条截断trial继续剔除，无新增删人或删trial。重新读取的旧vmPFC体素与历史缓存逐值一致。
- lOFC与旧vmPFC球无重叠；mOFC重叠58/272体素（21.3%）。三ROI来自同一批人、同一批trial，不能当作三份独立复制。

## 计算和QC

每人一个跨session脑RDM，脑与模型用1−Pearson，RDM间用Spearman；P1–P5分别算RSA后取平均。直接复用448个既有kcal模型RDM，无模型重跑。完成58张新脑RDM、25,984个逐人RSA分数；三ROI在每人内使用相同食品pair。

lOFC的session/run RDM相关均值为0.1604/0.2410，mOFC为0.0787/0.1507。采集结构明显，尤其lOFC；目前仅报告，未自动中心化、白化、剔除同run pair或替换raw主分析。它提示解释限制，但不能单凭该数值证明所有模型差异均由扫描结构造成。

Raw kcal RDM与脑相关均值：lOFC 0.00254、mOFC 0.00142；wanting参考分别为0.00842、0.00357。上述同样是描述性数值。

![覆盖与采集结构](coverage_and_acquisition_QC.png)

本次新增ROI是在既有vmPFC结果已见后的扩展，非独立确认性复制。新四属性的刺激和P规则已在读取本批OFC RSA结果前保存；不会按这里的正负方向变更ROI、提示或条件。
'''
    (out/'结果摘要.md').write_text(report);shutil.copy2(__file__,out/'script_snapshot.py')
    (out/'report_audit.json').write_text(json.dumps({'status':'generated_pending_visual_QA','input_sha256':{str(p):sha(p) for p in [data/'audit.json',data/'summary.json',data/'group_curves.csv',old/'group_curves.csv',patterns/'coverage.csv']},'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2))

if __name__=='__main__':main()
