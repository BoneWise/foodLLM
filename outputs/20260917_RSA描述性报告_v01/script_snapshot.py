"""Describe fixed v06.1 RSA results; no p-values, intervals or new analyses."""
import argparse,json,shutil,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_roi import sha
from extract_rsa_patterns import rows

COL={'V':'#326b9e','T':'#cf6635','P':'#68736b','TV':'#cf6635','TP':'#7955a1'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    source=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((source/'audit.json').read_text());assert audit['status']=='complete' and not audit['inference_run']
    meta=json.loads((source/'image_index.json').read_text());subjects=meta['subject_ids'];x=np.arange(1,33)
    q=rows(source/'acquisition_QC.csv');kcal=np.array([float(r['kcal_rho']) for r in q]);ref=json.loads((source/'references.json').read_text())
    neural={p:np.load(source/p/'neural_scores.npy') for p in ['A','B']};behavior={p:np.load(source/p/'behavior_scores.npy') for p in ['A','B']};delta={p:np.load(source/p/'contrasts.npy') for p in ['A','B']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
    for j,p in enumerate(['A','B']):
        means=neural[p].mean(axis=0);ax=axs[0,j]
        for label,y in [('V',means[:,0]),('T',means[:,1]),('P',means[:,2:].mean(axis=1))]: ax.plot(x,y,color=COL[label],label=label if label!='P' else 'P: mean of five RSA scores',lw=2)
        ax.fill_between(x,means[:,2:].min(1),means[:,2:].max(1),color=COL['P'],alpha=.15,label='Five-P range (not CI)')
        ax.axhline(kcal.mean(),ls=':',color='#ad913e',label='Raw kcal reference');ax.axhline(0,lw=.7,color='black',alpha=.4)
        ax.set_title(f'Prompt {p} — '+('primary' if p=='A' else 'robustness'));ax.set_ylabel('Mean participant Spearman rho');ax.legend(fontsize=8)
        ax=axs[1,j]
        for c,label in enumerate(['T − V','T − mean P']):ax.plot(x,delta[p][:,:,c].mean(0),label=label,color=[COL['TV'],COL['TP']][c],lw=2)
        ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Mean paired RSA difference');ax.set_xlabel('Decoder layer');ax.legend(fontsize=9);ax.set_xticks([1,8,16,24,32])
    # Use equal vertical scales across prompts so visual magnitude can be compared.
    for r in range(2):
        bounds=[a.get_ylim() for a in axs[r]];lo=min(b[0] for b in bounds);hi=max(b[1] for b in bounds)
        for a in axs[r]:a.set_ylim(lo,hi)
    fig.suptitle('Raw vmPFC RSA | 29 participants, 493-food common set\nDescriptive only; fixed participant masks; no inference or acquisition correction',fontsize=13)
    save(fig,'neural_alignment')
    fig,axs=plt.subplots(1,2,figsize=(13,4),sharey=True,layout='constrained')
    for ax,p in zip(axs,['A','B']):
        a=behavior[p]
        for label,y in [('V',a[:,0]),('T',a[:,1]),('P',a[:,2:].mean(1))]:ax.plot(x,y,color=COL[label],label=label if label!='P' else 'Mean P',lw=2)
        ax.fill_between(x,a[:,2:].min(1),a[:,2:].max(1),color=COL['P'],alpha=.15,label='Five-P range (not CI)')
        ax.axhline(ref['kcal_to_wanting_rho'],ls=':',color='#ad913e',label='Raw kcal / wanting');ax.axhline(0,color='black',lw=.7)
        ax.set_title(f'Prompt {p}');ax.set_xlabel('Decoder layer');ax.set_xticks([1,8,16,24,32]);ax.legend(fontsize=8)
    axs[0].set_ylabel('Spearman rho vs fixed mean-wanting RDM');fig.suptitle('Behavioral reference | no participant bootstrap or inference')
    save(fig,'behavioral_reference')
    fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for c,name in enumerate(['T − V','T − mean P']):
        bound=max(float(np.max(np.abs(delta[p][:,:,c]))) for p in ['A','B'])
        for j,p in enumerate(['A','B']):
            im=axs[c,j].imshow(delta[p][:,:,c],aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound,origin='upper')
            axs[c,j].set_title(f'{p}: {name}');axs[c,j].set_yticks(range(29),[s.replace('sub-','') for s in subjects],fontsize=7);axs[c,j].set_xticks([0,7,15,23,31],[1,8,16,24,32]);axs[c,j].set_xlabel('Decoder layer');axs[c,j].set_ylabel('Participant');fig.colorbar(im,ax=axs[c,j],label='Paired rho difference',shrink=.8)
    fig.suptitle('Individual participants | all layers retained; no significance threshold')
    save(fig,'participant_contrasts')
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained');xx=np.arange(29)
    for j,name in enumerate(['session','run']):
        yy=np.array([float(r[name+'_rho']) for r in q]);axs[j].bar(xx,yy,color=['#58778d','#806a91'][j]);axs[j].axhline(0,color='black',lw=.8);axs[j].set_xticks(xx,[s.replace('sub-','') for s in subjects],rotation=90,fontsize=7);axs[j].set_title(f'Brain RDM vs {name} identity');axs[j].set_xlabel('Participant');axs[j].set_ylabel('Spearman rho')
    fig.suptitle('Acquisition QC | run identity = session + run; raw RDM unchanged')
    save(fig,'acquisition_QC')
    summary={'interpretation':'descriptive_post_MSE_follow_up_no_inference','primary_prompt':'A','layer_average_is_descriptive_not_new_primary_test':True,'prompts':{},'references':{'kcal_brain_mean':float(kcal.mean()),'kcal_wanting':ref['kcal_to_wanting_rho']},'QC':{}}
    for p in ['A','B']:
        means=neural[p].mean(0);dm=delta[p].mean(0);b=behavior[p]
        summary['prompts'][p]={'neural_layer_average':{'V':float(means[:,0].mean()),'T':float(means[:,1].mean()),'P':float(means[:,2:].mean())},'contrasts':{},'behavior_layer_average':{'V':float(b[:,0].mean()),'T':float(b[:,1].mean()),'P':float(b[:,2:].mean())}}
        for c,name in enumerate(['T-V','T-mean_P']):summary['prompts'][p]['contrasts'][name]={'mean_across_layers':float(dm[:,c].mean()),'layer_min':float(dm[:,c].min()),'layer_max':float(dm[:,c].max()),'positive_group_mean_layers':int((dm[:,c]>0).sum()),'participants_positive_layer_average':int((delta[p][:,:,c].mean(1)>0).sum()),'participant_layer_average_range':[float(delta[p][:,:,c].mean(1).min()),float(delta[p][:,:,c].mean(1).max())]}
    for name in ['session','run']:
        yy=np.array([float(r[name+'_rho']) for r in q]);summary['QC'][name]={'mean':float(yy.mean()),'median':float(np.median(yy)),'min':float(yy.min()),'max':float(yy.max()),'positive_participants':int((yy>0).sum())}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Raw trial-β RSA首轮描述性结果','', '日期：2026-09-17。v06.1；已见MSE结果后的follow-up。先A再B，未运行显著性检验、置信带、bootstrap、中心化或pair筛除。','', '## 数据与方法','', '- 29人、493张共同图片、13,765条脑trial；固定vmPFC支持257–272个体素。保留既有两条脑剔除，无新增筛人/剔除。', '- 346个实际SPM来源与历史哈希一致，输入为desc-preproc BOLD；当前LSA脚本与归档相同，无额外空间平滑。该检查不等于所有上游BOLD字节的取证。', '- 提取pattern全部finite且非恒定；ROI均值与旧账本逐条完全一致。raw β直接进入1−Pearson RDM，无额外预处理。', '- 每人三个session合为大RDM；所有条件、层、A/B共用该人的有效pair。模型/脑RDM下三角Spearman；P先逐套算score再平均，参与者等权。','', '## 神经结果','', '**以下跨32层均值只是量级摘要，不是新设主检验；完整曲线及逐人结果均保留。**','', '| 提示 | V | T | 平均P | T−V | T−平均P |','|---|---:|---:|---:|---:|---:|']
    for p in ['A','B']:
        s=summary['prompts'][p];v=s['neural_layer_average'];d=s['contrasts'];lines.append(f"| {p} | {v['V']:.6f} | {v['T']:.6f} | {v['P']:.6f} | {d['T-V']['mean_across_layers']:+.6f} | {d['T-mean_P']['mean_across_layers']:+.6f} |")
    lines+=['','![完整神经曲线](neural_alignment.png)','','| 提示/对比 | 32层群体差值范围 | 群体均值为正的层数 | 跨层平均差值为正的人数 |','|---|---:|---:|---:|']
    for p in ['A','B']:
        for name,d in summary['prompts'][p]['contrasts'].items():lines.append(f"| {p} / {name} | {d['layer_min']:+.6f} 至 {d['layer_max']:+.6f} | {d['positive_group_mean_layers']}/32 | {d['participants_positive_layer_average']}/29 |")
    lines+=['','正向层数不是独立重复次数，人数计数不是显著性检验。P阴影是五套错配实现的群体分数范围，不是CI。','','![逐人差值](participant_contrasts.png)','','## 行为与营养参考','','| 提示 | Wanting–V | Wanting–T | Wanting–平均P |','|---|---:|---:|---:|']
    for p in ['A','B']:
        b=summary['prompts'][p]['behavior_layer_average'];lines.append(f"| {p} | {b['V']:.6f} | {b['T']:.6f} | {b['P']:.6f} |")
    lines+=['',f"以上为跨层描述性均值。Raw kcal对脑RSA的29人均值为{float(kcal.mean()):.6f}，对wanting为{ref['kcal_to_wanting_rho']:.6f}。Wanting使用单张493图群体均值RDM，不构造29份伪独立样本。",'', '![行为参照](behavioral_reference.png)','','## 采集结构QC','','| Identity RDM | 平均ρ | 中位数 | 范围 | 正向人数 |','|---|---:|---:|---:|---:|']
    for name,a in summary['QC'].items():lines.append(f"| {name} | {a['mean']:.6f} | {a['median']:.6f} | {a['min']:.6f}–{a['max']:.6f} | {a['positive_participants']}/29 |")
    lines+=['','Run使用session＋run联合身份。以上表明raw脑RDM含可见的采集结构；不能将其忽略，也不能仅凭这些相关认定模型差异全部来自扫描噪声。本轮按约定只报告，没有触发去均值、白化或pair删除。','','![扫描结构QC](acquisition_QC.png)','','## 验证与解释边界','',f"- A/B每人×32层×7条件共{audit['neural_score_count']:,}个脑RSA分数全部finite。A流程正确性通过后才计算B，未按效果决定。",f"- 与独立SciPy直接求秩相关核对，每提示{audit['validation']['A']['direct_reference_comparisons']}处；最大差A={audit['validation']['A']['max_scipy_difference']:.3g}、B={audit['validation']['B']['max_scipy_difference']:.3g}。",'- 模型14组hidden及图片索引重新计算哈希，与既有矩阵审计一致。距离、平局、共同缺失mask、ID重排和P聚合数值验证通过。','- 输入来源、pattern有限值与数值正确性不等于脑测量可靠性或总体效应已经验证。不能由本轮曲线宣称显著改善、条件等效、新食品泛化或模型/人的营养因果机制。','- 这套脑数据没有每人同图独立重复；扫描结构及单次估计噪声仍限制解释。若后续考虑敏感性或正式推断，须记录当前曲线已可见，不改写raw主结果。','', '## 文件入口','',f'- [全部逐人分数](../{source.name}/neural_scores.csv)、[配对差值](../{source.name}/contrasts.csv)、[逐层群体与行为曲线](../{source.name}/group_curves.csv)。',f'- [采集QC明细](../{source.name}/acquisition_QC.csv)、[RSA计算审计](../{source.name}/audit.json)。','- 本目录提供PNG及PDF科学图、summary.json；不包含统计推断或敏感性结果。']
    (out/'结果摘要.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,out/'script_snapshot.py')
    ra={'status':'complete','source_audit':str(source/'audit.json'),'source_audit_sha256':sha(source/'audit.json'),'script_sha256':sha(__file__),'matplotlib':matplotlib.__version__,'python':platform.python_version(),'figures':4,'p_values':False,'confidence_intervals':False,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'report_audit.json').write_text(json.dumps(ra,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
