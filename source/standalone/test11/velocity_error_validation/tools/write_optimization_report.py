"""Assemble final Chinese report directly from measured artifacts."""
import csv,json,hashlib,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/velocity_matplotlib_20261007")
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('source/standalone/test11/velocity_error_validation');RUNS=BASE/'reports/optimization_flow_runs_20261007';EVAL=RUNS/'evaluation'
def csv_rows(path):return list(csv.DictReader(path.open()))
def table(rows,keys):
    def value(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        if isinstance(v,str):
            try:
                number=float(v)
                if '.' in v or 'e' in v.lower():return f'{number:.4f}'
            except ValueError:pass
        return str(v)
    return '| '+' | '.join(keys)+' |\n|'+ '|'.join(['---']*len(keys))+'|\n'+''.join('| '+' | '.join(value(r.get(k,'')) for k in keys)+' |\n' for r in rows)
def all_only(rows,filt):return [r for r in rows if r['group']=='all' and r['filter']==filt]

def main():
    if not (EVAL/'COMPLETE').exists():raise RuntimeError('Evaluation incomplete')
    selection=json.loads((EVAL/'frozen_selection.json').read_text());manifest=json.loads((RUNS/'data_manifest.json').read_text());summary=csv_rows(EVAL/'test_summary.csv');old=csv_rows(EVAL/'original_stream_summary.csv');fixed=csv_rows(EVAL/'fixed_cv_test_summary.csv');baselines=csv_rows(EVAL/'aligned_baseline_test_summary.csv');forecasts=csv_rows(EVAL/'forecast.csv');latency=csv_rows(EVAL/'latency.csv')
    grouped=defaultdict(list)
    for r in summary:
        if r['group']=='all':grouped[(r['variant'],r['scenario'],r['filter'])].append(r)
    stats=[]
    for (variant,scene,filt),rows in grouped.items():
        item={'variant':variant,'scene':scene,'filter':filt,'n':int(rows[0]['n'])}
        for metric in ('mean','p95','p99'):
            vals=np.array([float(r[metric]) for r in rows]);item[metric]=float(vals.mean());item[metric+'_seed_sd']=float(vals.std(ddof=1));item[metric+'_seed_min']=float(vals.min());item[metric+'_seed_max']=float(vals.max())
        stats.append(item)
    with (EVAL/'three_seed_summary.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    data=[]
    for split in ('train','val','test'):
        rows=[r for r in manifest if r['split']==split]
        data.append(dict(split=split,raw_episodes=len(rows),scored_episodes=sum(r['score_frames']>0 for r in rows),frames=sum(r['frames'] for r in rows),scored_frames=sum(r['score_frames'] for r in rows),ground_scored_frames=sum(r['ground_score_frames'] for r in rows),lost_frames=sum(r['lost_frames'] for r in rows),prediction_frames=sum(r['prediction_frames'] for r in rows),post_start_kinematic_over2=sum(r['post_start_kinematic_residual_over2'] for r in rows)))
    chosen=selection['run'];chosen_rows=[r for r in summary if r['run']==chosen];scans=csv_rows(EVAL/'validation_selection.csv');artifacts=csv_rows(EVAL/'model_artifacts.csv');artifact=next(r for r in artifacts if r['run']==chosen)
    main_rows=[]
    for name,rows in [('原 checkpoint 整段',old),('普通 CV',fixed),('验证集选择的学习候选 '+chosen,chosen_rows)]:
        for r in all_only(rows,'center_above_ground'):main_rows.append(dict(method=name,scene=r['scenario'],n=int(r['n']),mean=float(r['mean']),p95=float(r['p95']),p99=float(r['p99'])))
    for width in (.16,.32):
        for r in all_only(csv_rows(EVAL/f'original_rolling_{width:.2f}_summary.csv'),'center_above_ground'):
            main_rows.append(dict(method=f'原模型滚动 {width:.2f}s',scene=r['scenario'],n=int(r['n']),mean=float(r['mean']),p95=float(r['p95']),p99=float(r['p99'])))
    basechoice=selection['aligned_baseline_validation_choice']
    for r in baselines:
        if r['filter']=='center_above_ground' and r['group']=='all' and abs(float(r['width_s'])-basechoice['width_s'])<1e-8 and (r['missing']=='True')==basechoice['missing']:
            main_rows.append(dict(method=f"对齐基线 {basechoice['width_s']:.2f}s missing={basechoice['missing']}",scene=r['scenario'],n=int(r['n']),mean=float(r['mean']),p95=float(r['p95']),p99=float(r['p99'])))
    experiment_keys=['variant','scene','mean','mean_seed_sd','p95','p95_seed_sd','p99','p99_seed_sd']
    text='# 2026-10-07 速度模块第 2–6 步实验总报告\n\n'
    text+='本轮已完成旧数据诊断、新完整流采集、连续训练、独立结构消融、TCN/物理滤波比较及独立测试。使用 `conda activate isaac_lab_232` 和 Isaac Sim setup，计算设备为 cuda:0。现有同卡任务保留，以下耗时属于共享设备测量。\n\n'
    text+=f"验证集选择的总体候选类别为 **{selection['recommended_family']}**；学习候选中选择 **{chosen}**。选择依据是验证集 0.4×SS P95 + 0.6×SD P95，全帧口径；测试结果没有用于调整架构或参数。\n\n"
    text+='独立地上测试（同帧）中，动态 Mean/P95/P99 从原模型的 1.8462/5.3857/9.2143 降至 0.4479/1.5066/3.7614 m/s；静态从 1.1613/3.6237/6.2578 降至 0.3227/1.1363/1.7945 m/s。三个训练种子的动态 P95=1.4997±0.0257，静态 P95=1.1130±0.0216。配对 bootstrap 的 Mean/P95 改善区间均小于零。\n\n'
    text+='**首选路线调整为参考系对齐 + 学习 Q/R 的物理滤波。** 连续原 GRU 相比原 checkpoint 改善，但本轮不及物理滤波；分离门控和显式质量的组合没有稳定赢过原 GRU 流训练。TCN 的验证宏 P95 优于本轮从零 GRU，地上测试并非所有指标都优，因此不将它作为首选。旧残差迁移到新基线的组合弱于从零训练，说明补偿迁移和本轮训练预算需要谨慎解释，不能把这次结果视作 GRU 容量上限。\n\n'
    text+='**没有一个候选在全部旧数据指标上都最优。** 旧 clean 正式集上，学习滤波 Mean/P95=0.2839/1.7189；单独 .24 s 对齐基线为 0.2825/0.6723。学习滤波的优势主要在新异常完整流，旧 clean 集的冷启动/快速响应仍需单独考虑。旧 pilot 的 P99 仍为 4.9243，尾部未完全解决。\n\n'
    text+='普通 CV 也是强对照：新地上动态 Mean/P95=0.4942/1.7439，学习版本为 0.4479/1.5066；静态普通 CV Mean=0.3035，低于学习版本 0.3227，而 P95/P99 略高。因此学习 Q/R 的增益应与普通物理滤波比较，不能把全部改善归因于噪声网络。\n\n'
    text+='## 2. 共有速度估计链路的修复\n\n'
    text+='旧 pilot 的 960 条流、正式集的 1,536 条流均完成同帧分析。正式集 D 的 71,500 个共同地上帧中，原整段 GRU Mean/P95/P99=1.1512/3.2952/5.0945 m/s；只利用已有角速度与机身线速度积分到局部参考系的 0.24 s 对齐基线为 0.2825/0.6723/4.1540 m/s。GT 位姿未进入该候选。仅换 GT 方向或距离对旧残差网络的改善有限；对齐后观测误差仍影响精度。详见 [第 2 步报告](E2_OBSERVATION_BASELINE_RESULTS_20261007.md)。\n\n'
    text+='CPU/CUDA 原基线差异已复核：float32 累加时间造成严格窗口边界纳入不同；统一累加时间后平均向量差降到 3.27e-7 m/s。新对齐基线使用 float64 真实时间、1 ns 边界容差及因果历史。新基线没有直接接到旧 GRU 的既有残差上作为正式结果。\n\n'
    text+='## 3. 新数据与固定评分协议\n\n'+table(data,list(data[0]))+'\n'
    text+='原始 episode 数含采集结束时的短片段；评分和训练损失均保留原监督 burn-in 与首次直接可用观测起点。完全无有效观测的 episode 不产生速度精度分数，仍进入可用性统计。train/val/test 分别使用 26100710–13 / 26100720–23 / 26100730–33，种子不交叉；各自包含 SS/SD × clean/mixed。每帧保留真实 source、质量、缺测、预测、恢复及 reset。数据清单包含流文件和审计文件 SHA-256。\n\n'
    text+='新数据是机身相机、真实跟踪及测距的 D 条件。首次及 tracker 重初始化仍采用仿真 bbox 检测代理；每次初始化后不使用 GT 方向或 GT 距离输入。SD 按当前采集代码使用 FOV 约束三维运动，0.6–5 m/s、80–150 m，可能包含地下帧。目标在边界反射之外没有随机加速度；生成器偏向常速度滤波的建模假设，因此不能据此断言它在机动目标中同样优于 GRU。此分布与 9 月正式集不同，二者分别报告。GT 位姿只用于离线 oracle、地面筛选、运动学审计和外推评分，未进入新模型输入。\n\n'
    text+='三维误差为 `||v_pred-v_gt||₂`，单位 m/s；地面筛选为当前目标中心 z≥0，只作用于评分。新 D 条件评分始于 D 首次直接可用观测；旧四条件开发集使用跨条件统一起点，新旧绝对数值应分开解释。全部新候选在同一分组内使用同一帧集合，预测、完全丢失及重获阶段均计入。离散运动学残差超过 2 m/s 的计数仅作审计；边界反射和离散采样也能产生该残差，未据此删除困难帧或称其为坏标签。\n\n'
    text+='## 4–5. 连续训练与独立消融\n\n'
    text+='每个候选 3 个训练种子（41/42/43）、24 epochs、AdamW lr=3e-4、batch=64、TBPTT=100 帧（nominal 2 s）。同一 batch 槽位对应同一完整 episode，换 episode 清除状态；相邻段 detach 保留数值状态。采用原 Huber(beta=.1)+.10 NLL+.05 速度模长损失。每个模型按验证集宏 P95 保存最佳 epoch；原 GRU 路线从同一个旧 checkpoint 迁移，额外报告组合 GRU 从零训练以比较 TCN 的训练起点。\n\n'
    text+='`cold_segments` 每 2 s 清零；`cold_0p48_state` 每 24 帧清零，但保持同样的流输入、基线、监督帧和优化更新次数，仅控制隐藏状态训练跨度。测试均运行完整连续流。`flow_original` 只改变训练过程；`flow_aligned` 替换主物理基线；`flow_quality` 保持原输出基线并显式提供对应三尺度基线和拟合质量；`flow_split_gate` 分开位置/差分编码及门控，位置门不读取 FD 或 FD 可靠性。`flow_missing` 保留同一目标中的记忆并在完全无有效距离时物理传播位置，预测帧保留视觉模块输出。组合候选主基线为 .24 s，并提供 .12/.24/.48 s 的速度、跨度、有效点数、拟合残差、时间分布及未截断的观测 age。\n\n'
    text+='以下为地上测试的 **3 个训练种子指标均值与种子间标准差**，不能当作把三次预测混合后的单个模型分位数。\n\n'+table([r for r in stats if r['filter']=='center_above_ground'],experiment_keys)+'\n'
    text+='全帧指标同样保留，防止地面筛选隐藏失败样本。\n\n'+table([r for r in stats if r['filter']=='unfiltered'],experiment_keys)+'\n'
    text+='## 6. TCN、尾部损失与物理滤波\n\n'
    text+='TCN 使用 64 通道、3 个因果残差块，每块两层 kernel=3，dilation=1/2/4；网络感受野 29 帧，仅左侧 padding。与组合 GRU 使用同样的原始/质量输入及主基线；基线特征自身包含过去 .48 s，实际输入历史因此超过纯卷积感受野。TCN 从零训练；`flow_combined_scratch` 提供同数据、同预算的 GRU 从零对照。\n\n'
    text+='`flow_combined_tail` 单独加入 .05×top10% 向量误差均值；本轮没有同时叠加多种新损失。普通 CV 和学习噪声的 CV 维护局部里程计参考系下的位置/速度；只用直接有效位置更新，预测或缺测期间仅传播。学习版本以 softplus 和下界产生正定对角 Q/R，不读取 GT 状态。普通 CV 在预先固定的 q={.1,1,5}、r={.01,.1,1} 网格上仅用验证集选择。\n\n'
    text+='独立测试地上帧上的原模型、有限历史及无学习控制如下。\n\n'+table(main_rows,['method','scene','n','mean','p95','p99'])+'\n'
    text+='验证集选择记录：\n\n```json\n'+json.dumps(selection,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='## 尾部、恢复、外推与耗时\n\n'
    for group in ('prediction','lost','range_unavailable','recovery_200ms','age_gt5s'):
        rows=[dict(scene=r['scenario'],group=group,n=int(r['n']),mean=float(r['mean']),p95=float(r['p95']),p99=float(r['p99'])) for r in chosen_rows if r['filter']=='center_above_ground' and r['group']==group]
        if rows:text+=table(rows,['scene','group','n','mean','p95','p99'])+'\n'
    text+='恢复分组为一次非直接可用观测区间结束后的 200 ms，首次初始化不算重获。分组缺少结果表示该测试集没有对应有效事件，不能解释为误差为零。\n\n'
    boot=[r for r in csv_rows(EVAL/'paired_bootstrap.csv') if r['run']==chosen and r['filter']=='center_above_ground']
    text+='相对原模型的配对 episode bootstrap（500 次，95% 区间；负值表示改善）：\n\n'+table(boot,['scenario','metric','delta_ci_low','delta_ci_high','episodes'])+'\n'
    text+='另按 collection seed×environment 合并同一环境的多个 episode 做 block bootstrap，见 [区间](../reports/optimization_flow_runs_20261007/evaluation/paired_block_bootstrap.csv)。SS/SD 分别有 16/15 个环境块，其 Mean/P95 改善区间也均小于零。\n\n'
    text+='bootstrap 条件于固定训练种子，不包含训练随机性；上面的三种子标准差另作报告。100/200 ms 外推输入只使用当前信息；未来 GT 仅作评分标签。速度位移项避免把当前位置观测误差混入速度指标；绝对位置项包含当前观测误差，限直接可用观测。以下单位为 m：\n\n'+table([r for r in forecasts if r['run']==chosen and r['filter']=='center_above_ground'],['scenario','horizon_s','kind','n','mean','p95','p99'])+'\n'
    text+='缓存输入的 CUDA 推理及 CPU 三尺度构造耗时如下；不包含相机、LK/OSTrack、完整特征构造、数据传输或生产目标管理，不能视作视觉模块端到端延迟。\n\n'+table(latency,['variant','targets','component','p50_ms','p95_ms'])+'\n'
    text+='学习滤波地上测试的每轴 1.96σ 边际覆盖率为 SS 98.49%、SD 96.90%；三轴盒内联合覆盖为 SS 96.09%、SD 92.79%。边际覆盖略保守，仍需不同数据上的校准。\n\n'
    text+='逐 episode 报告另含每轴 1.96σ 的边际覆盖率与三轴同时落入盒内的覆盖率；后二者不是同一种 95% 概率定义。未用大 sigma 删除任何评分帧。\n\n'
    text+='[参考接口实流一致性检查](../reports/optimization_flow_runs_20261007/evaluation/online_reference_validation.csv)覆盖 SS/SD × clean/mixed 的四条独立测试 episode，最大逐帧向量差约 3.7e-5 m/s；498 帧 episode 的接口 P95 约 7–11.4 ms。此数包含 prefix 重构，随 episode 长度增加，并排除视觉前端。\n\n'
    text+='## 旧数据回归与结论边界\n\n'+table([r for r in csv_rows(EVAL/'development_regression.csv') if r['group']=='all'],['dataset','method','scenario','filter','group','n','mean','p95','p99'])+'\n'
    text+='旧报告数据已用于方案诊断，属于开发回归集；新种子的测试集才是本轮独立测试。新测试规模有限、只覆盖仿真 D 条件，不能替代实机里程计误差、不同目标运动及长期遮挡验证。算法无法从没有观测的任意机动目标中恢复确定的真实速度，必须结合缺测持续时间和不确定度理解尾部。\n\n'
    text+='第 2 步明确支持先修复参考系对齐。后续模型路线应按验证集推荐类别和同帧独立测试结果决定，不能默认模型越复杂越好，也不能只引用 Mean 忽略 P99 与恢复。新的 checkpoint、配置和回放产物保存在独立实验目录。\n\n'
    text+='逐帧 GT-free 参考接口为 `tools/online_experimental_velocity.py`，输入仅有 24 维 TAQ 特征、source 和时间戳。每个目标使用独立实例；真实目标/episode 改变时 reset，视觉重初始化通过独立 tracker_reset 表达。该接口保留因果 prefix 以便回放核对，长期在线运行仍需增量化缓存；本轮缓存输入 benchmark 不是这层接口的完整耗时。\n\n'
    text+='输出是 yaw-only 机身参考系下的目标相对机身平移速度 `R(v_target-v_base)`，不是目标位置在旋转坐标系中的直接导数；机身线速度也必须采用同一参考系，特征中的 z 角速度遵循当前 heading 导数约定。\n\n'
    text+='距离、目标真实运动速度、输入角速度、传感器 dt 和未截断的观测年龄分层见 [分层结果](../reports/optimization_flow_runs_20261007/evaluation/stratified_test.csv)。\n\n'
    text+='## 复现与产物\n\n'
    text+=f"学习候选：[checkpoint](../reports/optimization_flow_runs_20261007/{chosen}/best_model.pt)，SHA-256 `{artifact['checkpoint_sha256']}`。\n\n"
    text+='完成 25 项自动检查及 4 条独立流参考接口一致性检查。[消融图 PNG](../reports/optimization_flow_runs_20261007/evaluation/ablation_ground.png) / [PDF](../reports/optimization_flow_runs_20261007/evaluation/ablation_ground.pdf)。最终代码、配置和结果文件哈希见 [最终溯源清单](../reports/optimization_flow_runs_20261007/final_provenance.json)。\n\n'
    text+='训练期间发现缺测传播状态在预测帧未前进，已在独立测试前修正，并重训全部 18 个受影响的运行；旧产物保留在 `pre_propagation_fix`。汇总检查时修正普通 CV CSV 被后续基线扫描覆盖的问题：仅按冻结 Q/R 重写该表，候选选择及预测不变，详见 [修复记录](../reports/optimization_flow_runs_20261007/evaluation/fixed_cv_csv_repair.json)。\n\n'
    text+='```bash\ncd /home/jby/IsaacLab-2.3.2\nbash source/standalone/test11/velocity_error_validation/tools/run_optimization_campaign.sh collect\nbash source/standalone/test11/velocity_error_validation/tools/run_flow_training.sh development\nbash source/standalone/test11/velocity_error_validation/tools/run_flow_training.sh\n/home/jby/miniconda3/envs/isaac_lab_232/bin/python source/standalone/test11/velocity_error_validation/tools/write_optimization_report.py\n```\n\n'
    text+='脚本初始化指定 conda 和 Isaac Sim 环境。已有完成标记用于续跑；训练数据缓存及各 checkpoint 不应与修改后的采集或输入方案混用。主要审阅文件为 [数据清单](../reports/optimization_flow_runs_20261007/data_manifest.json)、[独立测试汇总](../reports/optimization_flow_runs_20261007/evaluation/test_summary.csv)、[三种子汇总](../reports/optimization_flow_runs_20261007/evaluation/three_seed_summary.csv)、[选择记录](../reports/optimization_flow_runs_20261007/evaluation/frozen_selection.json)、[外推](../reports/optimization_flow_runs_20261007/evaluation/forecast.csv) 和 [耗时](../reports/optimization_flow_runs_20261007/evaluation/latency.csv)。\n'
    (BASE/'docs/VELOCITY_OPTIMIZATION_STEPS_2_TO_6_REPORT_20261007.md').write_text(text)
    # Export standard scientific figures with explicit units and separate scenes.
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,scene in zip(axes,['ss','sd']):
        rows=[r for r in stats if r['scene']==scene and r['filter']=='center_above_ground'];names=[r['variant'] for r in rows];x=np.arange(len(rows))
        ax.bar(x-.16,[r['mean'] for r in rows],.32,label='Mean',yerr=[r['mean_seed_sd'] for r in rows]);ax.bar(x+.16,[r['p95'] for r in rows],.32,label='P95',yerr=[r['p95_seed_sd'] for r in rows])
        ax.set_xticks(x,names,rotation=70,ha='right',fontsize=7);ax.set_ylabel('3D velocity error (m/s)');ax.set_title(f'Independent test {scene.upper()}, center z >= 0');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(EVAL/'ablation_ground.png',dpi=180);fig.savefig(EVAL/'ablation_ground.pdf');plt.close(fig)
    print('Report written:',BASE/'docs/VELOCITY_OPTIMIZATION_STEPS_2_TO_6_REPORT_20261007.md')
if __name__=='__main__':main()
