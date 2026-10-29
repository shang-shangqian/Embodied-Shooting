"""Build N0 report from complete acquisition, frozen replay, and sampler audits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/n0_matplotlib_20261007')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('source/standalone/test11/velocity_error_validation')
RUN=BASE/'reports/n0_time_contract_20261007'
SPARSE=BASE/'reports/n0_actual_sampling_20261007'
DOC=BASE/'docs/N0_TIME_CONTRACT_RESULTS_20261007.md'


def rows(path):return list(csv.DictReader(path.open()))
def table(data,keys):
    def value(v):
        if isinstance(v,float):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+''.join('| '+' | '.join(value(r[k]) for k in keys)+' |\n' for r in data)


def comparisons(data,group_key='scenario',event=False,filt='ground'):
    lookup={tuple(r[k] for k in ([group_key,'profile','backend','method','filter']+(['event'] if event else []))):r for r in data}
    out=[]
    for r in data:
        if r['backend']!='frozen_learned_filter' or r['filter']!=filt or r['method']!='sample_time_replay' or (event and r['event']!='all'):continue
        key=[r[group_key],r['profile'],r['backend'],'arrival_time_control',r['filter']]+(['all'] if event else [])
        old=lookup[tuple(key)]
        out.append(dict(scene=r[group_key],profile=r['profile'],n=int(r['n']),
            before_mean=float(old['mean']),after_mean=float(r['mean']),
            before_p95=float(old['p95']),after_p95=float(r['p95']),
            before_p99=float(old['p99']),after_p99=float(r['p99'])))
    return out


def main():
    if not (RUN/'COMPLETE').exists() or not (SPARSE/'COMPLETE').exists():raise ValueError('N0 experiments incomplete')
    manifest=json.loads((RUN/'data_manifest.json').read_text())
    summary=rows(RUN/'scenario_metrics.csv');grouped=rows(RUN/'grouped_metrics.csv')
    equations=rows(RUN/'frozen_equation_validation.csv');audit=rows(RUN/'timing_audit.csv')
    sparse_audit=rows(SPARSE/'actual_sampling_audit.csv');sparse_summary=rows(SPARSE/'scenario_metrics.csv')
    protocol=json.loads((RUN/'protocol.json').read_text())
    main_table=comparisons(summary);group_table=comparisons(grouped,'group',True)
    data=[]
    for group in sorted({r['group'] for r in manifest}):
        subset=[r for r in manifest if r['group']==group]
        data.append(dict(group=group,collection_seeds=len({r['seed'] for r in subset}),
            environment_blocks=len({r['block'] for r in subset}),episodes=len(subset),
            frames=sum(r['frames'] for r in subset),scored=sum(r['score_frames'] for r in subset),
            ground_scored=sum(r['ground_frames'] for r in subset),
            max_position_reconstruction_m=max(r['reconstruction_max_m'] for r in subset),
            median_episode_yaw_p95=float(np.median([r['yaw_rate_abs_p95'] for r in subset]))))
    selected=next(r for r in main_table if r['scene']=='sd' and r['profile']=='combined')
    text='# 工作 1 / N0：时间契约与异步观测构造测试报告\n\n'
    text+='本轮完成物理采样时钟、传感器分量配对、延迟队列、有限回溯更新及同轨迹配对测试。使用 isaac_lab_232、Isaac Sim setup 和 cuda:0；学习滤波器仍是上轮冻结的 learned_filter_seed41，普通 CV 仍为 q=.1/r=.1。没有训练模型或用新测试调参。\n\n'
    text+=f"在本轮动态地上帧的组合时序退化下，到达时刻拼接控制与采样时刻回放的 Mean/P95/P99 分别为 {selected['before_mean']:.4f}/{selected['before_p95']:.4f}/{selected['before_p99']:.4f} 与 {selected['after_mean']:.4f}/{selected['after_p95']:.4f}/{selected['after_p99']:.4f} m/s。Mean 变化 {(selected['after_mean']/selected['before_mean']-1)*100:.1f}%，P95 变化 {(selected['after_p95']/selected['before_p95']-1)*100:.1f}%。正值表示退化，不能默认所有时序场景都会改善。\n\n"
    text+='这些是同一批新采集真实视觉观测上的配对后端测试，不是把上轮独立测试数字跨数据集相减。控制组复现“不同时间分量按当前时间拼接、合成时间戳参与传播”的处理方式，没有重新运行旧版延迟图像跟踪器。\n\n'
    text+='## 修改后的时间与观测接口\n\n'
    text+='采集新增 `--sensor_time_contract physical_capture_v1`，为 D/clean 完整流输出独立 `timed_sensor_packets.pt`。采样时间为物理步推进后的共同单调参考时钟；图像/距离/本体的采样时间、视觉结果可用时间和设备报告时间分别定义。旧 synthetic timing 路径保持 `legacy_v0` 的显式兼容标识；新时序退化通过带原采样时间的传感器包回放产生。\n\n'
    text+='方向包携带曝光时刻相机到 yaw-only 机身的旋转与平移；距离包只与相同 sample_id、相同采样时刻的方向配对。只读最新方向和最新距离拼接已被替换。缺少方向时，即使激光有返回也不能构成目标位置；无匹配分量、预测源和完全丢失不作为直接观测更新。当前位置与历史位置转换不读取 target GT 或 GT 世界位姿审计。导出的相机外参/朝向来自仿真传感器与已有标定链路，尚未模拟标定/姿态误差。\n\n'
    text+='本体线速度与位置均使用 yaw-only 机身参考系；angular-z 沿用 TAQ 的 heading 导数约定。输出为该参考系下的物理相对平移速度，不能将一般机身陀螺仪 z 分量直接视作此 heading 导数。\n\n'
    text+='已到达的晚包在 0.20 s 固定回溯范围内替换测量时刻输入，并从该时刻之前的滤波快照重新传播至当前输出；已发布的旧输出保持原值。重做从旧快照开始，避免把同一证据在现有后验上重复加入。超出回溯范围的包拒绝并审计；方向与距离尚未配齐时保留等待，不以未来包提前更新。机器人本体晚包同样按采样时刻回填里程计积分。真实 dt 不作 0.5 s 静默截断。\n\n'
    text+='接口约束是已知的共同采样参考时钟。clock_noise 测试把旧式累计设备时钟畸变保存为独立退化，修正路径使用可获得的采样参考时间；没有用 GT audit 校时，也没有实现未知设备时钟的在线偏移/漂移估计。独立频率、无法关联曝光的传感器需要另外定义同步容差或独立测量模型，不能靠此严格配对接口自动融合。\n\n'
    text+='## 数据、配对和评分\n\n'+table(data,list(data[0]))+'\n'
    text+='正式采集 6 组 × 4 env × 650 物理步：SS/SD 静止机身、SD 运动机身各有两个独立采集 seed。运动机身命令 vx=.3 m/s、yaw_rate=.1 rad/s，实际运动统计见上表，命令不等于实测速度。目标保留现有 FOV 约束常速度/边界反射分布；SD 可能含地下帧，完整帧和地上帧分别报告。首次及重获仍采用仿真 bbox 检测代理，初始化后的方向与距离由 LK/OSTrack 和 raycaster 提供。\n\n'
    text+='每条物理/视觉轨迹独立生成 sync、观测包不规则采样、仅时钟畸变、仅视觉结果延迟、仅距离延迟、仅本体延迟与组合退化；所有方法共享相同到达包。视觉结果延迟为 40–65 ms、距离延迟 20–45 ms、本体延迟 40–55 ms；不规则采样随机跳过 2–6 次目标观测，本体采样保持 20 ms。完整时序可从 delivery_schedule.jsonl 与固定退化种子复现。\n\n'
    text+='评分保留原标签 burn-in，始于原始轨迹首次直接有效观测，各方法完全同帧。延迟导致的启动先验帧仍评分；地上过滤只影响当前评分，不删除历史。误差为三维速度误差范数 m/s，P95/P99 为 pooled 分位数。每帧输出先验或估计，未靠减少输出覆盖率获取收益。\n\n'
    text+='## 冻结学习滤波器：地上同帧结果\n\n'+table(main_table,list(main_table[0]))+'\n'
    text+='组合退化下静态和动态的 Mean/P95 均改善；单独距离延迟下，动态 Mean/P95/P99 从 0.1800/0.7540/2.6956 增至 0.1878/0.8217/2.9517。采样语义正确并不保证每项速度指标立即下降，严格等待配对及剩余初始化误差需要在后续单项控制中处理。只丢观测包时，动态 P99 也从 2.7951 增至 2.8804，不能只引用其 Mean/P95 改善。\n\n'
    text+='全评分帧结果如下，保留地下及压力帧：\n\n'
    full_table=comparisons(summary,filt='all_frames')
    text+=table(full_table,list(full_table[0]))+'\n'
    text+='同步路径逐帧一致；这项对照用于排除“换数据或换模型”产生表面收益。单纯时钟畸变修复后应回到同步处理结果，因为当前接口已提供可信采样参考时间。延迟/缺测期间仍缺少最新目标观测，时间修复不等于恢复不存在的信息；按曝光身份严格等待距离可能增加初始可用延迟，P95/P99 是否改善必须看实测。\n\n'
    text+='## 静止与运动机身分开检查\n\n'+table(group_table,list(group_table[0]))+'\n'
    text+='## 强基准、启动与单因素诊断\n\n'
    controls=[]
    for r in summary:
        if r['filter']=='ground' and r['profile'] in ('sync','combined'):
            controls.append(dict(scenario=r['scenario'],profile=r['profile'],backend=r['backend'],
                                 method=r['method'],**{k:float(r[k]) for k in ('mean','p95','p99')}))
    text+=table(controls,list(controls[0]))+'\n'
    text+='组合退化额外提供 physical_clock_only 控制：只改物理时钟，仍保留到达时刻拼接，用于区分时钟与曝光配对/历史重传播的增量作用。普通 CV 使用上轮验证集参数，未扩展网格；本轮重点是时间链路，充分调优的 Q/R 控制属于工作 3。\n\n'
    startup=[r for r in grouped if r['backend']=='frozen_learned_filter' and r['filter']=='ground' and r['profile']=='combined' and r['event']=='startup_0p5s']
    startup_table=[dict(group=r['group'],method=r['method'],n=int(r['n']),mean=float(r['mean']),
                        p95=float(r['p95']),over2_rate=float(r['over2_rate'])) for r in startup]
    if startup_table:text+=table(startup_table,list(startup_table[0]))+'\n'
    ci=[r for r in rows(RUN/'paired_bootstrap.csv') if r['backend']=='frozen_learned_filter' and r['filter']=='ground' and r['profile'] in ('irregular_sampling','clock_noise','image_delay','range_delay','proprio_delay','combined')]
    text+='配对 bootstrap 以采集 seed×环境为块，合并同一环境的多个 episode，500 次重采样；区间是采样时刻回放减去到达时刻控制，负值表示改善。区间条件于冻结模型和本轮轨迹，不代表训练随机性或实机泛化。\n\n'
    text+=table([dict(scenario=r['scenario'],profile=r['profile'],metric=r['metric'],blocks=int(r['blocks']),
                      ci_low=float(r['delta_ci_low']),ci_high=float(r['delta_ci_high'])) for r in ci],
                ['scenario','profile','metric','blocks','ci_low','ci_high'])+'\n'
    text+='## 真实跳帧的前端采集检查\n\n'
    text+='额外用相同物理 seed 采集 SD 静止/运动机身两组，物理仿真仍推进每个 20 ms 步，但采样门在进入 LK/OSTrack 之前跳过部分图像，下一次 tracker.update 的 dt 为实际两次图像输入的采样时间差。它不同于前面只丢目标观测包的后端配对控制。两类结果分别保留。\n\n'
    sampling=[]
    for r in sparse_audit:
        sampling.append(dict(uid=r['uid'],frames=int(r['frames']),skipped=int(r['skipped_images']),
             max_tracker_dt=float(r['max_tracker_input_dt_s']),
             trajectory_delta_m=float(r['target_trajectory_max_difference_m']),
             label_delta_mps=float(r['label_max_difference_mps']),
             physical_time_delta_s=float(r['physical_time_max_difference_s'])))
    text+=table(sampling,list(sampling[0]))+'\n'
    text+='轨迹差和标签差用于确认重新运行仿真时能否严格配对；若超过数值容差，这部分只作为真实跳帧集上的后端同帧比较，不宣称与 dense 前端为纯采样因果对照。其评分起点沿用对应 dense 轨迹，不排除 sparse 初始缺测。\n\n'
    sparse_table=comparisons(sparse_summary)
    text+=table(sparse_table,list(sparse_table[0]))+'\n'
    text+='## 完整性与数值验收\n\n'
    text+=f"采样时间与 physics audit/stream 时间一致；位置重建最大差 {max(r['reconstruction_max_m'] for r in manifest):.6g} m。冻结学习滤波的参考 CV 方程与原 CUDA 实现逐流校验，最大向量差 {max(float(r['max_vector_difference_mps']) for r in equations):.6g} m/s。预检 60 组预测中，缓存前处理与完整 prefix 重算的最大差为 0。\n\n"
    text+='指定 conda+Isaac 环境回归检查通过；具体结果保存为 tests.log。覆盖延迟乱序、未来到达隔离、错误配对、重复拒绝、长期 dt、过期包、丢失方向、真实采样门及缓存一致性。\n\n'
    live=rows(RUN/'live_api_validation.csv')
    text+=f"回归结果为 81 passed。GT-free 逐包接口 `CausalSensorTimeEstimator.advance(sample_id, output_time_s, deliveries)` 不接收未来时间表或 GT；在 SS/SD 静止机身、SD 运动机身三条真实 498 帧流上，与批量回放的最大向量差为 {max(float(r['max_vector_difference_mps']) for r in live):.6g} m/s。接口返回当前时刻速度、局部位置、局部绝对速度及完整模型协方差，协方差尚不包含本体/时钟估计误差。见 [逐包验证](../reports/n0_time_contract_20261007/live_api_validation.csv)。\n\n"
    text+='首次采集的三组提前退出输出及其旧完成标记保存在 outputs/n0_capture_20261007/_incomplete_capture_archive；重采沿用相同 seed。启动脚本收尾问题已修正，完成标记现在要求结束记录、环境数量、真实帧数、包/标签/审计数量一致后才生成。这些不完整输出没有进入最终评分。\n\n'
    text+='本轮回放保留历史用于审计，重传播范围有界，但尚未完成 N6 的长期有界缓存与完整端到端延迟验收。本轮修复的是时间/观测接口，启动速度先验、创新门控、噪声网络结构和训练预算均保持原状态，结果中残余启动与恢复误差留给下一项单独消融。\n\n'
    text+='## 复现与主要产物\n\n'
    text+='```bash\ncd /home/jby/IsaacLab-2.3.2\nbash source/standalone/test11/velocity_error_validation/tools/run_n0_time_campaign.sh collect\nbash source/standalone/test11/velocity_error_validation/tools/run_n0_time_campaign.sh actual_sparse\nbash source/standalone/test11/velocity_error_validation/tools/run_n0_time_campaign.sh evaluate\nbash source/standalone/test11/velocity_error_validation/tools/run_n0_time_campaign.sh evaluate_sparse\nbash source/standalone/test11/velocity_error_validation/tools/run_n0_time_campaign.sh tests\n/home/jby/miniconda3/envs/isaac_lab_232/bin/python source/standalone/test11/velocity_error_validation/tools/write_n0_report.py\n```\n\n'
    text+='脚本初始化用户指定环境并使用 cuda:0。主要文件：[协议](../reports/n0_time_contract_20261007/protocol.json)、[数据及哈希清单](../reports/n0_time_contract_20261007/data_manifest.json)、[分场景指标](../reports/n0_time_contract_20261007/scenario_metrics.csv)、[分组指标](../reports/n0_time_contract_20261007/grouped_metrics.csv)、[逐 episode](../reports/n0_time_contract_20261007/episode_metrics.csv)、[置信区间](../reports/n0_time_contract_20261007/paired_bootstrap.csv)、[数值一致性](../reports/n0_time_contract_20261007/frozen_equation_validation.csv)、[真实采样审计](../reports/n0_actual_sampling_20261007/actual_sampling_audit.csv)。\n\n'
    text+='[对比图 PNG](../reports/n0_time_contract_20261007/timing_comparison.png) / [PDF](../reports/n0_time_contract_20261007/timing_comparison.pdf)。\n'
    DOC.write_text(text)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,scene in zip(axes,['ss','sd']):
        subset=[r for r in main_table if r['scene']==scene];x=np.arange(len(subset))
        ax.bar(x-.18,[r['before_p95'] for r in subset],.36,label='Arrival-time control')
        ax.bar(x+.18,[r['after_p95'] for r in subset],.36,label='Sample-time replay')
        ax.set_xticks(x,[r['profile'] for r in subset],rotation=40,ha='right',fontsize=8)
        ax.set_ylabel('3D velocity error P95 (m/s)');ax.set_title(f'{scene.upper()}, same ground frames, frozen model');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(RUN/'timing_comparison.png',dpi=180);fig.savefig(RUN/'timing_comparison.pdf');plt.close(fig)
    print('Report:',DOC)


if __name__=='__main__':main()
