"""Generate work 2 report only from complete frozen paired measurements."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/n6_matplotlib_20261007')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('source/standalone/test11/velocity_error_validation')
RUN=BASE/'reports/n6_interface_20261007'
DOC=BASE/'docs/N6_INTERFACE_RESULTS_20261007.md'


def read(name):return list(csv.DictReader((RUN/name).open()))
def table(rows,keys=None):
    if not rows:return '无符合此分组的评分帧。\n'
    keys=list(rows[0]) if keys is None else keys
    def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+''.join('| '+' | '.join(fmt(r[k]) for k in keys)+' |\n' for r in rows)


def paired(rows,*,group=False,backend='frozen_learned_filter',profiles=('sync','combined'),horizons=(.1,.2)):
    keys=['dataset','scenario','profile','backend','filter','event','horizon_s']
    if group:keys=['group']+keys
    lookup={tuple(r[k] for k in keys)+(r['method'],):r for r in rows}
    out=[]
    for r in rows:
        if r['backend']!=backend or r['filter']!='ground' or r['event']!='direct_arrival' or r['method']!='posterior_position_with_same_velocity':continue
        if profiles is not None and r['profile'] not in profiles:continue
        if float(r['horizon_s']) not in horizons:continue
        raw=lookup[tuple(r[k] for k in keys)+('raw_position_with_same_velocity',)]
        item=dict(dataset=r['dataset'],scene=r['group'] if group else r['scenario'],profile=r['profile'],horizon_ms=int(float(r['horizon_s'])*1000),n=int(r['n']))
        for k in ('mean','p95','p99'):item['raw_'+k]=float(raw[k]);item['posterior_'+k]=float(r[k])
        out.append(item)
    return out


def main():
    if not (RUN/'COMPLETE').exists():raise ValueError('Evaluation incomplete')
    forecast=read('forecast_metrics.csv');groups=read('grouped_forecast_metrics.csv')
    unc=read('uncertainty_metrics.csv');checks=read('interface_validation.csv')
    bootstrap=read('paired_bootstrap.csv');runtime=read('runtime_diagnostic.csv')
    manifest=json.loads((RUN/'data_manifest.json').read_text());execution=json.loads((RUN/'execution.json').read_text())
    protocol=json.loads((RUN/'protocol.json').read_text())
    tests=(RUN/'tests.log').read_text().strip().splitlines()[-1]
    if 'failed' in tests or 'passed' not in tests:raise ValueError('Regression tests did not pass')
    pairs=paired(forecast)
    text='# 工作 2 / N6 接口诊断：后验位置、联合协方差与外推结果\n\n'
    text+='本轮完成明确时间与参考系的滤波状态输出、100/200 ms 后验位置外推、同帧原始位置控制、完整协方差和覆盖诊断。使用 isaac_lab_232、Isaac Sim setup、cuda:0；复用工作 1 的已完成采集，冻结 learned_filter_seed41 和普通 CV 的 q=.1/r=.1，没有训练或参数搜索。\n\n'
    text+=f'评估 {len(manifest)} 条 episode，原始物理帧 {sum(r["frames"] for r in manifest):,}，共有 {len(checks)} 条“episode×退化×后端”输出流。新增接口与工作 1 的冻结速度最大差为 {max(float(r["velocity_max_difference_mps"]) for r in checks):.8g} m/s，输出覆盖率全部为 100%。本轮优化的是状态接口和位置外推使用方式，速度误差没有进一步下降。\n\n'
    text+='## 输出契约与调用\n\n'
    fields=[
        dict(field='state_time_s / output_time_s',meaning='返回状态及预测对应的物理参考时间；逐包接口中二者相等，不是 Python 调用完成的墙钟时间'),
        dict(field='posterior_time_s',meaning='送入输出函数的原始滤波状态时间；若显式传播到较晚 output_time，则与返回状态时间不同'),
        dict(field='last_direct_measurement_time_s / measurement_age_s',meaning='最近已融合直接观测的采样时刻及到输出时刻的年龄，不能用到达时间替代'),
        dict(field='position_local_m / velocity_local_mps',meaning='固定局部 yaw 里程计参考系中的目标位置和绝对平移速度'),
        dict(field='position_base_m / velocity_base_mps',meaning='输出时刻 yaw-only 机身系中的相对位置和物理相对平移速度；不是旋转坐标位置的总导数'),
        dict(field='covariance_local / covariance_base',meaning='未裁剪 6×6 联合协方差，顺序 [px,py,pz,vx,vy,vz]；整体旋转包含 Ppv/Pvp'),
        dict(field='sigma_base_mps',meaning='由完整速度协方差对角得到的未裁剪标准差；旧 sigma_mps 字段保留训练接口兼容语义'),
        dict(field='reference_epoch_s / reference_generation',meaning='局部原点为 episode 首个 tick 的机身位置，初始 heading 为零；真正身份重置产生新 generation'),
        dict(field='initialized / position_available / forecast_valid / status',meaning='区分未初始化、收到直接观测更新、纯预测；首次直接观测前位置不可用，缺测仍输出，旧快照不被晚包修改'),
        dict(field='covariance_scope',meaning='条件于当前本体 pose/velocity 和采样时钟的目标模型不确定度，未包括这些输入的估计误差'),
    ]
    text+=table(fields)+'\n'
    text+='`CausalSensorTimeEstimator.advance()` 是工作 1 物理时间接口；`CausalVelocityCandidate.update()` 的 CV 后端也输出相同状态字段。GRU/TCN 只有速度状态，不伪造其六维物理状态或协方差。angular-z 仍须是 yaw-only heading 导数。独立设备时钟和无法按曝光关联的传感器仍需要同步适配器。\n\n'
    text+='旧 feature 接口显式校验 feature dt 与 timestamp 差一致、首 tick 的 dt 为零；CV 大于 0.5 s 的间隔明确报错并要求使用物理逐包接口，避免旧内部截断却将状态标成更晚时间。本轮正式物理逐包路径按真实总时长传播，没有该截断。\n\n'
    text+='```python\nfrom n0_time_replay import CausalSensorTimeEstimator, NoiseBackend\nfrom speed_prediction.filter_state_output import forecast_local\n\n# model: 已加载冻结权重的 LearnedCVFilter；每个目标一份 estimator\nestimator = CausalSensorTimeEstimator(NoiseBackend(model, "cuda:0"))\nestimate = estimator.advance(sample_id, reference_time_s, arrived_deliveries)\nif estimate["forecast_valid"]:\n    p100 = forecast_local(estimate, 0.100)\n    # p100["position_local_m"], p100["covariance_position_local"]\n    # prediction_time_s 是 estimate["state_time_s"] + 0.100\n# 真正身份改变调用 estimator.reset()；跟踪器短暂重获不重置目标状态。\n```\n\n'
    text+='导入 tools 中的接口时将该 tools 目录及 test11 根目录放入 Python import 路径。输出均是独立快照，不共享滤波器的可变数组。\n\n'
    text+='## 外推协议：位置不同，速度完全相同\n\n'
    text+='后验路径使用 `p_hat(t) + H*v_hat_abs(t)`。控制路径使用最后已到达、直接有效观测的原始局部位置 `z(t_s) + (t-t_s+H)*v_hat_abs(t)`，已包含观测到输出的年龄补偿。两者使用同一个冻结滤波速度，不能把相对机身速度直接当作目标绝对速度。直接观测组要求当前 tick 新融合一个有效曝光配对；纯预测及未初始化单独评分，未推迟 score 起点。\n\n'
    text+='GT 世界锚点、目标速度标签和未来目标坐标只参与事后评分；估计器只收到共同物理 tick 和已到达包。未来端点须在同一 episode 内且有精确对应物理帧，不跨身份重置。地上过滤按外推起点定义；全帧压力结果保存在 CSV。新物理参考系不会读取 GT 世界锚点，该锚点仅将局部预测转换为可评分的标签坐标。\n\n'
    text+='dense 为工作 1 六组完整采集上的七种配对传输；actual_sparse 为两组在 LK/OSTrack 前真正跳过图像的采集，后端传输同步。下表 n 是两种位置路径共用的直接观测评分帧数，误差为三维位置误差范数，单位 m。\n\n'
    text+=table(pairs)+'\n'
    for scene in ('sd','ss'):
        for profile in ('sync','combined'):
            r=next(r for r in pairs if r['dataset']=='dense' and r['scene']==scene and r['profile']==profile and r['horizon_ms']==100)
            dm=(r['posterior_mean']/r['raw_mean']-1)*100;dp=(r['posterior_p95']/r['raw_p95']-1)*100
            text+=f'{scene.upper()} / {profile} 的 100 ms 外推：Mean 变化 {dm:+.1f}%，P95 变化 {dp:+.1f}%；正值为退化。\n\n'
    text+='因此本轮提供两个明确的输出选项，不把后验位置自动当作所有场景的优胜方案；接口可用与模型精度提升是两个独立验收项。没有根据测试 GT 逐帧挑选较优路径。\n\n'
    text+='## 单因素、机身运动与普通 CV 控制\n\n'
    text+=table(paired(forecast,profiles=None,horizons=(.1,)))+'\n'
    text+=table(paired(groups,horizons=(.1,),group=True))+'\n'
    text+='普通 CV 使用上轮已冻结的参数，未在本轮测试集调参；下表区分时间链路修复与学习噪声的贡献。\n\n'
    text+=table(paired(forecast,backend='fixed_cv'))+'\n'
    text+='## 缺测和未初始化输出\n\n'
    other=[dict(dataset=r['dataset'],scene=r['scenario'],profile=r['profile'],event=r['event'],horizon_ms=int(float(r['horizon_s'])*1000),
        n=int(r['n']),mean=float(r['mean']),p95=float(r['p95']),p99=float(r['p99'])) for r in forecast
        if r['backend']=='frozen_learned_filter' and r['filter']=='ground' and r['event'] in ('prediction_only','uninitialized','all_outputs')
        and float(r['horizon_s'])==.1 and r['profile'] in ('sync','combined')]
    text+=table(other)+'\n'
    text+='缺测表只有后验预测路径；单独列出未初始化输出，不用删除这些帧来降低 all_outputs 误差。未初始化先验的三维位置误差可达百米，接口因此显式返回 position_available=False / forecast_valid=False，forecast_local 默认拒绝外推；诊断评分显式保留这个先验。收到第一份直接位置后，缺测传播仍有效。状态输出覆盖率 100% 不等于有效定位覆盖率 100%。直接观测与缺测的帧集合不同，不能用两张表的数字相减判断改善。\n\n'
    availability=[]
    for r in forecast:
        if r['filter']!='ground' or r['event']!='all_outputs' or float(r['horizon_s'])!=0 or r['method']!='posterior_position_with_same_velocity':continue
        unknown=next((int(u['n']) for u in forecast if all(u[k]==r[k] for k in ('dataset','scenario','profile','backend','filter','horizon_s','method')) and u['event']=='uninitialized'),0)
        availability.append(dict(dataset=r['dataset'],scene=r['scenario'],profile=r['profile'],backend=r['backend'],score_frames=int(r['n']),uninitialized=unknown,position_available_fraction=1-unknown/int(r['n'])))
    with (RUN/'position_availability.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(availability[0]));w.writeheader();w.writerows(availability)
    text+='评分帧中的有效位置可用率如下，缺测传播继续计为可用，未初始化计为不可用；详见 position_availability.csv。\n\n'
    text+=table([r for r in availability if r['backend']=='frozen_learned_filter' and r['profile'] in ('sync','combined')])+'\n'
    text+='## 未裁剪完整协方差、NEES 和覆盖\n\n'
    text+='外推使用 `Ppp + H*(Ppv+Pvp) + H²*Pvv + diag(q)*H³/3`，完整联合传播同步保留交叉项。q 为输出时刻因果噪声网络的过程强度，在外推区间内保持不变；没有查询未来观测或网络输出。新接口输出未裁剪协方差，训练 log_sigma 截断不应用到此协方差。\n\n'
    chosen=[dict(dataset=r['dataset'],scene=r['scenario'],profile=r['profile'],event=r['event'],kind=r['kind'],horizon_ms=int(float(r['horizon_s'])*1000),n=int(r['n']),
                 nees_mean=float(r['nees_mean']),ellipsoid_coverage95=float(r['ellipsoid_coverage95']),axis_coverage95=float(r['axis_coverage95']),mean_ellipsoid_volume=float(r['mean_ellipsoid_volume'])) for r in unc
        if r['backend']=='frozen_learned_filter' and r['filter']=='ground' and r['event']=='all_outputs'
        and r['profile'] in ('sync','combined') and ((r['kind']=='forecast_position' and float(r['horizon_s'])==.1) or r['kind'] in ('current_velocity','current_joint_state'))]
    text+=table(chosen)+'\n'
    text+='NEES 直接使用完整协方差解线性系统。三维位置/速度的 95% 椭球阈值为 χ²(df=3)，六维联合状态为 χ²(df=6)；按轴 ±1.96σ 覆盖率另列，不能替代椭球覆盖率。位置椭球体积单位 m³，速度为 (m/s)³，联合体积为 m³·(m/s)³。NEES 的理想均值分别为 3/6，但相关帧和模型失配限制此参照，不作独立样本显著性检验。\n\n'
    text+='这仍是目标条件协方差。本体积分、姿态、时钟误差及观测偏差未纳入，当前覆盖率不能视为实机已校准风险或用于直接承诺命中概率；也没有用扩大协方差调高覆盖率。\n\n'
    sd_cov=[r for r in chosen if r['dataset']=='dense' and r['scene']=='sd' and r['kind']=='forecast_position']
    text+='本轮 SD 的 100 ms 三维位置“95%”椭球实际覆盖率为 '+ '、'.join(f'{r["profile"]} {r["ellipsoid_coverage95"]*100:.2f}%' for r in sd_cov)+'，低于名义 95%，因此协方差接口通过数值验收，但概率校准尚未通过。\n\n'
    text+='## 配对区间、验收与性能边界\n\n'
    intervals=[dict(dataset=r['dataset'],scene=r['scenario'],profile=r['profile'],horizon_ms=int(float(r['horizon_s'])*1000),metric=r['metric'],blocks=int(r['blocks']),
        ci_low=float(r['delta_ci_low']),ci_high=float(r['delta_ci_high'])) for r in bootstrap
        if r['backend']=='frozen_learned_filter' and r['filter']=='ground' and float(r['horizon_s'])==.1]
    text+='500 次配对 bootstrap 以采集 seed×环境为块，区间为后验位置路径减去原始位置路径；负值为改善。模型冻结，区间不包含训练种子或实机泛化的不确定性。\n\n'+table(intervals)+'\n'
    text+=f'回归测试：**{tests}**。所有 {len(checks)} 条输出流的协方差最小特征值为 {min(float(r["covariance_min_eigenvalue"]) for r in checks):.8g}，最大不对称量为 {max(float(r["covariance_max_asymmetry"]) for r in checks):.8g}，状态时间/输出时间与物理参考时钟的最大差为 {max(float(r["state_output_time_max_difference_s"]) for r in checks):.8g} s。测试覆盖完整交叉协方差传播、参考系旋转、0.5 s 以上传播、不确定度不裁剪、因果快照、身份重置和旧速度兼容。\n\n'
    if (RUN/'legacy_bridge_validation.csv').exists():
        legacy=read('legacy_bridge_validation.csv')
        text+=f'原 feature CV 接口另外在旧 SS/SD×clean/mixed 四条完整长流（共 {sum(int(r["frames"]) for r in legacy)} 帧）上验收，对原冻结 CUDA 预测的最大速度差为 {max(float(r["velocity_max_difference_mps"]) for r in legacy):.8g} m/s；导出后验均值与内部状态完全相同，协方差只消除浮点非对称量。见 [兼容接口验证](../reports/n6_interface_20261007/legacy_bridge_validation.csv)。\n\n'
    timing=[]
    for backend in ('frozen_learned_filter','fixed_cv'):
        for start in (0,100,300):
            subset=[float(r['p95_ms']) for r in runtime if r['backend']==backend and int(r['frame_start'])==start]
            if subset:timing.append(dict(backend=backend,frame_start=start,streams=len(subset),median_stream_p95_ms=float(np.median(subset)),maximum_stream_p95_ms=max(subset)))
    text+='下表为参考逐包接口的分长度诊断，包含前处理、延迟重传播、状态输出和 CUDA 噪声网络同步；统计量是各条流 P95 的中位数/最大值，不是全部帧的 pooled P95。GT 评分和图像跟踪不在计时内。\n\n'+table(timing)+'\n'
    text+=f'正式评估耗时 {execution["elapsed_s"]:.1f} s，CUDA 设备 {execution["gpu"]}。这次使用小型冻结模型和 CUDA 协方差评分，不需要占满显存。\n\n'
    text+='本轮是 N6 的接口诊断。历史仍用于参考回放审计，尚未完成长期有界缓存、10–30 分钟运行、多目标调度与完整图像到预测的端到端时延验收；不能据此宣称生产在线性能已经达标。\n\n'
    text+='## 复现与产物\n\n'
    text+='```bash\ncd /home/jby/IsaacLab-2.3.2\nbash source/standalone/test11/velocity_error_validation/tools/run_n6_interface_campaign.sh tests\nbash source/standalone/test11/velocity_error_validation/tools/run_n6_interface_campaign.sh evaluate\nbash source/standalone/test11/velocity_error_validation/tools/run_n6_interface_campaign.sh report\n```\n\n'
    text+='脚本初始化指定 conda/Isaac 环境并选择 cuda:0。已完成正式目录不可覆盖；重新评估需使用新的 `--out`，以保留协议与哈希。\n\n'
    text+='现有正式结果已完成；如需复跑，在上述 conda/Isaac 环境内直接执行以下命令，输出写入新目录：\n\n```bash\npython source/standalone/test11/velocity_error_validation/tools/evaluate_n6_interface.py \\\n  --device cuda:0 \\\n  --out source/standalone/test11/velocity_error_validation/reports/n6_interface_rerun_20261007\n```\n\n'
    text+='代码：[统一状态输出](../../speed_prediction/filter_state_output.py)、[物理逐包接口](../tools/n0_time_replay.py)、[原 CV 兼容接口](../tools/online_experimental_velocity.py)、[评估](../tools/evaluate_n6_interface.py)、[回归测试](../tests/test_filter_state_output.py)。\n\n'
    text+='产物：[冻结协议](../reports/n6_interface_20261007/protocol.json)、[数据清单](../reports/n6_interface_20261007/data_manifest.json)、[外推指标](../reports/n6_interface_20261007/forecast_metrics.csv)、[机身运动分组](../reports/n6_interface_20261007/grouped_forecast_metrics.csv)、[完整协方差诊断](../reports/n6_interface_20261007/uncertainty_metrics.csv)、[配对置信区间](../reports/n6_interface_20261007/paired_bootstrap.csv)、[逐帧状态与未裁剪协方差](../reports/n6_interface_20261007/state_outputs.pt)、[速度及时间一致性](../reports/n6_interface_20261007/interface_validation.csv)。\n\n'
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for axis,scene in zip(axes,('ss','sd')):
        data=[r for r in pairs if r['dataset']=='dense' and r['scene']==scene]
        labels=[f'{r["profile"]}\n{r["horizon_ms"]} ms' for r in data];x=np.arange(len(data))
        axis.bar(x-.18,[r['raw_p95'] for r in data],.36,label='Raw position + same velocity')
        axis.bar(x+.18,[r['posterior_p95'] for r in data],.36,label='Posterior position + same velocity')
        axis.set_xticks(x,labels);axis.set_title(scene.upper()+', matched direct ground frames')
        axis.set_ylabel('Position forecast P95 (m)');axis.grid(axis='y',alpha=.2)
        axis.set_ylim(0,max(max(r['raw_p95'],r['posterior_p95']) for r in data)*1.25)
        axis.legend(fontsize=8)
    fig.savefig(RUN/'position_forecast_comparison.png',dpi=180);fig.savefig(RUN/'position_forecast_comparison.pdf');plt.close(fig)
    text+='[外推对比图 PNG](../reports/n6_interface_20261007/position_forecast_comparison.png) / [PDF](../reports/n6_interface_20261007/position_forecast_comparison.pdf)。\n'
    DOC.write_text(text);print(DOC)


if __name__=='__main__':main()
