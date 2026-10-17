"""后坐力模型参数定义。

参考文献
--------
陆星宇, 曹岩枫, 庄圆, 高云龙, 康锦煜, 徐诚.
四足打击平台多工况力学特性及射击扰动.
工程科学学报, 2024, 46(8): 1434-1445.
DOI: 10.13374/j.issn2095-9389.2023.06.29.002

本模块对应论文式 (2)-(13) 所需的全部输入参数。

单位约定
--------
全部使用 SI 单位: m, kg, s, Pa, N, rad.
角度类参数在配置中可写角度, 内部统一转成弧度。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field


@dataclass
class InteriorBallisticsParams:
    """内弹道方程组 (论文式 2-9) 的参数。

    默认值取自公开文献中 7.62 mm 弹药的典型量级, 并与论文给出的
    枪口初速 825 m/s、理论射速 750 发/分 相吻合。

    `f_powder` 与 `I_K` 由 `calibrate()` 自动标定到目标 (v0, Ft_peak),
    这里给出的默认值是已标定过的结果。
    """

    # ---- 弹丸 ----
    m_b: float = 9.6e-3
    """弹丸质量 m_b [kg]。7.62x54R 级弹药典型值 9.6 g。"""

    v_0: float = 825.0
    """枪口初速 v_0 [m/s]。论文 1.1 节给出: 825 m/s。

    注意: 在求解器里这是 **标定目标**, 不是积分的终止条件; 积分终止于
    弹丸行程达到 `L_barrel`。真实的物理因果是 "枪管长度 + 装药决定初速",
    而不是反过来。
    """

    L_barrel: float = 0.55
    """弹丸有效行程 (枪管长度) [m]。

    这是积分终止条件, 也是把 (f_powder, I_K) 定住的第二个几何约束。
    7.62 mm 机枪枪管长 0.5-0.6 m; 取 0.55 m。
    """

    caliber: float = 7.62e-3
    """口径 d [m]。论文 1.1 节给出: 7.62 mm。"""

    # ---- 发射药 ----
    omega: float = 3.1e-3
    """装药量 omega [kg]。"""

    rho_g: float = 1600.0
    """火药密度 rho_g [kg/m^3]。单基/双基药典型 1550-1650。"""

    f_powder: float = 4.721653e5
    """火药力 f [J/kg]。由 `calibrate()` 标定到 (v_0=825 m/s, F_peak=19 kN)。"""

    alpha: float = 1.0e-3
    """火药气体余容 alpha [m^3/kg]。典型 0.9e-3 - 1.1e-3。"""

    k_adiabatic: float = 1.25
    """绝热指数 k。内弹道常用 1.20 - 1.27。"""

    # ---- 火药形状函数 (式 2) ----
    chi_g: float = 0.85
    """火药形状特性量 chi_g。七孔粒状药典型 0.75-0.90。"""

    lam_g: float = -0.15
    """火药形状特性量 lambda_g。典型 -0.05 - -0.25。"""

    # ---- 药室 ----
    W_0: float = 4.0e-6
    """药室容积 W_0 [m^3]。7.62x54R 药室自由容积约 3.5-4.5 cm^3。"""

    # ---- 其他 ----
    phi: float = 1.10
    """次要功系数 phi (式 4, 5, 8)。典型 1.05 - 1.15。"""

    p_0: float = 30.0e6
    """起动压力 p_0 [Pa]: 弹丸挤进膛线所需压力, 定容期终止判据。"""

    I_K: float = 6.320312e4
    """压力全冲量 I_K [Pa*s]: dZ/dt = p / I_K (式 3)。由 `calibrate()` 标定。"""

    L_port: float = 0.10
    """导气孔位置 [m]: 弹丸行程达到该值时火药气体开始进入导气室。

    用于确定导气室作用力的起始时刻与 p_d (式 11 中的 "弹丸经过导气孔
    瞬间的膛内火药燃气平均压力")。

    取 0.10 m: 对应 t_open ≈ 0.80 ms, 与论文图 6(b) 插图中导气室压力
    的起升时刻一致 (7.62 mm 机枪的导气孔通常位于枪管前段)。
    """

    def S(self) -> float:
        """线膛内横断面积 S = pi*d^2/4 [m^2]。"""
        return math.pi * (self.caliber / 2.0) ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["S"] = self.S()
        return d


@dataclass
class GasChamberParams:
    """导气室作用力 (论文式 10-11) 的参数。

    p_s = p_d * exp(-t_s / b) * (1 - exp(-a * t_s / b))
    F_s = S_s * p_s

    注意: a 与 b 在原论文中仅称为 "导气装置结构参数" 与 "时间系数",
    未给出取值范围; 论文该式转引自文献 [27] 的布拉文经验公式。
    """

    S_s: float = 1.55e-4
    """导气装置活塞面积 S_s [m^2]。约合直径 14 mm。"""

    a: float = 0.65
    """导气装置结构参数 a (无量纲经验参数)。

    与 `b` 一起由图 6(b) 的插图标定: 峰值时刻 1.0 ms、峰值 7 kN、
    1.6 ms 衰减到约 1.2 kN。论文正文只说明 a 与 b 是经验参数, 未给
    取值范围; 上述取值使曲线与论文插图在 3 个点上同时吻合。
    """

    b: float = 2.384e-4
    """时间系数 b [s]。见 `a` 的说明。"""

    t_s_max: float = 3.0e-3
    """截断时长 [s]: t_s 超过该值后 F_s 按 0 处理 (曲线已衰减到可忽略)。"""

    def peak_time(self) -> float:
        """p_s 取极大值的时刻 t_s* = b * ln(1+a) / a [s]。"""
        return self.b * math.log1p(self.a) / self.a

    def peak_factor(self) -> float:
        """p_s 峰值 / p_d。"""
        return self.a * (1.0 + self.a) ** (-(1.0 + 1.0 / self.a))


@dataclass
class ImpactParams:
    """后坐/复进到位撞击力 (论文式 12-13) 的参数。

    原论文式 (12):
        F_h = m_z (v_h - v_fs) / dt_h * sin(pi t / dt_h)
    原论文式 (13):
        F_f = m_z v_fe / dt_f * sin(pi t / dt_f)

    **重要更正 — 半正弦归一化**

    半正弦脉冲在 [0, dt] 上的积分是振幅的 2*dt/pi, 而不是 1*dt。
    因此上式的冲量只有 m_z*Δv 的 2/pi ≈ 63.7%, 与正文所称
    "利用动量守恒定律求解" 矛盾。要满足动量守恒, 振幅须乘 pi/2:

        F_h = (pi/2) * m_z (v_h - v_fs) / dt_h * sin(pi t / dt_h)
        F_f = (pi/2) * m_z v_fe / dt_f * sin(pi t / dt_f)

    按此修正后, 用 m_z ≈ 0.5 kg、Δv ≈ 4 m/s、dt_h ≈ 1 ms 反算,
    峰值约 3.14 kN —— 与论文图 6(c) 读出的 3.0 kN 一致; 若不做修正
    则只有 2.0 kN。可判定论文绘图时用了修正后的振幅, 而公式排版漏了
    系数。本实现默认采用修正形式 (`apply_half_sine_normalization=True`),
    同时保留开关以便与原文公式逐字对照。
    """

    m_z: float = 0.5
    """自动机 (枪机/枪机框组件) 质量 m_z [kg]。"""

    v_h: float = 5.0
    """后坐到位速度 v_h [m/s]。"""

    v_fs: float = 1.0
    """复进开始速度 v_fs [m/s]。"""

    v_fe: float = 2.4
    """复进到位速度 v_fe [m/s]。论文未给出数值。"""

    balance_impacts: bool = True
    """是否强制后坐撞击与复进撞击的冲量相互抵消。

    **为什么需要这个开关**

    撞击力、复进簧力、导气室力对枪管/活塞的作用**都是武器内部的相互作用**:
    自动机在一个射击循环后回到闭锁位置、系统回到初始状态, 因此这些内力在
    一个完整循环上的冲量和必须为零。整个武器对外(即对架座/平台)的净冲量
    只来自膛底作用力 F_t —— 它的反作用是弹丸与火药气体带走的动量。

    但论文给出的参数 (m_z, v_h, v_fs, v_fe) 并不满足这个守恒关系:
    按图中参数算出的后坐撞击冲量为 m_z(v_h-v_fs) = 2.0 N*s, 而复进撞击
    冲量为 m_z*v_fe = 1.2 N*s, 二者相差 0.8 N*s。直接使用会让每发凭空
    多出 0.8 N*s (五连发 +9.5%), 这正是 `audit()` 会报出的偏差。

    开启本开关时, 取 v_fe = v_h - v_fs, 使两侧撞击冲量严格抵消。这一取值
    的物理依据: 自动机在每个循环结束时回到同一闭锁状态, 所以它在后坐到位
    撞击中交给机匣的动量, 必须在复进到位撞击中如数归还; 复进簧只改变动量
    传递的时间分布, 不改变总量。

    关闭本开关则逐字复现论文参数, `audit()` 会报出约 +9.5% 的冲量超计,
    仅用于对照。
    """

    def effective_v_fe(self) -> float:
        """实际用于生成 F_f 的复进到位速度 [m/s]。"""
        if self.balance_impacts:
            return self.v_h - self.v_fs
        return self.v_fe

    def strike_impulses(self) -> tuple[float, float]:
        """返回 (后坐撞动冲量, 复进撞击冲量) 的幅值 [N*s]。"""
        return (
            self.m_z * (self.v_h - self.v_fs),
            self.m_z * self.effective_v_fe(),
        )

    dt_h: float = 1.0e-3
    """后坐到位撞击作用时间 dt_h [s]。"""

    dt_f: float = 1.654e-3
    """复进到位撞击作用时间 dt_f [s]。

    论文图 6(d) 的插图中, F_f 峰值约 1.9 kN、低于 F_h 的 3.0 kN, 而两次
    撞击的**冲量必须相等** (自动机每循环回到同一闭锁状态, 见
    `balance_impacts`)。在冲量相等的前提下, 峰值之比只能由作用时间之比
    承担:

        dt_f = (pi/2) * m_z*(v_h-v_fs) / F_f_peak
             = (pi/2) * 2.0 / 1.9e3 = 1.654 ms

    这与图 6(d) 插图的横轴量程 (0.070-0.085 s, 15 ms 宽) 不矛盾。
    """

    t_h_offset: float = 3.0e-2
    """后坐到位撞击相对击发的起始时刻 [s]。论文图 6(c) 约 30 ms。"""

    t_f_offset: float = 7.6e-2
    """复进到位撞击相对击发的起始时刻 [s]。论文图 6(d) 约 76 ms。"""

    apply_half_sine_normalization: bool = True
    """是否施加 pi/2 半正弦归一化修正 (见上)。"""


@dataclass
class WeaponParams:
    """一件武器的完整后坐力参数集合。"""

    name: str = "7.62mm_machine_gun"
    interior: InteriorBallisticsParams = field(default_factory=InteriorBallisticsParams)
    gas_chamber: GasChamberParams = field(default_factory=GasChamberParams)
    impact: ImpactParams = field(default_factory=ImpactParams)

    rpm: float = 750.0
    """理论射速 [发/分]。论文 1.1 节: 750 发/分。"""

    def fire_interval(self) -> float:
        """相邻两发的间隔 [s]。"""
        return 60.0 / self.rpm

    def to_json(self) -> str:
        d = {
            "name": self.name,
            "rpm": self.rpm,
            "interior": self.interior.to_dict(),
            "gas_chamber": asdict(self.gas_chamber),
            "impact": asdict(self.impact),
        }
        return json.dumps(d, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 载荷合成模式
# ---------------------------------------------------------------------------

#: 只施加膛底作用力 F_t。
#:
#: 最简、最保守的模型。理论上 ∫F_t dt 就等于内弹道期的自由后坐冲量,
#: 因此这一项单独就能给出正确的净冲量。
MODE_BREECH_ONLY = "breech_only"

#: 施加 F_t + F_h - F_f (自动机全循环)。
#:
#: F_h 与 F_f 量级相近、方向相反, 二者对净冲量近似抵消, 但对
#: 峰值波形与相位有显著影响。这是 "刚性连接 + 完整自动机" 的推荐模型。
MODE_FULL_CYCLE = "full_cycle"

#: 施加 F_t + F_h - F_f, 并按论文假设同时把 F_s 作用在活塞上。
#:
#: 注意: F_s 与其反作用力构成内力对。若只施加活塞端而不施加气室端,
#: 每发会凭空多出约 2 N*s (约 +28%)。此模式仅供对照, 需要调用方
#: 自行确保反作用力被正确建模, 否则不要使用。
MODE_WITH_GAS_ON_PISTON = "with_gas_on_piston"

#: 四类载荷全部按正号叠加。
#:
#: **这是 docx 字面理解的错误做法**, 仅用于演示冲量超计, 不要用于
#: 生产仿真。audit 会报出约 +70% 的冲量偏差。
MODE_NAIVE_ALL_POSITIVE = "naive_all_positive"

VALID_MODES = (
    MODE_BREECH_ONLY,
    MODE_FULL_CYCLE,
    MODE_WITH_GAS_ON_PISTON,
    MODE_NAIVE_ALL_POSITIVE,
)


@dataclass
class RecoilLoadCfg:
    """载荷生成配置。"""

    weapon: WeaponParams = field(default_factory=WeaponParams)

    mode: str = MODE_FULL_CYCLE
    """载荷合成模式, 取值见 `VALID_MODES`。"""

    n_cycles: int = 5
    """连发发数。论文连发工况为五连发。"""

    dt_fine: float = 2.0e-6
    """细网格步长 [s], 用于精确积分脉冲。"""

    dt_phys: float = 5.0e-3
    """目标仿真的物理步长 [s], 用于把载荷重采样成逐步等效力。

    重采样按 **冲量积分** 而非逐点取值:
        F_step[k] = (1/dt) * ∫_{k*dt}^{(k+1)*dt} F(t) dt
    这样无论脉冲比 dt 短多少, 总冲量都严格守恒。
    """

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"未知载荷模式 {self.mode!r}, 可选: {VALID_MODES}")
