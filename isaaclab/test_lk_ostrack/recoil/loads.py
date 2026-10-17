"""由内弹道解算结果合成四类后坐载荷时程 (论文式 9-13)。

载荷分类与受力对象
------------------
本模块严格按论文 1.2 节的基本假设划分, **不改变** 各自的作用对象:

    ==============  ================  ==================  ============
    载荷            受力对象 (论文)   方向                本模块角色
    ==============  ================  ==================  ============
    F_t 膛底作用力   枪机              向后 (膛线 -x)     外载荷
    F_s 导气室力     活塞              向后               内力对, 见下
    F_h 后坐撞击力   枪体              向后               外载荷
    F_f 复进撞击力   枪体              向前               外载荷
    ==============  ================  ==================  ============

**F_s 的内力对性质**: 导气室固定在枪管上, 活塞受向后的 F_s, 气室
/ 枪管受等大向前的反力。二者对 "武器整体" 的净冲量恒为零。若按论文
假设 (3) 只把 F_s 施加在活塞上而不施加反作用力, 每发会凭空多出约
2 N*s (约 +28%)。因此本模块默认 **不把 F_s 计入对外净载荷**
(`MODE_FULL_CYCLE`), 只有在显式选择 `MODE_WITH_GAS_ON_PISTON`
时才输出, 并附带警告。

F_h 与 F_f 的方向
-----------------
论文正文 1.7.3 / 1.7.4 节说明: F_h 是 "后坐到位" 的碰撞, F_f 是
"复进到位" 的碰撞, 二者方向必然相反。论文图 6(c)(d) 均绘为正值,
属绘图约定。本模块中 F_h 沿膛线向后、F_f 沿膛线向前, 因此二者的
净冲量近似抵消 (约 ±0.6 N*s 的残差), 与多体模型中 "自动机在枪体
内往复、净动量交换接近零" 的物理图像一致。

半正弦归一化
------------
见 `params.ImpactParams` 的说明: 论文式 (12)(13) 若逐字实现, 冲量只
有 m_z*Δv 的 2/pi; 本模块默认施加 pi/2 修正以满足动量守恒。
"""

from __future__ import annotations

import numpy as np

from ._compat import trapz
from .interior_ballistics import InteriorBallisticsResult, solve_interior_ballistics
from .params import (
    MODE_BREECH_ONLY,
    MODE_FULL_CYCLE,
    MODE_NAIVE_ALL_POSITIVE,
    MODE_WITH_GAS_ON_PISTON,
    WeaponParams,
)


def _half_sine_envelope(t: np.ndarray, t0: float, dt_imp: float) -> np.ndarray:
    """在 [t0, t0+dt_imp] 上取值为 sin(pi*(t-t0)/dt_imp), 其余为 0。"""
    env = np.zeros_like(t)
    mask = (t >= t0) & (t <= t0 + dt_imp)
    if dt_imp > 0.0:
        env[mask] = np.sin(np.pi * (t[mask] - t0) / dt_imp)
    return env


def gas_chamber_force(
    t: np.ndarray,
    t0: float,
    p_d: float,
    weapon: WeaponParams,
) -> np.ndarray:
    """导气室作用力 F_s = S_s * p_d * exp(-u) * (1 - exp(-a*u)), u = t_s/b。

    Args:
        t: 绝对时间网格 [s]。
        t0: 导气室作用力起始时刻 (弹丸经过导气孔的时刻) [s]。
        p_d: 弹丸经过导气孔瞬间的膛内平均压力 [Pa] (式 11 的 p_d)。
        weapon: 武器参数。

    Returns:
        与 `t` 等长的 F_s 数组 [N], 在 t < t0 或 t_s > t_s_max 处为 0。
    """
    g = weapon.gas_chamber
    ts = t - t0
    u = ts / g.b
    # 数值防护: exp(-u) 在 u 很大时下溢, 直接置零
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ps = p_d * np.exp(-u) * (1.0 - np.exp(-g.a * u))
    fs = g.S_s * ps
    valid = (ts >= 0.0) & (ts <= g.t_s_max)
    return np.where(valid, fs, 0.0)


def impact_force(
    t: np.ndarray,
    t0: float,
    dt_imp: float,
    momentum: float,
    apply_half_sine_normalization: bool = True,
) -> np.ndarray:
    """撞击力脉冲 (论文式 12/13 的统一形式)。

    Args:
        t: 绝对时间网格 [s]。
        t0: 撞击起始时刻 [s]。
        dt_imp: 撞击作用时间 Δt [s]。
        momentum: 动量变化 m_z * Δv [N*s]。与论文式 (12)(13) 的分子对应。
        apply_half_sine_normalization: 是否乘 pi/2 使 ∫F dt == momentum。
            为 False 时复现论文公式的 2/pi 缺陷 (仅用于对照)。

    Returns:
        与 `t` 等长的撞击力数组 [N] (幅值, 未施加方向符号)。
    """
    if dt_imp <= 0.0:
        return np.zeros_like(t)
    amp = momentum / dt_imp
    if apply_half_sine_normalization:
        amp *= 0.5 * np.pi
    return amp * _half_sine_envelope(t, t0, dt_imp)


def _shot_loads(
    t: np.ndarray,
    t_fire: float,
    weapon: WeaponParams,
    interior: InteriorBallisticsResult,
) -> dict[str, np.ndarray]:
    """生成单发射击在绝对时间网格 `t` 上的四条载荷曲线 (按方向带符号)。

    符号约定: 膛线 +x 为射击正方向。
        F_t < 0   (向后)
        F_s < 0   (向后, 作用在活塞上)
        F_h < 0   (向后)
        F_f > 0   (向前)
    """
    imp = weapon.impact

    # --- F_t: 把内弹道解平移到 t_fire ---
    F_t = np.zeros_like(t)
    mask = (t >= t_fire) & (t <= t_fire + interior.duration)
    F_t[mask] = np.interp(t[mask] - t_fire, interior.t, interior.F_t)
    F_t = -F_t  # 向后

    # --- F_s: 起始时刻为弹丸经过导气孔的时刻 ---
    # 由内弹道解插值出该时刻与对应的平均压力 p_d
    t_port = float(np.interp(weapon.interior.L_port, interior.L, interior.t))
    p_d = float(np.interp(t_port, interior.t, interior.p))
    F_s = gas_chamber_force(t, t_fire + t_port, p_d, weapon)
    F_s = -F_s  # 向后 (作用在活塞上)

    # --- F_h / F_f ---
    F_h = impact_force(
        t,
        t_fire + imp.t_h_offset,
        imp.dt_h,
        imp.m_z * (imp.v_h - imp.v_fs),
        imp.apply_half_sine_normalization,
    )
    F_h = -F_h  # 后坐到位, 向后

    F_f = impact_force(
        t,
        t_fire + imp.t_f_offset,
        imp.dt_f,
        imp.m_z * imp.effective_v_fe(),
        imp.apply_half_sine_normalization,
    )  # 复进到位, 向前 (保持正号)

    return {"F_t": F_t, "F_s": F_s, "F_h": F_h, "F_f": F_f}


def _combine(mode: str, loads: dict[str, np.ndarray]) -> np.ndarray:
    """按模式把四条载荷合成为对外净载荷 (沿膛线方向)。"""
    if mode == MODE_BREECH_ONLY:
        return loads["F_t"].copy()
    if mode == MODE_FULL_CYCLE:
        # F_h 已带负号, F_f 已带正号, 直接相加即得净撞击载荷
        return loads["F_t"] + loads["F_h"] + loads["F_f"]
    if mode == MODE_WITH_GAS_ON_PISTON:
        # 仅作用在活塞上, 未施加气室反力 -> 每发多出约 2 N*s, 见模块 docstring
        return loads["F_t"] + loads["F_s"] + loads["F_h"] + loads["F_f"]
    if mode == MODE_NAIVE_ALL_POSITIVE:
        # docx 字面做法: 四项按同号叠加 (错误, 仅供对照)
        return (
            np.abs(loads["F_t"])
            + np.abs(loads["F_s"])
            + np.abs(loads["F_h"])
            + np.abs(loads["F_f"])
        )
    raise ValueError(f"未知载荷模式 {mode!r}")


class RecoilLoadSeries:
    """连发射击的四类载荷时程 + 合成净载荷。

    Attributes:
        t: 细网格时间轴 [s]。
        F_t, F_s, F_h, F_f: 四条分量时程 [N], 已带方向符号。
        F_net: 按模式合成的对外净载荷 [N]。
        t_fire: 各发次的开火时刻 [s]。
        interior: 内弹道解算结果 (单发)。
        J_components: 各分量的全过程冲量 {名称: N*s}。
        J_net: 净载荷过程冲量 [N*s]。
        J_reference: 每发理论自由后坐冲量 [N*s]。
        J_total_reference: n_cycles * J_reference。
    """

    def __init__(
        self,
        t: np.ndarray,
        F_t: np.ndarray,
        F_s: np.ndarray,
        F_h: np.ndarray,
        F_f: np.ndarray,
        F_net: np.ndarray,
        t_fire: list[float],
        interior: InteriorBallisticsResult,
        mode: str,
    ) -> None:
        self.t = t
        self.F_t = F_t
        self.F_s = F_s
        self.F_h = F_h
        self.F_f = F_f
        self.F_net = F_net
        self.t_fire = t_fire
        self.interior = interior
        self.mode = mode

        self.J_components = {
            "F_t": float(trapz(F_t, t)),
            "F_s": float(trapz(F_s, t)),
            "F_h": float(trapz(F_h, t)),
            "F_f": float(trapz(F_f, t)),
        }
        self.J_net = float(trapz(F_net, t))

        n = len(t_fire)
        # 符号约定: 沿膛线 +x 为射击正方向, 因此后坐冲量为负。
        # F_t 是唯一直接由火药气体施加的冲量, 其积分等于 (m_b + omega/2)*v_0。
        self.J_reference = -interior.J_target
        self.J_total_reference = -n * interior.J_target

    def peak(self, name: str) -> float:
        """返回某个分量或净载荷的峰值幅值 [N]。"""
        arr = getattr(self, name if name == "F_net" else f"F_{name}")
        return float(np.max(np.abs(arr)))

    def audit(self, tol: float = 0.05) -> str:
        """冲量守恒自检报告。

        这是本模型唯一可证伪的硬判据:

            ∫ F_net dt  ==  n_cycles * (m_b + omega/2) * v_0   (±tol)

        任何一条载荷符号搞错、半正弦未归一化、或把内力对当外力计入,
        都会在这里暴露出来。
        """
        rel = self.J_net / self.J_total_reference - 1.0
        ok = abs(rel) <= tol
        lines = [
            "后坐力模型冲量守恒自检",
            f"  载荷模式        : {self.mode}",
            f"  发数            : {len(self.t_fire)}",
            "  符号约定        : 沿膛线 +x 为正 (射击方向), 后坐为负",
            f"  单发自由后坐冲量: {self.J_reference:.4f} N*s"
            f"  = -(m_b + omega/2)*v_0",
            f"  理论总冲量      : {self.J_total_reference:.4f} N*s",
            "",
            "  各分量冲量 (按施加方向带符号):",
        ]
        for k, v in self.J_components.items():
            lines.append(f"    {k}: {v:+.4f} N*s")
        lines += [
            "",
            f"  净冲量 ∫F_net dt: {self.J_net:+.4f} N*s",
            f"  相对理论值偏差  : {rel * 100:+.2f} %",
            f"  判定 (tol={tol * 100:.0f}%): " + ("通过" if ok else "**不通过**"),
        ]
        return "\n".join(lines)


def generate_recoil_loads(
    weapon: WeaponParams,
    mode: str = MODE_FULL_CYCLE,
    n_cycles: int = 5,
    dt_fine: float = 2.0e-6,
    dt_phys: float = 5.0e-3,
    interior: InteriorBallisticsResult | None = None,
) -> RecoilLoadSeries:
    """生成后坐载荷时程。

    Args:
        weapon: 武器参数。
        mode: 载荷合成模式, 见 `params.VALID_MODES`。
        n_cycles: 连发发数。
        dt_fine: 细网格步长 [s]。
        dt_phys: 目标物理步长 [s]; 只用于把时间轴延长到覆盖整个连发序列。
            真正的重采样在 `to_step_impulse` 中进行。
        interior: 可复用的内弹道解; 为 None 时现场求解。

    Returns:
        RecoilLoadSeries
    """
    if interior is None:
        interior = solve_interior_ballistics(weapon.interior)

    interval = weapon.fire_interval()
    t_fire = [i * interval for i in range(n_cycles)]

    # 时间轴需覆盖到最后一发的复进撞击结束
    t_end = t_fire[-1] + max(
        interior.duration + 1.0e-3,
        weapon.impact.t_f_offset + weapon.impact.dt_f + 1.0e-4,
    )
    n = max(int(np.ceil(t_end / dt_fine)) + 1, 2)
    t = np.arange(n) * dt_fine

    acc = {k: np.zeros_like(t) for k in ("F_t", "F_s", "F_h", "F_f")}
    for tf in t_fire:
        shot = _shot_loads(t, tf, weapon, interior)
        for k in acc:
            acc[k] += shot[k]

    F_net = _combine(mode, acc)

    return RecoilLoadSeries(
        t=t,
        F_t=acc["F_t"],
        F_s=acc["F_s"],
        F_h=acc["F_h"],
        F_f=acc["F_f"],
        F_net=F_net,
        t_fire=t_fire,
        interior=interior,
        mode=mode,
    )


def to_step_impulse(series: RecoilLoadSeries, dt_phys: float) -> np.ndarray:
    """把连续载荷重采样成每个物理步内的**等效力**, 保证冲量严格守恒。

    F_step[k] = (1/dt) * ∫_{k*dt}^{(k+1)*dt} F(t) dt

    这是 dt=5 ms 下唯一正确的注入方式。若改用逐点取值, 1.2 ms 的膛底
    脉冲会落在单步之内, 峰值 19 kN 在一个 5 ms 步里会积分出 95 N*s,
    是真实值的 10 倍以上。

    Args:
        series: `generate_recoil_loads` 的输出。
        dt_phys: 物理步长 [s]。

    Returns:
        形状 (n_steps,) 的等效力数组 [N]。
    """
    if dt_phys <= 0.0:
        raise ValueError("dt_phys 必须为正。")

    n_steps = int(np.floor(series.t[-1] / dt_phys)) + 1
    out = np.zeros(n_steps)

    # 用累计积分再差分, 比逐个区间搜索更快且数值等价
    cumulative = np.concatenate(([0.0], np.cumsum(np.diff(series.t) * 0.5 * (series.F_net[1:] + series.F_net[:-1]))))
    bin_edges = np.arange(n_steps + 1) * dt_phys
    # series.t 与 cumulative 一一对应, 直接在 bin 边界上插值
    cum_at_edges = np.interp(bin_edges, series.t, cumulative, left=0.0, right=cumulative[-1])

    impulses = np.diff(cum_at_edges)
    out[: len(impulses)] = impulses / dt_phys
    return out
