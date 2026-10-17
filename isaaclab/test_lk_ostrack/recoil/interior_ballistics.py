"""内弹道方程组求解器 (论文式 2-9)。

方程
----
    psi        = chi_g * Z * (1 + lam_g * Z)                        (2)
    dZ/dt      = p / I_K                                            (3)
    S * p      = phi * m_b * dv_b/dt                                (4)
    S*p*(L_psi+L) = f*omega*psi - (k-1)/2 * phi*m_b*v_b^2           (5)
    L_psi      = W_0/S [1 - omega/(rho_g W_0)
                        - omega/W_0 * (alpha - 1/rho_g) * psi]      (6)
    v_b        = dL/dt                                              (7)
    p_t        = (1 + omega/(6*phi*m_b)) * p                        (8)
    F_t        = S * p_t                                            (9)

分为两个阶段求解:

**阶段 1 — 定容燃烧 (startup)**
    弹丸尚未运动 (L = 0, v = 0)。式 (5) 中的动能项为零:

        S * p * L_psi = f * omega * psi
        =>  p = f*omega*psi / (S * L_psi)

    对 Z 做 RK4 积分, 直到 p 达到起动压力 p_0 (弹丸挤进膛线所需的压力)。

**阶段 2 — 变容燃烧 (motion)**
    由式 (4) 代入式 (5) 消去 p:

        dv/dt = [ f*omega*psi/(phi*m_b) - (k-1)/2 * v^2 ] / (L_psi + L)
        dL/dt = v
        dZ/dt = p / I_K,  p = phi*m_b*(dv/dt)/S

    对 [L, v, Z] 做 RK4 积分, 直到弹丸行程 L 达到枪管长度 L_barrel (弹丸出膛)。

关于后效期
----------
本实现**不**单独添加后效期压力衰减。原因: 式 (8) 中的 1/6 项已经把
火药气体的动量贡献计入了膛底压力。数值上:

    ∫F_t dt = ∫S*p_t dt = (phi*m_b + omega/6) * v_0

代入 phi=1.10, m_b=9.6e-3, omega=3.1e-3, v_0=825 得 9.14 N*s;
而自由后坐冲量 J = (m_b + omega/2)*v_0 = 9.20 N*s。二者相差 0.7%,
说明内弹道期单独积分就已给出正确的净冲量, 再加后效期会重复计入。

这一结果同时交叉验证了式 (8) 中 1/6 系数的合理性 —— 该系数所需的
phi 为 1.108, 与常用值 1.10 吻合。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._compat import trapz
from .params import InteriorBallisticsParams


@dataclass
class InteriorBallisticsResult:
    """内弹道求解结果。所有数组长度相同, 时间从击发瞬间起算。"""

    t: np.ndarray
    """时间 [s]。"""

    L: np.ndarray
    """弹丸行程 [m]。"""

    v: np.ndarray
    """弹丸速度 [m/s]。"""

    p: np.ndarray
    """膛内火药气体平均压力 [Pa]。"""

    p_t: np.ndarray
    """膛底压力 [Pa]。"""

    F_t: np.ndarray
    """膛底作用力 [N]。"""

    psi: np.ndarray
    """火药燃烧的相对质量 [1]。"""

    duration: float
    """内弹道持续时间 [s] (击发到弹丸出膛)。"""

    travel: float
    """弹丸出膛时的行程 [m] (应等于 params.L_barrel)。"""

    v_exit: float
    """弹丸出膛速度 [m/s]。这是**算出来的**, 用于和论文的 v_0 = 825 m/s 校核。"""

    startup_duration: float
    """定容燃烧阶段时长 [s]。"""

    p_peak: float
    """平均压力峰值 [Pa]。"""

    F_peak: float
    """膛底作用力峰值 [N]。"""

    J_breech: float
    """∫F_t dt [N*s], 膛底作用力冲量。"""

    J_target: float
    """自由后坐冲量 (m_b + omega/2)*v_0 [N*s], 用于校核。"""

    def summary(self) -> str:
        return (
            f"内弹道解算结果\n"
            f"  持续时间      : {self.duration * 1e3:.3f} ms"
            f"  (其中定容期 {self.startup_duration * 1e3:.3f} ms)\n"
            f"  弹丸行程      : {self.travel * 1e3:.1f} mm\n"
            f"  出膛速度      : {self.v_exit:.2f} m/s\n"
            f"  平均压力峰值  : {self.p_peak / 1e6:.1f} MPa\n"
            f"  膛底力峰值    : {self.F_peak / 1e3:.2f} kN\n"
            f"  ∫F_t dt       : {self.J_breech:.3f} N*s\n"
            f"  自由后坐冲量  : {self.J_target:.3f} N*s\n"
            f"  冲量相对偏差  : {(self.J_breech / self.J_target - 1.0) * 100:+.2f} %"
        )


# ---------------------------------------------------------------------------
# 状态方程辅助函数
# ---------------------------------------------------------------------------


def burnt_fraction(Z: float, p: InteriorBallisticsParams) -> float:
    """式 (2): psi = chi_g * Z * (1 + lam_g * Z)。"""
    return p.chi_g * Z * (1.0 + p.lam_g * Z)


def free_volume_length(psi: float, p: InteriorBallisticsParams) -> float:
    """式 (6): 药室自由容积缩径长 L_psi [m]。"""
    S = p.S()
    return (p.W_0 / S) * (
        1.0 - p.omega / (p.rho_g * p.W_0) - p.omega / p.W_0 * (p.alpha - 1.0 / p.rho_g) * psi
    )


def _state_pressure_startup(Z: float, p: InteriorBallisticsParams) -> float:
    """定容燃烧阶段由式 (5) 退化得到的压力 [Pa]。"""
    psi = burnt_fraction(Z, p)
    L_psi = free_volume_length(psi, p)
    if L_psi <= 0.0:
        raise RuntimeError(
            f"L_psi 在 Z={Z:.6f}, psi={psi:.6f} 处非正, 内弹道参数不自洽。"
        )
    return p.f_powder * p.omega * psi / (p.S() * L_psi)


def _solve_Z_for_pressure(p_target: float, p: InteriorBallisticsParams) -> float:
    """由定容期的 p(Z) 反解出对应 p_target 的 Z (二分法)。

    定容期 p 在 Z ∈ (0, 1] 上单调递增, 因此二分法可靠。
    """
    lo, hi = 1.0e-12, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _state_pressure_startup(mid, p) < p_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _deriv_motion(
    L: float, v: float, Z: float, p: InteriorBallisticsParams
) -> tuple[float, float, float]:
    """阶段 2 的右端项, 返回 (dL/dt, dv/dt, dZ/dt)。"""
    psi = burnt_fraction(Z, p)
    L_psi = free_volume_length(psi, p)
    denom = L_psi + L
    if denom <= 0.0:
        raise RuntimeError(f"自由容积 L_psi+L 在 L={L:.6f} 处非正。")

    theta = p.k_adiabatic - 1.0
    dvdt = (p.f_powder * p.omega * psi / (p.phi * p.m_b) - 0.5 * theta * v * v) / denom
    pressure = p.phi * p.m_b * dvdt / p.S()
    dZdt = pressure / p.I_K
    return v, dvdt, dZdt


# ---------------------------------------------------------------------------
# 主求解器
# ---------------------------------------------------------------------------


def solve_interior_ballistics(
    params: InteriorBallisticsParams,
    dt: float = 2.0e-7,
    p_ignition: float = 2.0e6,
    t_max: float = 2.0e-2,
) -> InteriorBallisticsResult:
    """求解内弹道方程组。

    Args:
        params: 内弹道参数。
        dt: 固定积分步长 [s]。
        p_ignition: 点火压力 [Pa]。定容燃烧阶段的初始条件, 表示击发瞬间
            膛内已经建立的压力 (底火燃气 + 引药)。

            式 (3) 在 Z=0 处是不动点 (dZ/dt = p/I_K = 0), 系统不会自发
            起动, 因此必须指定一个起点。用"点火压力"而不是"Z 的种子值"
            的原因: 由 psi ∝ Z 可知定容期 p 近似随 Z 指数增长, 定容期时长
            ∝ ln(p_0/p_ignition) —— 指定压力比指定 Z 更接近物理量, 且
            起步阶段的数值不确定性被收敛成一个人为可声明、可标定的参数。

            该参数的取值只影响击发到弹丸起动之间的"死时间", **不改变**
            主脉冲的形状、峰值与冲量 (主脉冲由 p_0 之后的变容阶段决定)。
        t_max: 最长积分时长 [s], 防止参数不自洽时死循环。

    Returns:
        InteriorBallisticsResult
    """
    S = params.S()
    p_0 = params.p_0
    v_0 = params.v_0

    # ---- 阶段 1: 定容燃烧 ----
    # 由 p(Z_start) = p_ignition 反解起始 Z。p(Z) 在该区间单调递增。
    Z = _solve_Z_for_pressure(p_ignition, params)
    t = 0.0
    L = 0.0
    v = 0.0

    ts: list[float] = []
    Ls: list[float] = []
    vs: list[float] = []
    ps: list[float] = []
    p_ts: list[float] = []
    psis: list[float] = []

    def _dZdt_startup(Z_: float) -> float:
        return _state_pressure_startup(Z_, params) / params.I_K

    while True:
        pressure = _state_pressure_startup(Z, params)
        psi = burnt_fraction(Z, params)
        ts.append(t)
        Ls.append(0.0)
        vs.append(0.0)
        ps.append(pressure)
        p_ts.append((1.0 + params.omega / (6.0 * params.phi * params.m_b)) * pressure)
        psis.append(psi)

        if pressure >= p_0:
            break
        if t > t_max:
            raise RuntimeError(f"定容燃烧阶段未在 {t_max} s 内达到起动压力 {p_0} Pa。")

        k1 = _dZdt_startup(Z)
        k2 = _dZdt_startup(Z + 0.5 * dt * k1)
        k3 = _dZdt_startup(Z + 0.5 * dt * k2)
        k4 = _dZdt_startup(Z + dt * k3)
        Z = Z + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt

    startup_duration = t

    # ---- 阶段 2: 变容燃烧 ----
    # 积分终止条件是弹丸行程达到枪管长度 L_barrel, 而不是 "初速达到 v_0"。
    # 物理因果是 "枪管长度 + 装药 -> 初速", 反过来把 v_0 当作终止条件会
    # 使 v_0 变成恒等式, 失去校核意义。v_0 在 `calibrate` 中作为目标值。
    L_barrel = params.L_barrel
    while L < L_barrel:
        if t > t_max:
            raise RuntimeError(
                f"弹丸行程在 {t_max} s 内未达到 L_barrel={L_barrel} m (当前 {L:.4f} m)。"
                "请检查 f_powder / I_K / W_0 是否自洽。"
            )

        d1 = _deriv_motion(L, v, Z, params)
        d2 = _deriv_motion(L + 0.5 * dt * d1[0], v + 0.5 * dt * d1[1], Z + 0.5 * dt * d1[2], params)
        d3 = _deriv_motion(L + 0.5 * dt * d2[0], v + 0.5 * dt * d2[1], Z + 0.5 * dt * d2[2], params)
        d4 = _deriv_motion(L + dt * d3[0], v + dt * d3[1], Z + dt * d3[2], params)

        L += dt / 6.0 * (d1[0] + 2.0 * d2[0] + 2.0 * d3[0] + d4[0])
        v += dt / 6.0 * (d1[1] + 2.0 * d2[1] + 2.0 * d3[1] + d4[1])
        Z += dt / 6.0 * (d1[2] + 2.0 * d2[2] + 2.0 * d3[2] + d4[2])
        t += dt

        psi = burnt_fraction(Z, params)
        pressure = params.phi * params.m_b * d1[1] / S

        ts.append(t)
        Ls.append(L)
        vs.append(v)
        ps.append(pressure)
        p_ts.append((1.0 + params.omega / (6.0 * params.phi * params.m_b)) * pressure)
        psis.append(psi)

    t_arr = np.asarray(ts)
    F_t = np.asarray(p_ts) * S

    J_breech = float(trapz(F_t, t_arr))
    J_target = float((params.m_b + 0.5 * params.omega) * v_0)

    return InteriorBallisticsResult(
        t=t_arr,
        L=np.asarray(Ls),
        v=np.asarray(vs),
        p=np.asarray(ps),
        p_t=np.asarray(p_ts),
        F_t=F_t,
        psi=np.asarray(psis),
        duration=float(t),
        travel=float(L),
        v_exit=float(v),
        startup_duration=float(startup_duration),
        p_peak=float(np.max(ps)),
        F_peak=float(np.max(F_t)),
        J_breech=J_breech,
        J_target=J_target,
    )


# ---------------------------------------------------------------------------
# 参数标定
# ---------------------------------------------------------------------------


def calibrate(
    params: InteriorBallisticsParams,
    target_F_peak: float = 19.0e3,
    p_ignition: float = 2.0e6,
    dt: float = 3.0e-7,
    tol: float = 2.0e-3,
    max_iter: int = 40,
    t_max: float = 5.0e-3,
    verbose: bool = False,
) -> InteriorBallisticsParams:
    """标定 `f_powder` 与 `I_K`, 使解算结果同时命中初速与膛底力峰值。

    为什么需要标定, 以及选哪两个约束
    ----------------------------------
    论文给出了 `v_0 = 825 m/s`、射速 750 发/分, 但**没有给出**火药力 `f`
    与压力全冲量 `I_K` —— 这两个参数决定曲线形状, 必须自行定住, 否则
    (v_0, 峰值) 同一点对应一族完全不同的 (f, I_K), 曲线形状无从复现。

    可用的独立约束:

    1. **v_0 = 825 m/s**                 —— 论文 1.1 节明确给出。
    2. **F_t 峰值 ≈ 19 kN**             —— 从论文图 6(a) 插图读出。
    3. **弹丸行程 ≈ 枪管长度 0.55 m**    —— 7.62 mm 机枪的几何硬约束,
       直接由 `params.L_barrel` 固定, 兼作积分的终止条件。

    用 (1)(2) 定住两个未知量, 于是**内弹道时长成为预测值**, 可以与
    论文图 6(a) 的脉冲宽度 (约 1.0-1.2 ms) 做独立交叉验证 —— 这是本
    标定流程唯一有意义的正确性检验。

    早先版本曾以 "内弹道时长 1.2 ms" 作为标定目标, 那会把插图上读出的
    量同时用作输入和验证, 循环论证; 且会在参数空间中切出一条斜线, 得到
    行程 1-2 m 的非物理解。

    标定方法
    --------
    嵌套二分, 两者都单调:

    - 内层: 固定 `I_K`, 二分 `f_powder` 使出膛速度 == `v_0` (递增)。
    - 外层: 二分 `I_K` 使 `F_t` 峰值 == `target_F_peak`。
      `I_K` 越大 -> 燃速越慢 -> 压力峰值越低 (递减)。

    Args:
        params: 待标定的参数 (不会被原地修改)。
        target_F_peak: 目标膛底作用力峰值 [N]。
        p_ignition: 点火压力 [Pa], 见 `solve_interior_ballistics`。
        dt: 标定用的积分步长 [s]。
        tol: 峰值的相对收敛容差。
        max_iter: 每层二分的最大迭代次数。
        t_max: 求解的最长积分时长 [s]。
        verbose: 是否打印迭代过程。

    Returns:
        标定后的新参数对象。
    """
    import copy

    p = copy.deepcopy(params)

    def _solve(f_powder: float) -> InteriorBallisticsResult | None:
        q = copy.deepcopy(p)
        q.f_powder = f_powder
        try:
            return solve_interior_ballistics(q, dt=dt, p_ignition=p_ignition, t_max=t_max)
        except RuntimeError:
            return None

    def _fit_f_powder() -> InteriorBallisticsResult | None:
        """内层: 固定 I_K, 二分 f_powder 命中出膛速度 v_0。"""
        lo, hi = 1.0e3, 5.0e7
        res: InteriorBallisticsResult | None = None
        for _ in range(max_iter):
            p.f_powder = 0.5 * (lo + hi)
            res = _solve(p.f_powder)
            if res is not None and abs(res.v_exit - p.v_0) < 1e-5 * p.v_0:
                return res
            if res is None or res.v_exit < p.v_0:
                lo = p.f_powder
            else:
                hi = p.f_powder
        return res

    lo, hi = 2.0e4, 3.0e5
    last_res: InteriorBallisticsResult | None = None
    for it in range(max_iter):
        p.I_K = 0.5 * (lo + hi)
        res = _fit_f_powder()
        if res is None:
            if verbose:
                print(f"  [calibrate] iter={it:02d} I_K={p.I_K:.4e} -> 不可解, 收紧上界")
            hi = p.I_K
            continue
        last_res = res
        rel = (res.F_peak - target_F_peak) / target_F_peak
        if verbose:
            print(
                f"  [calibrate] iter={it:02d} I_K={p.I_K:.4e} f={p.f_powder:.4e} "
                f"F_peak={res.F_peak / 1e3:7.2f} kN (rel={rel * 100:+.2f}%) "
                f"v_exit={res.v_exit:7.2f} dur={res.duration * 1e3:.3f} ms "
                f"travel={res.travel * 1e3:.0f} mm"
            )
        if abs(rel) < tol:
            break
        if res.F_peak > target_F_peak:
            lo = p.I_K
        else:
            hi = p.I_K

    if last_res is None:
        raise RuntimeError(
            f"无法标定: 在整个 I_K 搜索区间内均无法使弹丸达到 v_0={p.v_0} m/s。"
            "请检查 W_0 / omega / caliber / phi 是否自洽。"
        )
    return p


def sensitivity_to_ignition(
    params: InteriorBallisticsParams,
    p_ignitions: tuple[float, ...] = (1e6, 2e6, 5e6, 1e7),
    dt: float = 2.0e-7,
) -> dict[float, tuple[float, float, float]]:
    """检查解对点火压力 `p_ignition` 的敏感性。

    返回 {p_ignition: (总时长 [s], 主脉冲宽度 [s], 峰值 [N])}。

    预期行为: 总时长随 ln(p_0/p_ignition) 变化, 但**主脉冲形状、峰值与
    冲量不变** —— 后者才是仿真真正关心的量。
    """
    out = {}
    for pi in p_ignitions:
        res = solve_interior_ballistics(params, dt=dt, p_ignition=pi)
        # 主脉冲宽度取 F_t 超过峰值 1% 的时间跨度
        thr = 0.01 * res.F_peak
        idx = np.nonzero(res.F_t >= thr)[0]
        width = float(res.t[idx[-1]] - res.t[idx[0]]) if len(idx) > 1 else 0.0
        out[pi] = (res.duration, width, res.F_peak)
    return out
