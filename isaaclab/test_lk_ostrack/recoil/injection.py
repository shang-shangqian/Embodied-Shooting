"""把后坐载荷注入 Isaac Lab articulation。

施加对象与坐标系
----------------
按论文 1.2 节的基本假设, 载荷施加在武器侧而非平台侧。在本项目的
`lite3_with_turret` 资产里, 武器对应 `pitch_link` (参见
`TurretAimEnvCfg.muzzle_body_name_expr`)。

坐标系取 ** тела局部系**, 而不是世界系:

- `pitch_link` 局部 +x 轴就是膛线指向 (`muzzle_forward_axis = (1,0,0)`)。
- 后坐载荷沿膛线方向且反向, 因此局部力就是 `(-|F|, 0, 0)`。
- 用局部系的好处是俯仰关节转动后力矢量自动跟着转, 无需每步重算旋转矩阵。

不施加力矩
----------
`turret.urdf` 中炮管可视化体的 origin 是 `xyz="0.045 0 0"`, **z 偏移为 0**,
即膛线轴穿过俯仰关节轴线与 `pitch_link` 质心。因此

    tau = r x F,  r = (0.045, 0, 0),  F = (-|F|, 0, 0)  =>  tau = 0

后坐力对 `pitch_link` 的力矩恒为零, 平台响应的主导项是整体后仰 (力作用在
机身质心上方产生俯仰力矩) 与腿部的支撑反力。这与"云台伺服不会被打饱和"
的结论一致 (见模块 `README.md` 的讨论)。

若后续把炮管抬高到俯仰轴上方 (真实武器站的常见布局), 需要在此处补上
`torques` 项。

注入时机
--------
必须在 `InteractiveScene.write_data_to_sim()` **之前** 调用 `apply()`,
本项目的调用位置是 `TurretAimEnv._apply_action()` (每个物理步执行一次,
共 `decimation` 次)。外部力只是写进缓冲区, 真正进 PhysX 发生在
`write_data_to_sim()`。

写入接口
--------
Isaac Lab >= 2.3 用 `asset.permanent_wrench_composer.set_forces_and_torques`
承载外部力; 旧的 `Articulation.set_external_force_and_torque` 只是它的
转发包装, 且每次调用都会触发一条废弃警告 (rate-limit 到每 5 s 一次)。
本模块优先走新接口, 旧版 Isaac Lab 自动回退到旧接口。

注意 `permanent_wrench_composer` 的语义是"每步都施加", 与环境 `reset()`
无关: 序列结束后 `_write_forces` 仍然每个物理步写入零力, 因此**必须**
保持每步调用 `apply()`, 不能"没在射击就跳过"。

冲量守恒
--------
`apply()` 使用的是 `loads.to_step_impulse()` 的输出, 即**步内等效力**
`F_step = (1/dt) * ∫F dt`。无论脉冲比 `dt` 短多少, 总冲量都严格守恒。
不要改成逐点取值 —— dt=5 ms 下 1.4 ms 的膛底脉冲会落在单步之内, 逐点
取值会把真实冲量放大到 4 倍以上。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .loads import RecoilLoadSeries, to_step_impulse


@dataclass
class RecoilInjectorCfg:
    """注入器配置。"""

    body_name_expr: str = "pitch_link"
    """承受后坐载荷的刚体名 (支持正则)。论文假设 (2)(3)(4) 中"枪体/枪机"在本
    资产里对应 `pitch_link`。"""

    body_local_force_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """膛线方向在该刚体局部坐标系下的单位向量。载荷按此方向取负施加。

    `TurretAimEnvCfg.muzzle_forward_axis` 同为 (1,0,0), 二者必须一致。"""

    auto_fire: bool = False
    """是否自动按射速循环射击 (离线标定/自检用)。生产训练中应保持 False,
    由环境的开火逻辑显式调用 `trigger()`。"""

    intraburst_collapse: bool = True
    """是否把同一物理步内的多发合并成一个等效冲量。

    射速 750 发/分 -> 每发间隔 80 ms, 远大于常见的 5 ms 物理步, 因此默认
    情况下不需要合并。开启后即使把物理步放到 100 ms 也不会丢冲量。
    """


class RecoilInjector:
    """把一个连发序列的逐步等效力施加到指定 articulation 的刚体上。

    典型用法 (在 `TurretAimEnv._apply_action` 中)::

        self._recoil = RecoilInjector(self.turret, self.cfg_recoil, series)

        # 开火时
        self._recoil.trigger(fire_env_ids)

        # 每个物理步
        self._recoil.apply()

    Args:
        asset: 目标 `Articulation`。
        cfg: 注入器配置。
        series: `loads.generate_recoil_loads` 的输出。
        dt_phys: 物理步长 [s]。必须与环境 `sim.dt` 一致。
    """

    def __init__(
        self,
        asset,
        cfg: RecoilInjectorCfg,
        series: RecoilLoadSeries,
        dt_phys: float,
    ) -> None:
        self._asset = asset
        self.cfg = cfg
        self.series = series
        self.dt_phys = float(dt_phys)

        # 逐步等效力 (n_steps,), 已按 +x=射击方向 的符号约定
        self._f_step = torch.tensor(
            to_step_impulse(series, self.dt_phys),
            dtype=torch.float32,
            device=asset.device,
        )
        self.n_steps = int(self._f_step.numel())

        # 解算承受载荷的刚体
        body_ids, body_names = asset.find_bodies(cfg.body_name_expr)
        if len(body_ids) != 1:
            raise RuntimeError(
                f"body_name_expr={cfg.body_name_expr!r} 应恰好匹配 1 个刚体, "
                f"实际匹配 {len(body_names)}: {body_names}. "
                f"可用刚体: {asset.body_names}"
            )
        self.body_id = int(body_ids[0])
        self.body_name = body_names[0]

        # 局部系下的力方向单位向量
        axis = torch.tensor(cfg.body_local_force_axis, dtype=torch.float32, device=asset.device)
        self._axis = axis / (torch.norm(axis) + 1e-9)

        self.num_envs = int(getattr(asset, "num_instances", 1))
        self._idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=asset.device)
        """每个环境当前的步索引; -1 表示未在射击。"""

        self._burst_impulse_log: list[float] = []
        self._total_impulse = torch.zeros(self.num_envs, dtype=torch.float64, device=asset.device)

        # 预分配缓冲, 避免每步新建张量
        self._forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=asset.device)
        self._torques = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=asset.device)
        self._body_ids_t = [self.body_id]

        # Isaac Lab >= 2.3 用 `permanent_wrench_composer.set_forces_and_torques`
        # 承载外部力; 旧的 `set_external_force_and_torque` 只是它的转发包装,
        # 且每次调用都会发一条废弃警告 (每 5 s 一次)。直接调新接口既无警告,
        # 也避免每步多做一次 warp 转换。
        self._use_wrench_composer = hasattr(asset, "permanent_wrench_composer")

    # ------------------------------------------------------------------
    # 触发 / 重置
    # ------------------------------------------------------------------

    def trigger(self, env_ids: torch.Tensor | None = None, *, restart: bool = True) -> None:
        """开始 (或重新开始) 指定环境的射击序列。

        Args:
            env_ids: 环境索引张量; None 表示全部环境。
            restart: 若某环境正在射击, 是否从头重新开始。False 时忽略已在
                射击中的环境 (避免连发指令打断进行中的序列)。
        """
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self._asset.device)
        else:
            ids = env_ids.to(device=self._asset.device, dtype=torch.long).reshape(-1)
        if restart:
            self._idx[ids] = 0
        else:
            fresh = ids[self._idx[ids] < 0]
            self._idx[fresh] = 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """清空射击状态 (环境 reset 时调用)。"""
        if env_ids is None:
            self._idx.fill_(-1)
            self._total_impulse.zero_()
        else:
            ids = env_ids.to(device=self._asset.device, dtype=torch.long).reshape(-1)
            self._idx[ids] = -1
            self._total_impulse[ids] = 0.0

    def disable(self) -> None:
        """立即停止所有载荷并清零外部力缓冲。"""
        self._idx.fill_(-1)
        self._write_forces(torch.zeros(self.num_envs, device=self._asset.device))

    @property
    def active(self) -> torch.Tensor:
        """形状 (num_envs,) 的 bool 张量, 标记哪些环境正在射击。"""
        return self._idx >= 0

    # ------------------------------------------------------------------
    # 每步施加
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """推进一个物理步并写入外部力。

        必须在 `scene.write_data_to_sim()` 之前调用。
        """
        f_now = torch.zeros(self.num_envs, device=self._asset.device)

        active = torch.nonzero(self._idx >= 0, as_tuple=False).squeeze(-1)
        if active.numel() > 0:
            idx = self._idx[active]

            # 越界的环境本步结束射击, 本步不再施加载荷
            done = idx >= self.n_steps
            self._idx[active[done]] = -1

            ids = active[~done]
            idx = idx[~done]
            if ids.numel() > 0:
                f_now[ids] = self._f_step[idx]
                # 冲量累计仅用于自检: 只统计仍在序列内的那些步
                self._total_impulse[ids] += (f_now[ids] * self.dt_phys).double()
                self._idx[ids] += 1

        self._write_forces(f_now)

    def _write_forces(self, f_scalar: torch.Tensor) -> None:
        """把沿膛线方向的标量载荷写成 (num_envs, 1, 3) 的局部力张量。"""
        self._forces[:, 0, :] = self._axis.unsqueeze(0) * f_scalar.unsqueeze(-1)
        # 力矩为零: 见模块 docstring 的推导
        self._torques.zero_()
        if self._use_wrench_composer:
            self._asset.permanent_wrench_composer.set_forces_and_torques(
                forces=self._forces,
                torques=self._torques,
                body_ids=self._body_ids_t,
            )
        else:  # pragma: no cover - 旧版 Isaac Lab 兼容路径
            self._asset.set_external_force_and_torque(
                forces=self._forces,
                torques=self._torques,
                body_ids=self._body_ids_t,
            )

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def summary(self) -> str:
        per_shot = self.series.J_reference
        n = len(self.series.t_fire)
        return (
            "RecoilInjector\n"
            f"  承受载荷刚体    : {self.body_name} (body_id={self.body_id})\n"
            f"  腔线局部方向    : {tuple(round(float(a), 4) for a in self._axis)}\n"
            f"  物理步长        : {self.dt_phys * 1e3:.3f} ms\n"
            f"  序列长度        : {self.n_steps} 步 ({self.n_steps * self.dt_phys * 1e3:.1f} ms)\n"
            f"  每发理论冲量    : {per_shot:+.4f} N*s ({n} 发)\n"
            f"  逐步等效力峰值  : {float(torch.max(torch.abs(self._f_step))) / 1e3:.2f} kN\n"
            f"    注意: 该峰值受物理步长影响, 不是真实瞬时峰值。\n"
            f"          dt={self.dt_phys * 1e3:.1f} ms 无法分辨 1.4 ms 的膛底脉冲, \n"
            f"          只有总冲量是可信的。"
        )


# ---------------------------------------------------------------------------
# 便捷构造
# ---------------------------------------------------------------------------


def make_injector(
    asset,
    series: RecoilLoadSeries,
    dt_phys: float,
    *,
    body_name_expr: str = "pitch_link",
    body_local_force_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> RecoilInjector:
    """用默认配置构造 `RecoilInjector`。"""
    cfg = RecoilInjectorCfg(
        body_name_expr=body_name_expr,
        body_local_force_axis=body_local_force_axis,
    )
    return RecoilInjector(asset, cfg, series, dt_phys)
