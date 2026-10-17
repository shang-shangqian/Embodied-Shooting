"""与 `test11/turret_aim_env.py` 的集成。

本模块不修改 `TurretAimEnv` 本身, 而是提供一个可选的 Mixin 与一段
最小改动清单, 便于独立开关后坐力。

接入点 (三个)
-------------
1. `TurretAimEnvCfg` 增加 `enable_recoil` 等字段 (见 `RecoilEnvCfgMixin`)。
2. `TurretAimEnv.__init__` 末尾: 创建注入器。
3. `TurretAimEnv._apply_action` 末尾: `self._recoil_injector.apply()`。
4. `TurretAimEnv._reset_idx` 末尾: `self._recoil_injector.reset(env_ids)`。
5. 开火处 (`_fire_command` 置位之后): `self._recoil_injector.trigger(...)`。

为什么必须在 `_apply_action` 里施加
-----------------------------------
Isaac Lab 的 `DirectRLEnv.step()` 结构是:

    _pre_physics_step(actions)
    for _ in range(decimation):
        _apply_action()          <-- 在这里写外部力
        scene.write_data_to_sim() <-- 外部力在这里进 PhysX
        sim.step()
    scene.update(dt)

`set_external_force_and_torque` 只是填缓冲区, 真正的写入发生在
`write_data_to_sim()`。因此必须在 `_apply_action()` 内调用 `apply()`,
每物理步一次 (共 decimation 次), 而不是每 env step 一次。

注意: 本项目的自研控制器需确认 `decimation` 循环确实调用了
`_apply_action`, 而不是用 `env.step()` 的默认实现。当前
`TurretAimEnv` 未覆写 `step()`/`_apply_action` 的调用次数逻辑,
`_apply_action` 每个物理步执行一次, 与上述一致。
"""

from __future__ import annotations

import torch

from .injection import RecoilInjector, RecoilInjectorCfg
from .loads import generate_recoil_loads
from .params import MODE_FULL_CYCLE, RecoilLoadCfg


def build_injector(env, cfg: RecoilLoadCfg | None = None, mode: str = MODE_FULL_CYCLE) -> RecoilInjector:
    """为 `TurretAimEnv` 实例构造注入器。

    物理步长直接取 `env.physics_dt` (即 `sim.dt`), 保证与仿真一致 —— 这点
    很关键, 因为冲量守恒依赖 `F_step * dt` 的积分关系。

    Args:
        env: `TurretAimEnv` 实例 (需已执行 `super().__init__`)。
        cfg: 载荷配置; 默认五连发、full_cycle 模式。
        mode: 载荷合成模式。

    Returns:
        已绑定到 `env.turret` 的 `RecoilInjector`。
    """
    cfg = cfg or RecoilLoadCfg()
    series = generate_recoil_loads(
        cfg.weapon,
        mode=mode,
        n_cycles=cfg.n_cycles,
        dt_fine=cfg.dt_fine,
        dt_phys=cfg.dt_phys,
    )
    return RecoilInjector(
        asset=env.turret,
        cfg=RecoilInjectorCfg(
            body_name_expr=env.cfg.muzzle_body_name_expr,
            body_local_force_axis=tuple(float(a) for a in env.cfg.muzzle_forward_axis),
        ),
        series=series,
        dt_phys=float(env.physics_dt),
    )


# ---------------------------------------------------------------------------
# 最小改动清单 (供手工 patch turret_aim_env.py 时对照)
# ---------------------------------------------------------------------------

INTEGRATION_SNIPPET = '''
# ---------------------------------------------------------------------------
# 1) TurretAimEnvCfg 内新增字段
# ---------------------------------------------------------------------------
    enable_recoil: bool = False
    """是否启用后坐力载荷。默认关闭, 保证既有实验可复现。"""

    recoil_mode: str = "full_cycle"
    """载荷模式: breech_only | full_cycle | with_gas_on_piston | naive_all_positive"""

    recoil_n_cycles: int = 5
    """连发发数。"""

    recoil_dt_fine: float = 2.0e-6
    """离线载荷生成用的细网格步长 [s]。"""


# ---------------------------------------------------------------------------
# 2) TurretAimEnv.__init__ 开头 (super().__init__ 之前)
# ---------------------------------------------------------------------------
        # 先置 None, 使 _reset_idx / _apply_action 在其存在前也能安全调用。
        self._recoil_injector = None


# ---------------------------------------------------------------------------
# 3) 懒构造 getter (新增方法)
# ---------------------------------------------------------------------------
    def _get_recoil_injector(self):
        """按需构造; 未启用时返回 None。

        不能在 _setup_scene 里构造 —— 那时 articulation 的 PhysX view 还没
        initialize, asset.device / body_names 都取不到。
        """
        if not self.cfg.enable_recoil:
            return None
        if self._recoil_injector is None:
            self._recoil_injector = self._build_recoil_injector()
            print(self._recoil_injector.summary())
            print(self._recoil_injector.series.audit())
        return self._recoil_injector


# ---------------------------------------------------------------------------
# 4) _pre_physics_step 内, self._fire_command = fire_now.float() 之后
# ---------------------------------------------------------------------------
        recoil = self._get_recoil_injector()
        if recoil is not None:
            # 每一发触发一个完整的自动机循环 (膛底 + 后坐撞击 + 复进撞击)
            recoil.trigger(torch.nonzero(fire_now, as_tuple=False).squeeze(-1))


# ---------------------------------------------------------------------------
# 5) _apply_action 末尾 (必须在这里: 每个物理步一次, 且在 write_data_to_sim 之前)
# ---------------------------------------------------------------------------
        recoil = self._get_recoil_injector()
        if recoil is not None:
            recoil.apply()


# ---------------------------------------------------------------------------
# 6) _reset_idx 末尾
# ---------------------------------------------------------------------------
        recoil = self._get_recoil_injector()
        if recoil is not None:
            recoil.reset(env_ids_t)
'''


def write_integration_snippet(path: str) -> None:
    """把集成清单写到文件, 便于对照修改 `turret_aim_env.py`。"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("```python\n")
        fh.write(INTEGRATION_SNIPPET.strip())
        fh.write("\n```\n")
