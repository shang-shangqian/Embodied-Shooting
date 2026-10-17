"""后坐力载荷模型。

参考文献: 陆星宇 等. 四足打击平台多工况力学特性及射击扰动.
工程科学学报, 2024, 46(8): 1434-1445.

模块结构
--------
- `params`              : 武器/内弹道/导气室/撞击参数, 载荷合成模式
- `interior_ballistics` : 内弹道方程组 (论文式 2-9) 的 RK4 求解器
- `loads`               : 四类载荷时程合成 (式 9-13) + 冲量守恒自检
- `injection`           : 把载荷注入 Isaac Lab articulation
- `integration`         : 与 `test11/turret_aim_env.py` 的接线清单
- `validate`            : 离线验证脚本 (不需要 Isaac Sim)

快速开始
--------
    from recoil import WeaponParams, generate_recoil_loads, MODE_FULL_CYCLE

    weapon = WeaponParams()
    series = generate_recoil_loads(weapon, mode=MODE_FULL_CYCLE, n_cycles=5)
    print(series.audit())
"""

from .interior_ballistics import (
    InteriorBallisticsResult,
    calibrate,
    sensitivity_to_ignition,
    solve_interior_ballistics,
)
from .loads import (
    RecoilLoadSeries,
    generate_recoil_loads,
    gas_chamber_force,
    impact_force,
    to_step_impulse,
)
from .params import (
    MODE_BREECH_ONLY,
    MODE_FULL_CYCLE,
    MODE_NAIVE_ALL_POSITIVE,
    MODE_WITH_GAS_ON_PISTON,
    VALID_MODES,
    GasChamberParams,
    ImpactParams,
    InteriorBallisticsParams,
    RecoilLoadCfg,
    WeaponParams,
)

# `injection` / `integration` 依赖 torch 与 Isaac Lab, 而 `params` 以上的模块
# 只需 numpy。后坐载荷的离线生成与验证 (见 `validate.py`) 不应被迫装一整套
# Isaac Sim, 因此把硬件相关的导入做成可选的: 有 torch 时正常导出, 没有时
# 仅跳过这两个名字, 其余功能不受影响。
try:
    from .injection import RecoilInjector, RecoilInjectorCfg, make_injector
    from .integration import build_injector

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 取决于运行环境是否装了 torch
    _HAS_TORCH = False

__all__ = [
    "MODE_BREECH_ONLY",
    "MODE_FULL_CYCLE",
    "MODE_NAIVE_ALL_POSITIVE",
    "MODE_WITH_GAS_ON_PISTON",
    "VALID_MODES",
    "GasChamberParams",
    "ImpactParams",
    "InteriorBallisticsParams",
    "InteriorBallisticsResult",
    "RecoilLoadCfg",
    "RecoilLoadSeries",
    "WeaponParams",
    "calibrate",
    "gas_chamber_force",
    "generate_recoil_loads",
    "impact_force",
    "sensitivity_to_ignition",
    "solve_interior_ballistics",
    "to_step_impulse",
]

if _HAS_TORCH:
    __all__ += [
        "RecoilInjector",
        "RecoilInjectorCfg",
        "build_injector",
        "make_injector",
    ]
