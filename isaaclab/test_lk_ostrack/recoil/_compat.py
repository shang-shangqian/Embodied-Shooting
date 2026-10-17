"""版本兼容辅助。

numpy 2.0 把 `np.trapz` 改名为 `np.trapezoid` (旧名保留但会告警);
本项目的其他模块 (`test11/speed_prediction/_compat.py`) 也做同样处理。
"""

from __future__ import annotations

import numpy as np

if hasattr(np, "trapezoid"):  # numpy >= 2.0
    trapz = np.trapezoid
else:  # numpy 1.x
    trapz = np.trapz

__all__ = ["trapz"]
