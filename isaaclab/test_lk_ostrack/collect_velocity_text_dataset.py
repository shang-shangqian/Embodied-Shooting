"""Collect compact primary-target sequences for the EKF/GRU velocity module.

RGB frames are used by LK/OSTrack in memory but are not saved. The robot is
commanded to stand still, frames with excessive measured base motion are
rejected, and each completed sequence contains only one stable primary target.
"""

from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_lk_ostrack.collect_multi_target_dataset import main


if __name__ == "__main__":
    main(
        force_text_only=True,
        force_static_robot=True,
        primary_target_only=True,
    )
