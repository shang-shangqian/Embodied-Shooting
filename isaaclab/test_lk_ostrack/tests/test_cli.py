from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from test_lk_ostrack.camera_geometry import CameraIntrinsics
from test_lk_ostrack.estimate_video import run_multi_target


def test_demo_runs_real_cli_from_another_directory(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    subprocess.run(
        [sys.executable, "-m", "test_lk_ostrack.demo", "--frames", "6", "--out_dir", str(output)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    with (output / "tracks.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert sum(int(row["selected"]) for row in rows) == 6
    assert len(list((output / "overlay").glob("*"))) == 6
    assert {int(row["target_id"]) for row in rows} == {0, 1}


def test_missing_ostrack_dependency_fails_before_tracking(tmp_path: Path) -> None:
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"targets": [{"id": 0, "bbox": [5, 5, 25, 25]}]}))
    args = SimpleNamespace(initial_boxes_file=initial, disable_fine_tracking=False,
                           ostrack_repo=tmp_path, ostrack_checkpoint=tmp_path / "model.pth.tar",
                           ostrack_config="default", device="cuda:0", debug=0)
    fine = Mock()
    fine.prepare.side_effect = ImportError("missing optional dependency")
    with patch("test_lk_ostrack.estimate_video.OSTrackFrameTracker", return_value=fine):
        with pytest.raises(ImportError, match="missing optional dependency"):
            run_multi_target(args, CameraIntrinsics(100, 100, 32, 32, 64, 64), [])
    fine.initialize.assert_not_called()
