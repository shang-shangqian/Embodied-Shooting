from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator
import os
import time
import sys

from .camera_geometry import bbox_clip


@dataclass
class TrackedBox:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int] | None
    score: float | None = None
    ok: bool = True
    track_time_s: float | None = None

    @property
    def bbox_area(self) -> int:
        if self.bbox_xyxy is None:
            return 0
        x1, y1, x2, y2 = self.bbox_xyxy
        return int(max(0, x2 - x1) * max(0, y2 - y1))


def _xyxy_to_xywh(box_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def _xywh_to_xyxy(box_xywh: Iterable[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + max(1.0, w), y + max(1.0, h)]


def _read_rgb(path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to read frames for OSTrack tracking. "
            "Install a cv2 build compatible with the active NumPy version."
        ) from exc

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _set_torch_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but official OSTrack uses CUDA in its tracker class.")
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
    except ImportError as exc:
        raise ImportError("OSTrack requires PyTorch. Install the OSTrack environment first.") from exc


class OSTrackBoxTracker:
    """Adapter for the official botaoye/OSTrack tracker.

    The official tracker expects an OSTrack repository on PYTHONPATH, a yaml
    config name/path, and a checkpoint. This adapter keeps those dependencies
    outside this project and exposes a small iterator returning xyxy boxes.
    """

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint: str | Path,
        config: str = "vitb_256_mae_ce_32x4_ep300",
        device: str = "cuda:0",
        dataset_name: str = "custom",
        debug: int = 0,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.config = str(config)
        self.device = str(device)
        self.dataset_name = str(dataset_name)
        self.debug = int(debug)
        self._initial_cwd = Path.cwd()

    def _resolve_yaml(self) -> Path:
        config_path = Path(self.config)
        if config_path.exists():
            return config_path.resolve()
        cwd_yaml_path = self._initial_cwd / self.config
        if cwd_yaml_path.exists():
            return cwd_yaml_path.resolve()
        repo_relative_path = self.repo_path / self.config
        if repo_relative_path.exists():
            return repo_relative_path.resolve()
        yaml_path = self.repo_path / "experiments" / "ostrack" / f"{self.config}.yaml"
        if yaml_path.exists():
            return yaml_path
        raise FileNotFoundError(
            "Could not find OSTrack config yaml. Pass --ostrack_config as a file path "
            f"or a name under {self.repo_path / 'experiments' / 'ostrack'}."
        )

    def _build_params(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(f"OSTrack repo not found: {self.repo_path}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"OSTrack checkpoint not found: {self.checkpoint}")
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        _set_torch_device(self.device)

        from lib.config.ostrack.config import cfg, update_config_from_file

        yaml_path = self._resolve_yaml()
        update_config_from_file(str(yaml_path))
        params = SimpleNamespace()
        params.cfg = cfg
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.checkpoint = str(self.checkpoint)
        params.save_all_boxes = False
        params.debug = self.debug
        return params

    def _build_tracker(self):
        params = self._build_params()
        from lib.test.tracker.ostrack import get_tracker_class

        tracker_class = get_tracker_class()
        return tracker_class(params, self.dataset_name)

    def track(
        self,
        rgb_paths: list[Path],
        initial_box_xyxy: Iterable[float],
    ) -> Iterator[TrackedBox]:
        if not rgb_paths:
            return
        rgb_paths = [Path(path).resolve() for path in rgb_paths]

        old_cwd = Path.cwd()
        os.chdir(str(self.repo_path))
        try:
            tracker = self._build_tracker()
            first_image = _read_rgb(rgb_paths[0])
            h, w = first_image.shape[:2]
            initial_xyxy = bbox_clip(initial_box_xyxy, width=w, height=h)
            init_start = time.perf_counter()
            tracker.initialize(first_image, {"init_bbox": _xyxy_to_xywh(initial_xyxy)})
            init_time_s = time.perf_counter() - init_start
            yield TrackedBox(
                frame_index=0,
                bbox_xyxy=initial_xyxy,
                score=None,
                ok=True,
                track_time_s=init_time_s,
            )

            for frame_index, rgb_path in enumerate(rgb_paths[1:], start=1):
                image = _read_rgb(rgb_path)
                h, w = image.shape[:2]
                track_start = time.perf_counter()
                out = tracker.track(image, {})
                track_time_s = time.perf_counter() - track_start
                bbox_xywh = out.get("target_bbox")
                if bbox_xywh is None:
                    yield TrackedBox(
                        frame_index=frame_index,
                        bbox_xyxy=None,
                        ok=False,
                        track_time_s=track_time_s,
                    )
                    continue
                bbox_xyxy = bbox_clip(_xywh_to_xyxy(bbox_xywh), width=w, height=h)
                score = out.get("best_score", out.get("target_score"))
                yield TrackedBox(
                    frame_index=frame_index,
                    bbox_xyxy=bbox_xyxy,
                    score=None if score is None else float(score),
                    ok=True,
                    track_time_s=track_time_s,
                )
        finally:
            os.chdir(str(old_cwd))
