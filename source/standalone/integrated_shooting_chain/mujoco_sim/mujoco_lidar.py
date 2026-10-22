"""MuJoCo array LiDAR with the test7_2_weight scan and range-estimation contract.

No Isaac Sim, Warp, Torch or renderer is required. Distances are metres and
angles passed to estimate_target are radians. The camera rotation maps ROS
optical coordinates (right, down, forward) into world coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np


@dataclass(frozen=True)
class ArrayLidarConfig:
    parent_body: str = 'TORSO'
    offset_position: tuple[float, float, float] = (0.23, 0.0, 0.073)
    offset_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    horizontal_fov_deg: float = 180.0
    vertical_fov_deg: float = 40.0
    horizontal_resolution_deg: float = 0.2
    vertical_resolution_deg: float = 0.3
    min_range: float = 0.3
    max_range: float = 165.0
    target_geom_group: int = 4
    postprocess_reference: str = 'parent'

    def __post_init__(self):
        values = [self.horizontal_fov_deg, self.vertical_fov_deg,
                  self.horizontal_resolution_deg, self.vertical_resolution_deg,
                  self.min_range, self.max_range]
        if not np.isfinite(values).all():
            raise ValueError('LiDAR angle/range settings must be finite.')
        if not 0 <= self.horizontal_fov_deg <= 360 or not 0 <= self.vertical_fov_deg <= 180:
            raise ValueError('Invalid horizontal or vertical field of view.')
        if self.horizontal_resolution_deg <= 0 or self.vertical_resolution_deg <= 0:
            raise ValueError('Angular resolution must be positive.')
        if not 0 <= self.min_range < self.max_range:
            raise ValueError('Require 0 <= min_range < max_range.')
        if not isinstance(self.target_geom_group, int) or not 0 <= self.target_geom_group < 6:
            raise ValueError('target_geom_group must be an integer in [0, 5].')
        if self.postprocess_reference not in ('parent', 'lidar'):
            raise ValueError('postprocess_reference must be parent or lidar.')
        if np.shape(self.offset_position) != (3,) or not np.isfinite(self.offset_position).all():
            raise ValueError('offset_position must contain three finite values.')
        q = np.asarray(self.offset_quaternion_wxyz)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError('Invalid mount quaternion (expected wxyz).')

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')


def lidar_directions(config):
    """Match IsaacLab lidar_pattern: inclusive linspace, vertical-major order.

    Only a full 360-degree horizontal scan omits its duplicate last column.
    Calculations use float64 rather than IsaacLab's float32.
    """
    channels = max(1, int(round(config.vertical_fov_deg / config.vertical_resolution_deg)) + 1)
    columns = math.ceil(config.horizontal_fov_deg / config.horizontal_resolution_deg) + 1
    vertical = np.deg2rad(np.linspace(-config.vertical_fov_deg / 2, config.vertical_fov_deg / 2, channels))
    horizontal = np.deg2rad(np.linspace(-config.horizontal_fov_deg / 2, config.horizontal_fov_deg / 2, columns))
    if abs(config.horizontal_fov_deg - 360) < 1e-6:
        horizontal = horizontal[:-1]
    v, h = np.meshgrid(vertical, horizontal, indexing='ij')
    directions = np.stack([np.cos(v) * np.cos(h), np.cos(v) * np.sin(h), np.sin(v)], axis=-1)
    directions = np.ascontiguousarray(directions.reshape(-1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions, (len(vertical), len(horizontal))


@dataclass(frozen=True)
class ArrayLidarScan:
    time: float
    origin_w: np.ndarray
    rotation_w: np.ndarray
    parent_position_w: np.ndarray
    directions_w: np.ndarray
    distances: np.ndarray
    geom_ids: np.ndarray
    ray_hits_w: np.ndarray
    valid: np.ndarray
    grid_shape: tuple[int, int]
    scan_seconds: float

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time=self.time, origin_w=self.origin_w,
                            rotation_w=self.rotation_w, parent_position_w=self.parent_position_w,
                            directions_w=self.directions_w,
                            distances=self.distances, geom_ids=self.geom_ids,
                            ray_hits_w=self.ray_hits_w, valid=self.valid, grid_shape=self.grid_shape)


@dataclass(frozen=True)
class TargetRangeEstimate:
    """surface_distance uses config.postprocess_reference; center_distance uses the camera."""
    hit_valid: bool
    center_distance: float
    surface_distance: float
    refined_yaw: float
    refined_pitch: float
    selected_count: int
    estimated_center_w: np.ndarray | None


class MujocoArrayLidar:
    """A fixed ray array carried by one body, scanning an explicit geom group.

    Put all desired ray surfaces in target_geom_group. By default only group 4
    is queried, matching the source's target-only mesh list: other objects do
    not occlude targets. Geom contact masks do not enable/disable ray detection.
    A fresh scan is returned each call; no cached target poses or truth fallback.
    Use separate MjData instances for independent environments.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or ArrayLidarConfig()
        self.body_id = model.body(self.config.parent_body).id
        self.directions_local, self.grid_shape = lidar_directions(self.config)
        self.num_rays = len(self.directions_local)
        self.geom_group_mask = np.zeros(6, dtype=np.uint8)
        self.geom_group_mask[self.config.target_geom_group] = 1
        if not np.any(model.geom_group == self.config.target_geom_group):
            raise ValueError(f'No ray target geoms in group {self.config.target_geom_group}.')
        q = np.array(self.config.offset_quaternion_wxyz, dtype=float, copy=True)
        q /= np.linalg.norm(q)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, q)
        self.offset_rotation = matrix.reshape(3, 3)

    def scan(self, data, *, update_kinematics=True):
        """Scan the current pose; call after mj_forward/mj_step if skipping update.

        A first surface outside the near/far range is invalid; near clipping does
        not search through that surface for a more distant return. Misses have
        distance/hit = inf and geom_id = -1, matching the Isaac-side hit buffers.
        """
        start = time.perf_counter()
        if update_kinematics:
            mujoco.mj_forward(self.model, data)
        parent_rotation = data.xmat[self.body_id].reshape(3, 3)
        rotation = parent_rotation @ self.offset_rotation
        origin = data.xpos[self.body_id] + parent_rotation @ np.asarray(self.config.offset_position)
        directions = np.ascontiguousarray(self.directions_local @ rotation.T)
        distances = np.empty(self.num_rays)
        geom_ids = np.empty(self.num_rays, dtype=np.int32)
        mujoco.mj_multiRay(self.model, data, origin, directions.ravel(), self.geom_group_mask,
                          True, -1, geom_ids, distances, None, self.num_rays, self.config.max_range)
        valid = (np.isfinite(distances) & (distances >= self.config.min_range)
                 & (distances <= self.config.max_range) & (geom_ids >= 0))
        distances[~valid], geom_ids[~valid] = np.inf, -1
        hits = np.full((self.num_rays, 3), np.inf)
        hits[valid] = origin + directions[valid] * distances[valid, None]
        return ArrayLidarScan(float(data.time), origin.copy(), rotation.copy(), data.xpos[self.body_id].copy(), directions,
                              distances, geom_ids, hits, valid, self.grid_shape, time.perf_counter() - start)

    def estimate_target(self, scan, *, yaw, pitch, camera_position_w,
                        camera_rotation_w, target_radius, target_geom_id=None):
        """Port the source cone -> lower median -> cluster -> mean -> centre estimate.

        yaw/pitch are externally supplied visual estimates, in radians. The camera
        rotation uses ROS optical axes, NOT MuJoCo camera axes. target_geom_id can
        select one already-visible target; it does not see through other targets.
        No target centre or ground-truth bearing is accepted by this interface.
        By default cone and surface-distance calculations use the parent origin,
        matching the source data.pos_w contract. Select postprocess_reference=lidar
        to explicitly use the physical ray origin instead. Raw scan distances
        always use the physical ray origin, regardless of this setting.
        """
        cam_pos = np.asarray(camera_position_w, dtype=float)
        cam_rot = np.asarray(camera_rotation_w, dtype=float)
        if cam_pos.shape != (3,) or not np.isfinite(cam_pos).all():
            raise ValueError('camera_position_w must be a finite 3-vector.')
        if (cam_rot.shape != (3, 3) or not np.isfinite(cam_rot).all()
                or not np.allclose(cam_rot.T @ cam_rot, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(cam_rot), 1, atol=1e-6)):
            raise ValueError('camera_rotation_w must be a proper ROS-optical-to-world rotation.')
        if not np.isfinite([yaw, pitch, target_radius]).all() or target_radius < 0:
            raise ValueError('Visual angles and nonnegative target radius must be finite.')
        invalid = TargetRangeEstimate(False, np.inf, np.inf, float(yaw), float(pitch), 0, None)
        desired = cam_rot @ np.array([np.sin(yaw) * np.cos(pitch), -np.sin(pitch),
                                      np.cos(yaw) * np.cos(pitch)])
        valid = scan.valid.copy()
        if target_geom_id is not None:
            valid &= scan.geom_ids == target_geom_id
        reference = scan.parent_position_w if self.config.postprocess_reference == 'parent' else scan.origin_w
        # Compute on finite hits only, avoiding inf*0 and invalid normalizations.
        vectors = scan.ray_hits_w[valid] - reference
        reference_distances = np.full(len(scan.valid), np.inf)
        reference_distances[valid] = np.linalg.norm(vectors, axis=1)
        cosine = np.full(len(scan.valid), -np.inf)
        cosine[valid] = (vectors @ desired) / np.maximum(reference_distances[valid], 1e-6)
        primary = max(4 * self.config.horizontal_resolution_deg, 4 * self.config.vertical_resolution_deg, 2.5)
        fallback = max(8 * self.config.horizontal_resolution_deg, 8 * self.config.vertical_resolution_deg, 5.0)
        candidates = valid & (cosine >= np.cos(np.deg2rad(primary)))
        if not candidates.any():
            candidates = valid & (cosine >= np.cos(np.deg2rad(fallback)))
        indices = np.flatnonzero(candidates)
        if not len(indices):
            return invalid
        distances = reference_distances[indices]
        middle = (len(distances) - 1) // 2
        median = np.partition(distances, middle)[middle]
        cluster = indices[np.abs(distances - median) <= max(0.35, 0.10 * median)]
        if not len(cluster):
            cluster = indices
        mean_hit = scan.ray_hits_w[cluster].mean(axis=0)
        surface_distance = float(reference_distances[cluster].mean())
        camera_to_hit = mean_hit - cam_pos
        center = mean_hit + camera_to_hit / max(np.linalg.norm(camera_to_hit), 1e-6) * target_radius
        camera_to_center = center - cam_pos
        distance = float(np.linalg.norm(camera_to_center))
        if not np.isfinite(center).all() or not np.isfinite(distance) or distance <= 1e-6:
            return invalid
        x, y, z = cam_rot.T @ camera_to_center
        z = max(z, 1e-6)
        refined_yaw = float(np.arctan2(x, z))
        refined_pitch = float(np.arctan2(-y, max(np.sqrt(x*x + z*z), 1e-6)))
        return TargetRangeEstimate(True, distance, surface_distance, refined_yaw,
                                    refined_pitch, len(cluster), center)
