"""Fixed-world SS target pool; truth is used for setup and hit scoring only."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tools.play_mujoco_lite3 import mujoco
from ..isaac_training_bridge import _bootstrap_box


def build_salvo_model(asset_directory, target_count=100, radius=.35):
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError('target_count must be a positive integer')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Target radius must be finite and positive')
    spec = mujoco.MjSpec.from_file(str(Path(asset_directory)/'scene.xml'))
    for target_id in range(target_count):
        body = spec.worldbody.add_body(name=f'salvo_target_{target_id}', mocap=True,
                                      pos=[0, 0, -1000-2*target_id])
        body.add_geom(name=f'salvo_surface_{target_id}', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[radius, 0, 0], group=4 if target_id == 0 else 5,
                      contype=0, conaffinity=0, rgba=[.95, .12, .06, 1])
    return spec.compile()


def sample_layout(camera_position, camera_rotation, target_count, *, seed=61, radius=.35,
                  distance_range=(80., 150.), yaw_range=(-.26, .26), pitch_range=(-.12, .12)):
    """Use the Isaac optical-sector distribution, rejecting below-ground spheres."""
    bounds = [distance_range, yaw_range, pitch_range]
    if any(len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1] for pair in bounds):
        raise ValueError('Sampling ranges must be finite ordered pairs')
    if distance_range[0] <= radius or target_count < 1 or radius <= 0 or not np.isfinite(radius):
        raise ValueError('Invalid target count, radius or distance range')
    rng = np.random.default_rng(seed)
    result = np.empty((target_count, 3))
    missing = np.ones(target_count, dtype=bool)
    for _ in range(256):
        count = int(missing.sum())
        r, yaw, pitch = (rng.uniform(*pair, size=count) for pair in bounds)
        local = np.column_stack((r*np.sin(yaw)*np.cos(pitch), -r*np.sin(pitch),
                                 r*np.cos(yaw)*np.cos(pitch)))
        world = local @ np.asarray(camera_rotation).T + camera_position
        # Quantize once, matching the float32 layout consumed by Isaac.
        world = world.astype(np.float32).astype(float)
        accepted = world[:, 2] >= radius
        indices = np.flatnonzero(missing)[accepted]
        result[indices] = world[accepted]
        missing[indices] = False
        if not missing.any():
            return result
    raise ValueError('Cannot sample above-ground targets in the configured camera sector')


class MujocoStaticTargetPool:
    def __init__(self, model, data, target_count=100, radius=.35):
        self.model, self.data, self.radius = model, data, float(radius)
        self.mocap_ids = np.array([model.body(f'salvo_target_{i}').mocapid[0] for i in range(target_count)])
        self.geom_ids = np.array([model.geom(f'salvo_surface_{i}').id for i in range(target_count)])
        self.positions = np.full((target_count, 3), np.nan)
        self.alive = np.ones(target_count, dtype=bool)
        self.selected_id = None
        self.bootstrap_consumed = False

    @property
    def removed_ids(self):
        return set(np.flatnonzero(~self.alive).tolist())

    def place(self, positions):
        points = np.asarray(positions, dtype=float)
        if np.isfinite(self.positions).all():
            raise RuntimeError('The static target layout may only be placed once')
        if points.shape != self.positions.shape or not np.isfinite(points).all() or (points[:, 2] < self.radius).any():
            raise ValueError('Layout must contain finite, above-ground positions for every target')
        self.positions[:] = points
        self.data.mocap_pos[self.mocap_ids] = points
        mujoco.mj_forward(self.model, self.data)

    def bootstrap_boxes(self, camera):
        """The only perception-bootstrap operation that may read pool positions."""
        if self.bootstrap_consumed:
            raise RuntimeError('GT boxes may be consumed only once per target pool')
        self.bootstrap_consumed = True
        position, rotation = camera.pose(self.data)
        local = (self.positions-position) @ rotation
        k = camera.intrinsic_matrix
        boxes = {}
        for target_id, (x, y, z) in enumerate(local):
            box = None if z <= 0 else _bootstrap_box(
                np.arctan2(x, z), np.arctan2(-y, np.hypot(x, z)), np.linalg.norm([x, y, z]),
                fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2],
                width=camera.width, height=camera.height, target_radius_m=self.radius, min_box_px=8.)
            if box is None:
                raise ValueError(f'Target {target_id} is outside the bootstrap image')
            boxes[target_id] = box
        return boxes

    def select(self, target_id):
        if target_id is not None and (not 0 <= target_id < len(self.alive) or not self.alive[target_id]):
            raise ValueError('Cannot select a missing or destroyed target')
        # All surviving targets render. Only the selected sphere belongs to
        # the LiDAR group, matching Isaac's target-only raycast mesh list.
        self.model.geom_group[self.geom_ids] = 5
        if target_id is not None:
            self.model.geom_group[self.geom_ids[target_id]] = 4
        self.selected_id = target_id

    def remove(self, target_id):
        if not 0 <= target_id < len(self.alive) or not self.alive[target_id]:
            raise ValueError('Target is already destroyed or does not exist')
        self.alive[target_id] = False
        self.data.mocap_pos[self.mocap_ids[target_id]] = [0, 0, -1000-2*target_id]
        self.model.geom_group[self.geom_ids[target_id]] = 5
        self.model.geom_rgba[self.geom_ids[target_id], 3] = 0
        if self.selected_id == target_id:
            self.selected_id = None
        mujoco.mj_forward(self.model, self.data)
