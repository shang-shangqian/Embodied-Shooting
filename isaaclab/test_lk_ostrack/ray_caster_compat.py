# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from Isaac Lab v2.3.2; see ISAACLAB_LICENSE.txt.
"""Fix mesh-ID output shape locally without modifying Isaac Lab."""
from collections.abc import Sequence
from isaaclab.sensors import MultiMeshRayCaster as _BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.utils.math import quat_mul
from isaaclab.utils.warp import raycast_dynamic_meshes


class MultiMeshRayCaster(_BaseMultiMeshRayCaster):
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data.

        Args:
            env_ids: The environment ids to update.
        """

        self._update_ray_infos(env_ids)

        # Update the mesh positions and rotations
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # update position of the target meshes
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # Mesh is not global, i.e. we have different meshes for each env
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        self._data.ray_hits_w[env_ids], _, _, _, mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_mesh_id=self.cfg.update_mesh_ids,
        )

        if self.cfg.update_mesh_ids:
            # ``raycast_dynamic_meshes`` returns mesh IDs with shape (N, B),
            # while the public sensor buffer stores one ID channel per ray
            # with shape (N, B, 1).
            self._data.ray_mesh_ids[env_ids] = mesh_ids.unsqueeze(-1)
