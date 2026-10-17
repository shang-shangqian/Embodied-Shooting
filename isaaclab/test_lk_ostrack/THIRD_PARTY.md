# Third-party code and assets

- `ostrack/OSTrack-main/` contains the transitive inference source subset and the
  `vitb_256_mae_ce_32x4_ep300` configuration from the locally supplied snapshot of
  [botaoye/OSTrack](https://github.com/botaoye/OSTrack). The original MIT license is
  preserved at `ostrack/OSTrack-main/LICENSE`. The archive did not include Git
  metadata, so no upstream commit identity is claimed. Benchmark datasets,
  training entrypoints and model weights are excluded. The snapshot includes
  the Python 3 / modern PyTorch `torch._six` import fallback in its loader.
- `ray_caster_compat.py` adapts `_update_buffers_impl` from Isaac Lab v2.3.2
  (BSD-3-Clause; full text in `ISAACLAB_LICENSE.txt`). The only algorithmic change
  is adding the final singleton dimension to returned mesh IDs. The subclass is
  used only by this package and does not patch the installed SDK.
- Lite3 and turret USD/URDF assets are copied from this experiment's local
  assets. Geometry is embedded and all layer references are relative. The
  remote SG2 camera reference is replaced with a local USD Camera using the
  existing mount transform and experiment intrinsics. No third-party camera
  asset download is needed.
- No additional license is imposed on the repository's own experiment code or
  robot assets by this packaging change. Preserved dependency licenses apply to
  their respective files.
