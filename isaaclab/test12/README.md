# test12: OSTrack bbox target direction estimation

This folder estimates target direction from a first-frame box and OSTrack
single-object tracking:

```text
initial box
  -> OSTrack bbox tracking
  -> bbox center + camera intrinsics = yaw / pitch
```

It reuses `test10` camera geometry and writes a CSV compatible with
`test10.evaluate_direction`.

## Dependencies

Install the official OSTrack repository separately and download an OSTrack
checkpoint. This adapter expects the official layout:

```text
OSTrack/
  experiments/ostrack/*.yaml
  lib/test/tracker/ostrack.py
  lib/config/ostrack/config.py
```

The official tracker uses CUDA in its tracker class, so run with a CUDA PyTorch
environment.

## Single Sequence

```bash
python -m test12.estimate_video \
  --rgb_dir test10_direction_eval_dataset/sequences/seq_000000_env_000/rgb \
  --initial_box_file test10_direction_eval_dataset/sequences/seq_000000_env_000/initial_box.txt \
  --intrinsics_json test10_direction_eval_dataset/sequences/seq_000000_env_000/intrinsics.json \
  --fps 50 \
  --ostrack_repo /path/to/OSTrack \
  --ostrack_checkpoint /path/to/OSTrack/checkpoints/OSTrack_ep0300.pth.tar \
  --ostrack_config vitb_256_mae_ce_32x4_ep300 \
  --device cuda:0 \
  --out_csv test12_outputs/seq_000000_est.csv \
  --overlay_dir test12_outputs/seq_000000_overlay
```

Then evaluate:

```bash
python -m test10.evaluate_direction \
  --est_csv test12_outputs/seq_000000_est.csv \
  --gt_csv test10_direction_eval_dataset/sequences/seq_000000_env_000/ground_truth_direction.csv
```

## Dataset Evaluation

```bash
python -m test12.evaluate_direction_dataset \
  --dataset_dir test10_direction_eval_dataset \
  --fps 50 \
  --ostrack_repo /path/to/OSTrack \
  --ostrack_checkpoint /path/to/OSTrack/checkpoints/OSTrack_ep0300.pth.tar \
  --ostrack_config vitb_256_mae_ce_32x4_ep300 \
  --out_dir test12_outputs/direction_dataset_eval \
  --overlay
```

## Notes

- `--initial_box` and `initial_box.txt` use `x1 y1 x2 y2`.
- OSTrack internally uses `x y w h`; this adapter converts formats.
- Direction is computed from the tracked bbox center, not from a segmentation
  mask, so it avoids mask-centroid bias when segmentation is incomplete.
