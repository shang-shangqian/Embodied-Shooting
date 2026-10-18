"""Run real LK/OSTrack + raycaster observations through the frozen ordinary GRU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "test11.speed_prediction.legacy_gru"

from .online_gru_velocity import OnlineLegacyGRU, load_legacy_checkpoint


TEST11_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TEST11_DIR.parents[1]
VISION_DIR = TEST11_DIR.parent / "test_lk_ostrack"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/velocity/best_model.pt"


def velocity_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    if not len(errors):
        return {"n": 0}
    norms = np.linalg.norm(errors, axis=1)
    return dict(n=len(errors), mean=float(norms.mean()), p50=float(np.quantile(norms, .5)),
                p95=float(np.quantile(norms, .95)), p99=float(np.quantile(norms, .99)),
                max=float(norms.max()), mae_xyz=np.abs(errors).mean(axis=0).tolist(),
                rmse_xyz=np.sqrt(np.square(errors).mean(axis=0)).tolist(),
                bias_xyz=errors.mean(axis=0).tolist(),
                over_0p5=float((norms > .5).mean()), over_1=float((norms > 1).mean()),
                over_2=float((norms > 2).mean()))


class VisionGRUMetrics:
    def __init__(self, model, checkpoint, args, output):
        contexts = ("rolling", "full") if args.context == "both" else (args.context,)
        self.model = model
        self.args = args
        self.on_device = False
        self.adapters = {}
        self.contexts = contexts
        self.smooth_window = int(checkpoint["preprocessing"]["pos_smooth_window"])
        self.episodes = {}
        self.groups = defaultdict(list)
        self.latencies = defaultdict(list)
        self.counters = defaultdict(int)
        self.output = output
        self.summary = None
        self.handle = (output / "frame_metrics.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=(
            "env_id", "episode_id", "timestamp_s", "context", "source_id", "range_valid",
            "direct_observation", "prediction_ready", "scored", "ground", "history_frames",
            "initialization_wait",
            "history_span_s", "prediction_vx", "prediction_vy", "prediction_vz",
            "target_vx", "target_vy", "target_vz", "error_norm", "inference_ms",
        ))
        self.writer.writeheader()

    def observe(self, frame, *, env_id, episode_id, metadata):
        if metadata.get("validation_gt_direction") or metadata.get("validation_gt_range"):
            raise ValueError("Evaluation must not feed GT direction or GT range into vision")
        if frame.audit.get("gt_direction_used_for_model_input") or frame.audit.get("gt_range_used_for_model_input"):
            raise ValueError("Oracle model input is forbidden")
        if not self.on_device:
            self.model.to(self.args.device)
            self.on_device = True
        episode_key = (env_id, episode_id)
        episode = self.episodes.setdefault(episode_key, dict(
            start=frame.timestamp_s, first_direct=None, first_prediction={}, first_scored={}))
        range_valid = bool(frame.range_valid and np.isfinite(frame.position_base_m).all())
        direct = frame.source_id <= 2 and range_valid
        if direct and episode["first_direct"] is None:
            episode["first_direct"] = frame.timestamp_s
        self.counters["observed_frames"] += 1
        self.counters["direct_frames"] += int(direct)
        self.counters["range_valid_frames"] += int(range_valid)
        initialization_wait = self.args.initialization_policy == "first_valid" and episode["first_direct"] is None
        self.counters["initialization_wait_frames"] += int(initialization_wait)
        for context in self.contexts:
            adapter_key = (context, env_id)
            if adapter_key not in self.adapters:
                self.adapters[adapter_key] = OnlineLegacyGRU(
                    self.model, device=self.args.device, context=context, window_s=self.args.window_s,
                    smooth_window=self.smooth_window, max_history_frames=self.args.max_history_frames,
                    measure_latency=self.args.measure_latency,
                    initialization_policy=self.args.initialization_policy,
                )
            result = self.adapters[adapter_key].update(
                timestamp_s=frame.timestamp_s, position_base_m=frame.position_base_m,
                robot_linear_velocity_base_mps=frame.robot_linear_velocity_base_mps,
                robot_angular_velocity_base_rps=frame.robot_angular_velocity_base_rps,
                episode_key=episode_key,
                direct_observation=direct,
            )
            if result is not None:
                self.counters[f"ready_frames_{context}"] += 1
                episode["first_prediction"].setdefault(context, frame.timestamp_s)
            target = np.asarray(frame.target_relative_velocity_base_mps, dtype=np.float64)
            target_position = frame.audit.get("target_position_world_m")
            ground = (bool(target_position[2] >= 0) if target_position is not None
                      and np.isfinite(target_position[2]) else None)
            scored = (result is not None and np.isfinite(target).all()
                      and frame.timestamp_s - episode["start"] >= self.args.burn_in_s - 1e-9
                      and episode["first_direct"] is not None)
            prediction = result["velocity"] if result is not None else [None] * 3
            error = prediction - target if scored else None
            if scored:
                episode["first_scored"].setdefault(context, frame.timestamp_s)
                self.counters[f"scored_{context}"] += 1
                age = frame.timestamp_s - episode["first_direct"]
                source_group = ("direct" if direct else "prediction" if frame.source_id == 3
                                else "lost" if frame.source_id == 4 else "range_unavailable")
                names = ["all", source_group,
                         "startup_0p5s" if age < .5 else "after_startup"]
                if not range_valid and "range_unavailable" not in names:
                    names.append("range_unavailable")
                for filter_name in ("unfiltered", "center_above_ground"):
                    if filter_name == "center_above_ground" and ground is not True:
                        continue
                    for group in names:
                        self.groups[(context, filter_name, group)].append(error.copy())
            if result is not None and result["inference_ms"] is not None:
                self.latencies[context].append(result["inference_ms"])
            self.writer.writerow(dict(
                env_id=env_id, episode_id=episode_id, timestamp_s=frame.timestamp_s, context=context,
                source_id=frame.source_id, range_valid=range_valid, direct_observation=direct,
                initialization_wait=initialization_wait,
                prediction_ready=result is not None, scored=scored, ground=ground,
                history_frames=result["history_frames"] if result else 0,
                history_span_s=result["history_span_s"] if result else 0,
                prediction_vx=prediction[0], prediction_vy=prediction[1], prediction_vz=prediction[2],
                target_vx=target[0], target_vy=target[1], target_vz=target[2],
                error_norm=float(np.linalg.norm(error)) if scored else None,
                inference_ms=result["inference_ms"] if result else None,
            ))
        if self.args.progress_every > 0 and self.counters["observed_frames"] % self.args.progress_every == 0:
            self.handle.flush()
            for context in self.contexts:
                metrics = velocity_metrics(self.groups[(context, "unfiltered", "all")])
                if metrics["n"]:
                    print(f"[GRU {context}] n={metrics['n']} Mean={metrics['mean']:.4f} "
                          f"P95={metrics['p95']:.4f} P99={metrics['p99']:.4f} m/s", flush=True)

    def finish(self, completed, checkpoint_path):
        if self.summary is not None:
            return self.summary
        self.handle.close()
        metrics = [dict(context=key[0], filter=key[1], group=key[2], **velocity_metrics(errors))
                   for key, errors in self.groups.items() if len(errors)]
        latency = {context: dict(n=len(values), mean_ms=float(np.mean(values)),
                                 p95_ms=float(np.quantile(values, .95)))
                   for context, values in self.latencies.items() if values}
        summary = dict(
            completed=completed, checkpoint=str(checkpoint_path),
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            scenario=self.args.scenario, profile=self.args.profile, condition=self.args.condition,
            seed=self.args.seed, device=self.args.device, episodes=len(self.episodes),
            counters=dict(self.counters), metrics=metrics, adapter_latency=latency,
            episode_timing=[dict(
                env_id=key[0], episode_id=key[1], start_s=episode["start"],
                first_direct_s=episode["first_direct"], first_prediction_s=episode["first_prediction"],
                first_scored_s=episode["first_scored"],
                first_output_delay_s={context: timestamp - episode["start"]
                                      for context, timestamp in episode["first_prediction"].items()},
                first_output_after_direct_s={context: timestamp - episode["first_direct"]
                                            for context, timestamp in episode["first_prediction"].items()
                                            if episode["first_direct"] is not None},
            ) for key, episode in self.episodes.items()],
            protocol=dict(
                scene="test11 TurretAimCollectEnv, torso-mounted camera", vision=str(VISION_DIR),
                frontend="MultiTargetVisionTracker LK + adaptive OSTrack; direction-driven raycaster range",
                initialization="simulation bbox detector proxy, including tracker reinitialization",
                initialization_policy=self.args.initialization_policy,
                initialization_input="first_valid skips inputs before the first direct valid position; first position seeds history only",
                label="target_relative_translation_velocity_yaw_base", unit="m/s",
                reset="new simulation episode only; tracker loss does not reset GRU",
                missing="after initialization, use visual prediction position; invalid measured positions zero-filled, no GT repair",
                window_s=self.args.window_s, max_history_frames=self.args.max_history_frames,
                pos_smooth_window=self.smooth_window, burn_in_s=self.args.burn_in_s,
                scoring="all ready frames after episode burn-in and first direct valid observation",
                ground="target center world z >= 0; score-only secondary mask",
                no_direct_episode_policy="excluded from accuracy, retained in availability counters",
                timestamp="collector sensor time; mixed timing semantics are not repaired here",
                latency="optional synchronized adapter+GRU timing, excludes vision and simulation",
                commands=[self.args.locomotion_vx_mps, self.args.locomotion_vy_mps,
                          self.args.locomotion_yaw_rate_radps],
            ),
        )
        (self.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# LK/OSTrack 与普通 GRU 实时联调", "",
                 f"完成：{completed}；场景：{self.args.scenario}/{self.args.profile}；设备：{self.args.device}。",
                 "Mean/P95/P99 为逐帧三维速度误差范数，单位 m/s。GT 只参与评分；初始化使用仿真检测框代理。",
                 f"初始化策略：{self.args.initialization_policy}；等待帧：{self.counters['initialization_wait_frames']}。首次输出时间见 summary.json 的 episode_timing。",
                 "", "| 上下文 | 筛选 | 分组 | 帧数 | Mean | P95 | P99 |",
                 "|---|---|---|---:|---:|---:|---:|"]
        for row in metrics:
            lines.append(f"| {row['context']} | {row['filter']} | {row['group']} | {row['n']} | "
                         f"{row['mean']:.4f} | {row['p95']:.4f} | {row['p99']:.4f} |")
        if not metrics:
            lines.append("没有可评分帧；检查运行长度、初始化、目标可见性与测距有效性。")
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[GRU] summary: {self.output / 'summary.json'}", flush=True)
        for row in metrics:
            if row["group"] == "all":
                print(f"[GRU final] {row['context']} {row['filter']}: n={row['n']} "
                      f"Mean={row['mean']:.4f} P95={row['p95']:.4f} P99={row['p99']:.4f} m/s", flush=True)
        self.summary = summary
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scenario", choices=("ss", "sd"), default="sd")
    parser.add_argument("--profile", choices=("clean", "mixed"), default="clean")
    parser.add_argument("--condition", choices=("D", "none"), default="D")
    parser.add_argument("--context", choices=("rolling", "full", "both"), default="rolling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--window_s", type=float, default=.48)
    parser.add_argument("--max_history_frames", type=int, default=32)
    parser.add_argument("--burn_in_s", type=float, default=.1)
    parser.add_argument("--initialization_policy", choices=("first_valid", "zero_filled"), default="first_valid",
                        help="Skip inputs until the first direct valid position; zero_filled reproduces the original initialization")
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--measure_latency", action="store_true")
    parser.add_argument("--locomotion_vx_mps", type=float, default=0.)
    parser.add_argument("--locomotion_vy_mps", type=float, default=0.)
    parser.add_argument("--locomotion_yaw_rate_radps", type=float, default=0.)
    parser.add_argument("--locomotion_policy", type=Path, required=True)
    parser.add_argument("--ostrack_repo", type=Path, default=VISION_DIR / "ostrack/OSTrack-main")
    parser.add_argument("--ostrack_checkpoint", type=Path, default=REPO_ROOT / "checkpoints/OSTrack_ep0300.pth.tar")
    args = parser.parse_args()
    if args.num_envs < 1 or args.num_steps < 1 or args.window_s <= 0 or args.max_history_frames < 2:
        parser.error("Positive environment/step/window settings and at least two history frames are required")
    if not 0 <= args.burn_in_s < args.window_s:
        parser.error("Require 0 <= burn_in_s < window_s")
    for name in ("locomotion_vx_mps", "locomotion_vy_mps", "locomotion_yaw_rate_radps"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"{name} must be finite")
    if args.condition == "none" and any(getattr(args, name) for name in (
        "locomotion_vx_mps", "locomotion_vy_mps", "locomotion_yaw_rate_radps"
    )):
        parser.error("Moving-robot evaluation requires condition D to avoid static-base filtering")
    return args


def collector_arguments(args, output):
    arguments = ["--scenario", args.scenario, "--out_dir", str(output / "observations"),
                 "--device", args.device, "--num_envs", str(args.num_envs),
                 "--num_steps", str(args.num_steps), "--seed", str(args.seed),
                 "--split", "test", "--validation_condition", args.condition,
                 "--pair_id", f"legacy_gru_vision_{args.seed}",
                 "--degradation_profile", args.profile, "--save_episode_stream",
                 "--keep_measurement_discontinuities", "--target_samples", "0",
                 "--no-use_gt_visibility_gate_after_initialization",
                 "--window_s", str(args.window_s), "--supervision_burn_in_s", str(args.burn_in_s),
                 "--progress_every", str(args.progress_every),
                 "--locomotion_policy", str(args.locomotion_policy.expanduser().resolve()),
                 "--ostrack_repo", str(args.ostrack_repo.expanduser().resolve()),
                 "--ostrack_checkpoint", str(args.ostrack_checkpoint.expanduser().resolve()),
                 "--locomotion_vx_mps", str(args.locomotion_vx_mps),
                 "--locomotion_vy_mps", str(args.locomotion_vy_mps),
                 "--locomotion_yaw_rate_radps", str(args.locomotion_yaw_rate_radps)]
    if args.headless:
        arguments.append("--headless")
    return arguments


def main():
    args = parse_args()
    for path in (args.checkpoint, args.locomotion_policy, args.ostrack_repo, args.ostrack_checkpoint):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)
    model, checkpoint = load_legacy_checkpoint(args.checkpoint.expanduser().resolve())
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.out_dir or Path("outputs/velocity/vision_eval") / timestamp).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observer = VisionGRUMetrics(model, checkpoint, args, output)
    arguments = collector_arguments(args, output)
    (output / "run_config.json").write_text(json.dumps(
        dict(arguments={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
             collector_arguments=arguments, checkpoint_config=checkpoint["config"]),
        ensure_ascii=False, indent=2), encoding="utf-8")
    previous_argv = sys.argv
    completed = False
    try:
        sys.argv = [str(TEST11_DIR / "speed_prediction/collect_vision_data.py"), *arguments]
        from .. import collect_vision_data

        collect_vision_data.main(
            frame_observer=observer.observe,
            collection_finished=lambda success: observer.finish(success, args.checkpoint.expanduser().resolve()),
        )
        completed = True
    finally:
        sys.argv = previous_argv
        observer.finish(completed, args.checkpoint.expanduser().resolve())


if __name__ == "__main__":
    main()
