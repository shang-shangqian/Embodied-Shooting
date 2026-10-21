"""Run A/B/C/D sequentially with one saved layout and summarize diagnostics.

Run from the Isaac conda environment after setup_conda_env.sh. No training is
started; each group is a separate evaluation process on the selected GPU.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DECISION = "runs/test7_2_weight/turret_aim_mixed_09-08-16-46-24/nn/last_turret_aim_ep_114000_rew__119.52544_.pth"
VELOCITY = "source/standalone/test11/speed_prediction/gru_runs_v2/20261009_152810_116165/best_model.pt"
LOCOMOTION = "logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt"


def group_command(group, output, *, gpu=1, timeout_s=30., seed=61, video=False):
    folder = output/group
    args = [sys.executable, "-u", str(ROOT/"source/standalone/test7_2_weight/eval_with_track_gt.py"),
            "--headless", "--device", f"cuda:{gpu}", "--num_envs", "1",
            "--checkpoint", str(ROOT/DECISION), "--locomotion_policy", str(ROOT/LOCOMOTION),
            "--yaw_error_mode", "wrapped_los",
            "--velocity_backend", "legacy_gru", "--full_chain_velocity_checkpoint", str(ROOT/VELOCITY),
            "--fine_tracking", "--fine_tracker_policy", "adaptive", "--case", "ss", "--static_salvo",
            "--salvo_targets", "100", "--episodes", "1", "--salvo_timeout_s", str(timeout_s),
            "--salvo_switch_policy", "timeout", "--salvo_stall_timeout_s", "3",
            "--salvo_retry_delay_s", "1", "--salvo_max_attempts", "3",
            "--salvo_log_interval_s", "10", "--salvo_ablation", group, "--no-salvo_wait_for_velocity",
            "--salvo_seed", str(seed),
            "--deterministic", "--no-pause_between_episodes", "--no-draw_tracks",
            "--salvo_report", str(folder/"salvo_report.json"),
            "--salvo_layout_out" if group == "A" else "--salvo_layout_in", str(output/"layout.json")]
    if video:
        args += ["--record_video", "--video_episodes", "1", "--video_env_id", "0", "--video_hit_hold_s", "0",
                 "--video_dir", str(folder/"video")]
    return args


def summarize(output):
    summaries = []
    reports = []
    for group in "ABCD":
        report = json.loads((output/group/"salvo_report.json").read_text())
        reports.append(report)
        if len(report["episodes"]) != 1:
            raise ValueError("Expected one episode per ablation group")
        episode = report["episodes"][0]
        with Path(report["diagnostics_csv"]).open() as stream:
            rows = list(csv.DictReader(stream))
        shots = [row for row in rows if row["fired"] == "True"]
        limits = [row for row in rows if row["post_pitch_at_limit"] == "True"]
        valid = [row for row in rows if row["decision_pred_valid"] == "True"]
        errors = sorted(abs(float(row["post_gt_muzzle_pitch_error_rad"]))*180/3.141592653589793
                        for row in valid if row["post_gt_muzzle_pitch_error_rad"])
        summaries.append(dict(group=group, shots=episode["shots"], destroyed=episode["destroyed"],
                              remaining=episode["remaining"], accuracy=episode["accuracy"], reason=episode["reason"],
                              last_fire_sim_s=float(shots[-1]["sim_s"]) if shots else None,
                              shots_per_engagement_s=episode["shots_per_engagement_s"],
                              pitch_limit_frames=len(limits), first_pitch_limit_sim_s=float(limits[0]["sim_s"]) if limits else None,
                              valid_frames=len(valid), median_valid_pitch_error_deg=errors[len(errors)//2] if errors else None,
                              hold_steps=episode["ablation"]["hold_steps"], actor_gru_resets=episode["ablation"]["actor_gru_resets"]))
    for field in ("layout_sha256", "decision_checkpoint", "velocity_checkpoint", "environment_seed", "rl_games_seed"):
        if reports[0].get(field) is None or len({report.get(field) for report in reports}) != 1:
            raise ValueError(f"Unmatched experiment condition: {field}")
    result = dict(layout_sha256=reports[0]["layout_sha256"], groups=summaries)
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# SS A/B/C/D 首轮对照", "", "同一保存布局、相同模型与显式 seed；每组单独进程顺序运行。单个布局不能代表统计结论。", "",
             "| 组别 | 枪数 | 击毁 | 命中率 | 末枪仿真秒 | pitch 限位帧数 | 保持帧数 | Actor GRU 重置次数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        accuracy = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "无射击"
        lines.append(f"| {s['group']} | {s['shots']} | {s['destroyed']} | {accuracy} | {s['last_fire_sim_s']} | {s['pitch_limit_frames']} | {s['hold_steps']} | {s['actor_gru_resets']} |")
    lines += ["", "A：原控制与循环状态；B：未就绪保持位置；C：切换后仅重置 Actor；D：两项同时启用。", "",
              "本四组对照显式关闭等待速度后推理，以保留原始实验定义；普通 static salvo 默认 D 已启用该等待流程。", "",
              "首次目标获取保留共同预热历史；后续有效目标 ID 改变时在推理前清空 Actor 隐状态一次。",
              "GT 仍只在一帧初始化跟踪，固定布局只用于模拟场景生成；诊断 GT 不输入 Actor。", "",
              f"布局 SHA256：`{result['layout_sha256']}`", ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def resume_completed(source, output, *, gpu, timeout_s, seed, video):
    """Copy completed groups and layout; keep interrupted source files untouched."""
    source=source.resolve()
    old_status=json.loads((source/"status.json").read_text())
    old_commands=json.loads((source/"commands.json").read_text())
    completed=old_status["completed"]
    if not completed or completed != list("ABCD")[:len(completed)]:
        raise ValueError("Resume requires a completed prefix starting with A")
    expected=group_command("A",output,gpu=gpu,timeout_s=timeout_s,seed=seed,video=video)
    old=old_commands["A"]
    for flag in ("--checkpoint","--full_chain_velocity_checkpoint","--locomotion_policy","--salvo_seed","--salvo_timeout_s"):
        actual=old[old.index(flag)+1];wanted=expected[expected.index(flag)+1]
        if flag in ("--salvo_seed","--salvo_timeout_s"):
            match=float(actual)==float(wanted)
        else:
            match=Path(actual).resolve()==Path(wanted).resolve()
        if not match:
            raise ValueError(f"Resume condition differs: {flag}")
    if ("--record_video" in old) != video:
        raise ValueError("Resume must retain the same recording mode")
    layout=json.loads((source/"layout.json").read_text())
    shutil.copy2(source/"layout.json",output/"layout.json")
    for group in completed:
        report=json.loads((source/group/"salvo_report.json").read_text())
        if report["layout_sha256"] != layout["sha256"] or any(e["reason"]=="interrupted" for e in report["episodes"]):
            raise ValueError(f"Cannot reuse incomplete or unmatched group {group}")
        shutil.copytree(source/group,output/group)
        report["resumed_source_report"]=str(source/group/"salvo_report.json")
        report["diagnostics_csv"]=str(output/group/Path(report["diagnostics_csv"]).name)
        (output/group/"salvo_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return list(completed),old_commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--timeout_s", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write exact commands without launching Isaac.")
    parser.add_argument("--resume_from", type=Path, help="Reuse completed groups/layout in a new output directory.")
    args = parser.parse_args()
    output = (args.output_dir or ROOT/"source/standalone/integrated_shooting_chain/video"/f"ablation_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = {group: group_command(group, output, gpu=args.gpu, timeout_s=args.timeout_s, seed=args.seed,
                                     video=args.record_video) for group in "ABCD"}
    completed=[]
    if args.resume_from:
        completed,old_commands=resume_completed(args.resume_from,output,gpu=args.gpu,timeout_s=args.timeout_s,
                                               seed=args.seed,video=args.record_video)
        for group in completed:
            commands[group]=old_commands[group]
    (output/"commands.json").write_text(json.dumps(commands, indent=2)+"\n")
    print(f"[ablation] output={output}", flush=True)
    if args.dry_run:
        return
    status = dict(state="running", completed=completed, active_group=None,
                  resumed_from=str(args.resume_from.resolve()) if args.resume_from else None)
    def save_status():
        (output/"status.json").write_text(json.dumps(status, indent=2)+"\n")
    save_status()
    for group, command in commands.items():
        if group in completed:
            print(f"[ablation] reusing completed {group}",flush=True)
            continue
        folder=output/group;folder.mkdir()
        status["active_group"]=group;save_status()
        print(f"[ablation] starting {group}", flush=True)
        started=time.monotonic()
        with (folder/"run.log").open("x") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status["pid"]=child.pid;save_status()
            try:
                returncode=child.wait()
            except KeyboardInterrupt:
                import signal
                child.send_signal(signal.SIGINT)
                child.wait()
                status["state"]="interrupted";save_status()
                raise
        if returncode != 0 or not (folder/"salvo_report.json").exists():
            status.update(state="failed", returncode=returncode);save_status()
            raise RuntimeError(f"Group {group} failed; see {folder/'run.log'}")
        status["completed"].append(group);save_status()
        print(f"[ablation] completed {group} wall_s={time.monotonic()-started:.1f}",flush=True)
    result=summarize(output)
    status.update(state="complete",active_group=None);save_status()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
