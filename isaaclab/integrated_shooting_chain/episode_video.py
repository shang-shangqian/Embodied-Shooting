"""Record selected Isaac evaluation episodes with the perception camera inset."""

from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np


class EpisodeVideoRecorder:
    def __init__(self, directory, *, env_id=0, episodes=10, step_dt=0.02, hit_hold_s=0.0, writer_factory=None):
        if episodes < 1 or step_dt <= 0 or env_id < 0 or hit_hold_s < 0:
            raise ValueError("Video episodes and step_dt must be positive; env_id and hit_hold_s must be nonnegative.")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.env_id = int(env_id)
        self.episodes = int(episodes)
        self.step_dt = float(step_dt)
        self.fps = 1.0 / self.step_dt
        self.hit_hold_s = float(hit_hold_s)
        self.completed = 0
        self.records = []
        self._writer_factory = writer_factory or self._open_writer
        self._writer = None
        self._frames = 0
        self._last_status = {}
        self._last_frame = None
        self._path = None
        self._closed = False
        self._save_manifest()

    @property
    def active(self):
        return not self._closed and self.completed < self.episodes

    def _open_writer(self, path, size):
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open MP4 video writer: {path}")
        return writer

    def _compose_frame(self, scene, camera, status):
        import cv2

        if "targets" in status:
            return self._compose_salvo_frame(scene, camera, status)
        frame = np.ascontiguousarray(scene[..., :3], dtype=np.uint8).copy()
        height, width = frame.shape[:2]
        inset_width = min(416, width // 3)
        inset_height = max(1, round(inset_width * camera.shape[0] / camera.shape[1]))
        inset = cv2.resize(camera[..., :3], (inset_width, inset_height))
        inset_left = width - inset_width - 12
        inset_top = 40
        frame[inset_top:inset_top + inset_height, inset_left:inset_left + inset_width] = inset
        cv2.rectangle(frame, (inset_left - 2, inset_top - 2),
                      (inset_left + inset_width + 1, inset_top + inset_height + 1), (40, 180, 255), 2)
        cv2.putText(frame, "Robot RGB (vision input)", (inset_left, inset_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if status["hit"]:
            cv2.putText(frame, "HIT | yellow: evaluated ballistic path", (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 255, 100), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
        caption = (f"env {self.env_id} | episode {len(self.records) + 1} | "
                   f"t={status['step'] * self.step_dt:.2f}s | fires={status['fires']} | hit={int(status['hit'])}")
        phase = "WARMUP" if status["warmup"] else "ACTIVE"
        cv2.putText(frame, caption, (12, height - 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{phase} | perception ready={int(status['ready'])}", (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.ascontiguousarray(frame[..., ::-1])

    def _compose_salvo_frame(self, scene, camera, status):
        import cv2

        height, width = scene.shape[:2]
        frame = cv2.resize(camera[..., :3], (width, height)).copy()
        scale_x, scale_y = width / camera.shape[1], height / camera.shape[0]
        for track in status.get("tracks", []):
            x1, y1, x2, y2 = track["box"]
            selected = track["id"] == status["target_id"]
            color = (255, 220, 0) if selected else ((100, 100, 100) if track["status"] == "lost" else (0, 210, 255))
            if selected and status["hit"]:
                color = (70, 255, 100)
            start = (round(x1 * scale_x), round(y1 * scale_y))
            end = (round(x2 * scale_x), round(y2 * scale_y))
            cv2.rectangle(frame, start, end, color, 2 if selected else 1)
            cv2.putText(frame, str(track["id"]), (start[0], max(15, start[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        inset_width = min(352, width // 3)
        inset_height = round(inset_width * height / width)
        left, top = width - inset_width - 12, 40
        frame[top:top + inset_height, left:left + inset_width] = cv2.resize(scene[..., :3], (inset_width, inset_height))
        cv2.rectangle(frame, (left - 1, top - 1), (left + inset_width, top + inset_height), (255, 255, 255), 1)
        cv2.putText(frame, "Full scene (observer)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        event = "CLEARED" if status["cleared"] else ("HIT" if status["hit"] else ("FIRE" if status["shot_fired"] else "TRACKING"))
        cv2.rectangle(frame, (0, 0), (max(400, width // 2), 35), (0, 0, 0), -1)
        cv2.putText(frame, f"SS {status['targets']} targets | {event} | target={status['target_id']}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 255, 100) if status["hit"] else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, height - 65), (width, height), (0, 0, 0), -1)
        accuracy = "NA" if status["accuracy"] is None else f"{100 * status['accuracy']:.1f}%"
        caption = (f"t={status['step'] * self.step_dt:.2f}s | fires={status['fires']} | "
                   f"hits={status['destroyed']} | remaining={status['remaining']} | accuracy={accuracy}")
        cv2.putText(frame, caption, (12, height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        phase = "WARMUP" if status["warmup"] else "ACTIVE"
        cv2.putText(frame, f"{phase} | perception ready={int(status['ready'])} | cyan=tracked yellow=selected gray=lost",
                    (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.ascontiguousarray(frame[..., ::-1])

    def capture(self, scene, camera, status):
        if not self.active:
            return
        if scene is None or camera is None or scene.ndim != 3 or camera.ndim != 3:
            raise RuntimeError("Recording requires both a scene RGB image and a robot-camera RGB image.")
        frame = self._compose_frame(scene, camera, status)
        if self._writer is None:
            self._path = self.directory / f"env{self.env_id}_episode_{len(self.records) + 1:03d}.mp4"
            self._writer = self._writer_factory(self._path, (frame.shape[1], frame.shape[0]))
        self._writer.write(frame)
        self._frames += 1
        self._last_status = dict(status)
        self._last_frame = frame

    def finish_episode(self, *, complete):
        if self._writer is None:
            return
        hold_frames = round(self.hit_hold_s * self.fps) if complete and self._last_status["hit"] and self._last_status.get("cleared", True) else 0
        for _frame in range(hold_frames):
            self._writer.write(self._last_frame)
        self._frames += hold_frames
        self._writer.release()
        self._writer = None
        self.records.append({
            "file": self._path.name,
            "complete": bool(complete),
            "frames": self._frames,
            "duration_s": self._frames / self.fps,
            "simulation_frames": self._frames - hold_frames,
            "hit_hold_frames": hold_frames,
            "simulation_duration_s": self._last_status["step"] * self.step_dt,
            **self._last_status,
        })
        self.completed += int(complete)
        self._frames = 0
        self._last_frame = None
        self._save_manifest()
        print(f"[eval][video] saved {self._path} (complete={complete}, {self.completed}/{self.episodes})")

    def _save_manifest(self):
        manifest = {
            "env_id": self.env_id, "requested_episodes": self.episodes,
            "completed_episodes": self.completed, "fps": self.fps,
            "layout": ("robot_rgb_tracks_with_scene_inset" if any("targets" in record for record in self.records)
                       else "scene_with_robot_rgb_inset"), "videos": self.records,
            "hit_hold_s": self.hit_hold_s,
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def close(self):
        if self._closed:
            return
        self.finish_episode(complete=False)
        self._closed = True
        self._save_manifest()
        print(f"[eval][video] recorded {self.completed}/{self.episodes} complete episodes in {self.directory}")


def install_episode_video_recorder(base_env, recorder, frame_provider=None):
    if recorder.env_id >= base_env.num_envs:
        raise ValueError("video_env_id must be smaller than num_envs.")
    original_step = base_env.step
    original_reset = base_env._reset_idx
    salvo = getattr(base_env, "_static_salvo", None)
    inside_step = False
    selected_reset = False
    captured_this_step = False

    if frame_provider is None:
        from .video_scene_view import SceneVideoView

        frame_provider = SceneVideoView(base_env, recorder.env_id)
    provide_frames = frame_provider

    def capture(terminal):
        nonlocal captured_this_step
        if not recorder.active:
            return
        selected = recorder.env_id
        scene, camera = provide_frames(terminal)
        status = {
            "step": int(base_env.episode_length_buf[selected].item()),
            "fires": int(base_env._episode_fire_count[selected].item()),
            "hit": bool(base_env._hit_flag[selected].item()),
            "warmup": bool(base_env._aim_warmup_left_steps[selected].item() > 0),
            "ready": bool(base_env._rl_pred_ready[selected].item()),
        }
        if salvo is not None:
            destroyed = int(salvo.hits[selected].item())
            fires = int(salvo.shots[selected].item())
            target_id = int(salvo.active[selected].item())
            targets = salvo.positions.shape[1]
            bridge = getattr(base_env, "_full_chain_perception_bridge", None)
            results = getattr(bridge, "_latest_results", None)
            result = None if results is None else results[selected]
            tracks = [] if result is None else [
                {"id": int(track.target_id), "box": [float(v) for v in track.bbox_xyxy], "status": str(track.status)}
                for track in result.targets
                if bool(salvo.alive[selected, track.target_id]) or track.target_id == target_id
            ]
            status.update(
                targets=targets, destroyed=destroyed, remaining=targets - destroyed,
                fires=fires, hit=bool(salvo.last_hit[selected].item()),
                shot_fired=bool(base_env._fire_command[selected].item()), target_id=target_id,
                accuracy=destroyed / fires if fires else None,
                cleared=bool(salvo.ready[selected]) and destroyed == targets,
                gt_bootstrap_frames=int(salvo.bootstrap_frames[selected].item()),
                gt_bootstrap_targets=int(salvo.bootstrap_targets[selected].item()), tracks=tracks,
            )
        recorder.capture(scene, camera, status)
        captured_this_step = True

    if salvo is not None:
        original_rewards = base_env._get_rewards

        def rewards_with_video(self):
            reward = original_rewards()
            # Capture post-physics hits before target switching or autoreset
            # clears the hit flag, ballistic cache, and selected target.
            if inside_step and not captured_this_step:
                capture(terminal=True)
            return reward

        base_env._get_rewards = types.MethodType(rewards_with_video, base_env)

    def reset_with_video(self, env_ids):
        nonlocal selected_reset
        reset_ids = range(self.num_envs) if env_ids is None else env_ids
        if hasattr(reset_ids, "tolist"):
            reset_ids = reset_ids.tolist()
        if recorder.env_id in reset_ids:
            selected_reset = True
            if inside_step and recorder.active:
                if not captured_this_step:
                    capture(terminal=True)
                recorder.finish_episode(complete=True)
            else:
                recorder.finish_episode(complete=False)
        return original_reset(env_ids)

    def step_with_video(self, actions):
        nonlocal inside_step, selected_reset, captured_this_step
        selected_reset = False
        captured_this_step = False
        inside_step = True
        try:
            result = original_step(actions)
        finally:
            inside_step = False
        if not selected_reset and not captured_this_step:
            capture(terminal=False)
        return result

    base_env._reset_idx = types.MethodType(reset_with_video, base_env)
    base_env.step = types.MethodType(step_with_video, base_env)
    return recorder
