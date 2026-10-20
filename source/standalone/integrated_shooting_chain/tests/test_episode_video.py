import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrated_shooting_chain.episode_video import EpisodeVideoRecorder, install_episode_video_recorder


@pytest.mark.parametrize("override", [None, "custom_recordings/new_run"])
def test_default_video_directory_is_repo_relative_and_override_is_preserved(tmp_path, monkeypatch, override):
    script = Path(__file__).resolve().parents[2] / "test7_2_weight/eval_with_track_gt.py"
    tree = ast.parse(script.read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "video_directory" for target in node.targets))
    monkeypatch.chdir(tmp_path)
    namespace = {
        "__file__": str(script), "Path": Path, "args": SimpleNamespace(video_dir=override),
        "datetime": SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda pattern: "test_timestamp")),
    }
    result = eval(compile(ast.Expression(assignment.value), str(script), "eval"), namespace)
    expected = override or str(script.parents[1] / "integrated_shooting_chain/video/test_timestamp")
    assert result == expected


@pytest.mark.parametrize("record_video,static_salvo,expected", [(False, False, True), (True, False, False), (False, True, False)])
def test_recording_disables_legacy_world_space_debug_tracks(record_video, static_salvo, expected):
    script = Path(__file__).resolve().parents[2] / "test7_2_weight/eval_with_track_gt.py"
    tree = ast.parse(script.read_text())
    draw_hook = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and "args.draw_tracks" in ast.unparse(node.test))
    namespace = {"env_cfg": SimpleNamespace(debug_vis=True),
                 "args": SimpleNamespace(draw_tracks=True, record_video=record_video, static_salvo=static_salvo)}
    result = eval(compile(ast.Expression(draw_hook.test), str(script), "eval"), namespace)
    assert result is expected


class MemoryWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self._episode_fire_count = torch.zeros(2, dtype=torch.long)
        self._hit_flag = torch.zeros(2, dtype=torch.bool)
        self._aim_warmup_left_steps = torch.zeros(2, dtype=torch.long)
        self._rl_pred_ready = torch.ones(2, dtype=torch.bool)

    def _reset_idx(self, env_ids):
        selected = list(range(2)) if env_ids is None else env_ids
        self.episode_length_buf[selected] = 0
        self._hit_flag[selected] = False
        self._episode_fire_count[selected] = 0

    def step(self, reset_ids):
        self.episode_length_buf += 1
        if reset_ids:
            self._hit_flag[reset_ids] = True
            self._episode_fire_count[reset_ids] = 1
            self._reset_idx(torch.tensor(reset_ids))
        return "unchanged_result"


@pytest.fixture
def recording(tmp_path):
    writers = []

    def writer_factory(path, size):
        writer = MemoryWriter()
        writers.append(writer)
        return writer

    recorder = EpisodeVideoRecorder(tmp_path / "videos", writer_factory=writer_factory)
    recorder._compose_frame = lambda scene, camera, status: scene.copy()
    env = FakeEnv()

    def provide_frames(terminal):
        frame = np.full((4, 4, 3), env.episode_length_buf[0].item(), dtype=np.uint8)
        return frame, frame

    install_episode_video_recorder(env, recorder, provide_frames)
    return SimpleNamespace(env=env, recorder=recorder, writers=writers)


def test_terminal_frame_is_captured_before_autoreset(recording):
    assert recording.env.step([]) == "unchanged_result"
    recording.env.step([0])
    assert [frame[0, 0, 0] for frame in recording.writers[0].frames] == [1, 2]
    assert recording.writers[0].released
    assert recording.recorder.records[0]["hit"]
    assert recording.recorder.records[0]["fires"] == 1
    assert recording.recorder.records[0]["complete"]
    recording.env.step([])
    assert recording.writers[1].frames[0][0, 0, 0] == 1


def test_initial_reset_does_not_count_as_episode(recording):
    recording.env._reset_idx(None)
    assert recording.recorder.completed == 0
    assert recording.recorder.records == []


def test_other_env_reset_does_not_split_video(recording):
    recording.env.step([1])
    recording.env.step([1])
    assert recording.recorder.completed == 0
    assert len(recording.writers) == 1
    assert len(recording.writers[0].frames) == 2


def test_ten_episode_limit_disables_capture(recording):
    for _episode in range(12):
        recording.env.step([0])
    assert recording.recorder.completed == 10
    assert len(recording.writers) == 10
    assert all(writer.released for writer in recording.writers)
    assert not recording.recorder.active


def test_hit_frame_hold_only_adds_playback_frames_without_stepping_physics(recording):
    recording.recorder.hit_hold_s = 1.5
    recording.env.step([])
    recording.env.step([0])
    record = recording.recorder.records[0]
    assert record["frames"] == 77
    assert record["simulation_frames"] == 2
    assert record["hit_hold_frames"] == 75
    assert record["simulation_duration_s"] == pytest.approx(0.04)
    assert record["duration_s"] == pytest.approx(1.54)
    assert len(recording.writers[0].frames) == 77
    assert all(frame[0, 0, 0] == 2 for frame in recording.writers[0].frames[1:])
    assert recording.env.episode_length_buf[1].item() == 2


def test_partial_episode_does_not_get_a_hit_freeze(recording):
    recording.recorder.hit_hold_s = 1.5
    recording.env._hit_flag[0] = True
    recording.env.step([])
    recording.recorder.close()
    assert recording.recorder.records[0]["hit_hold_frames"] == 0
    assert len(recording.writers[0].frames) == 1


def test_manual_reset_saves_partial_without_consuming_quota(recording):
    recording.env.step([])
    recording.env._reset_idx(torch.tensor([0]))
    assert recording.recorder.completed == 0
    assert not recording.recorder.records[0]["complete"]
    recording.env.step([0])
    assert recording.recorder.completed == 1
    assert recording.recorder.records[1]["file"] == "env0_episode_002.mp4"


def test_close_releases_partial_and_saves_manifest_idempotently(recording):
    recording.env.step([])
    recording.recorder.close()
    recording.recorder.close()
    manifest = json.loads((recording.recorder.directory / "manifest.json").read_text())
    assert recording.writers[0].released
    assert manifest["completed_episodes"] == 0
    assert len(manifest["videos"]) == 1
    assert manifest["videos"][0]["duration_s"] == pytest.approx(0.02)
    assert not manifest["videos"][0]["complete"]
    recording.env.step([0])
    assert len(recording.writers) == 1


def test_record_selected_env_one(tmp_path):
    env = FakeEnv()
    writer = MemoryWriter()
    recorder = EpisodeVideoRecorder(tmp_path / "env1", env_id=1, writer_factory=lambda path, size: writer)
    recorder._compose_frame = lambda scene, camera, status: scene
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    install_episode_video_recorder(env, recorder, lambda terminal: (frame, frame))
    env.step([0])
    assert recorder.completed == 0
    env.step([1])
    assert recorder.completed == 1
    assert recorder.records[0]["step"] == 2


def test_reject_existing_directory_and_invalid_env(tmp_path):
    with pytest.raises(FileExistsError):
        EpisodeVideoRecorder(tmp_path)
    recorder = EpisodeVideoRecorder(tmp_path / "invalid_env", env_id=2)
    with pytest.raises(ValueError, match="num_envs"):
        install_episode_video_recorder(FakeEnv(), recorder)


def test_camera_inset_rgb_is_converted_to_bgr(tmp_path):
    pytest.importorskip("cv2")
    writer = MemoryWriter()
    recorder = EpisodeVideoRecorder(tmp_path / "composite", writer_factory=lambda path, size: writer)
    scene = np.full((720, 1280, 3), (11, 22, 33), dtype=np.uint8)
    camera = np.full((576, 1024, 3), (255, 0, 0), dtype=np.uint8)
    status = {"step": 1, "fires": 0, "hit": False, "warmup": True, "ready": False}
    recorder.capture(scene, camera, status)
    assert tuple(writer.frames[0][100, 100]) == (33, 22, 11)
    assert tuple(writer.frames[0][100, 1000]) == (0, 0, 255)
    assert writer.frames[0].shape == (720, 1280, 3)
    recorder.close()


class FakeSalvoEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self._static_salvo = SimpleNamespace(
            positions=torch.zeros(2, 2, 3), alive=torch.ones(2, 2, dtype=torch.bool),
            ready=torch.ones(2, dtype=torch.bool), hits=torch.zeros(2, dtype=torch.long),
            shots=torch.zeros(2, dtype=torch.long), active=torch.zeros(2, dtype=torch.long),
            last_hit=torch.zeros(2, dtype=torch.bool), bootstrap_frames=torch.ones(2, dtype=torch.long),
            bootstrap_targets=torch.full((2,), 2, dtype=torch.long),
        )
        self._fire_command = torch.zeros(2)

    def _get_rewards(self):
        # The real salvo reward hook clears intermediate single-target hit flags.
        self._hit_flag[:] = False
        return 0

    def step(self, event):
        self.episode_length_buf += 1
        pool = self._static_salvo
        pool.last_hit[:] = False
        self._fire_command[:] = 0
        if event in ("hit", "miss"):
            self._episode_fire_count[0] += 1
            pool.shots[0] += 1
            self._fire_command[0] = 1
        if event == "hit":
            pool.hits[0] += 1
            pool.alive[0, pool.active[0]] = False
            pool.last_hit[0] = True
        self._get_rewards()
        if pool.hits[0] == 2 or event == "timeout":
            self._reset_idx(torch.tensor([0]))
        elif event == "hit":
            pool.active[0] = 1
        return "unchanged_result"

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        for env_id in env_ids:
            self._static_salvo.hits[env_id] = 0
            self._static_salvo.shots[env_id] = 0


@pytest.mark.parametrize("last_event,cleared", [("hit", True), ("timeout", False)])
def test_salvo_records_intermediate_hits_before_switch_and_keeps_one_file(tmp_path, last_event, cleared):
    env = FakeSalvoEnv()
    writers, seen = [], []
    def writer_factory(path, size):
        writer = MemoryWriter()
        writers.append(writer)
        return writer
    recorder = EpisodeVideoRecorder(tmp_path / "salvo", episodes=1, hit_hold_s=0.2, writer_factory=writer_factory)
    recorder._compose_frame = lambda scene, camera, status: scene.copy()
    def frames(terminal):
        seen.append((env._static_salvo.active[0].item(), env._static_salvo.last_hit[0].item()))
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return image, image
    install_episode_video_recorder(env, recorder, frames)
    env.step("miss")
    env.step("hit")
    assert seen == [(0, False), (0, True)]
    assert not recorder.records
    assert recorder._last_status["destroyed"] == 1
    assert recorder._last_status["remaining"] == 1
    assert recorder._last_status["hit"]
    assert len(writers[0].frames) == 2
    assert not writers[0].released
    assert env.step(last_event) == "unchanged_result"
    assert len(writers) == 1
    assert writers[0].released
    record = recorder.records[0]
    assert record["complete"]
    assert record["cleared"] == cleared
    assert record["destroyed"] == (2 if cleared else 1)
    assert record["remaining"] == (0 if cleared else 1)
    assert record["fires"] == (3 if cleared else 2)
    assert record["simulation_frames"] == 3
    assert record["hit_hold_frames"] == (10 if cleared else 0)
    assert record["accuracy"] == pytest.approx(2/3 if cleared else 1/2)


def test_salvo_composite_and_real_mp4_encoding_preserve_sensor_arrays(tmp_path):
    cv2 = pytest.importorskip("cv2")
    recorder = EpisodeVideoRecorder(tmp_path / "mp4", episodes=1)
    scene = np.full((720, 1280, 3), (11, 22, 33), dtype=np.uint8)
    camera = np.full((576, 1024, 3), (255, 0, 0), dtype=np.uint8)
    status = dict(step=1, fires=2, hit=True, warmup=False, ready=True,
                  targets=100, destroyed=1, remaining=99, cleared=False,
                  target_id=0, accuracy=.5, shot_fired=True,
                  tracks=[dict(id=0, box=[100., 100., 120., 120.], status="tracking")])
    composite = recorder._compose_frame(scene, camera, status)
    assert tuple(composite[300, 300]) == (0, 0, 255)
    assert tuple(composite[100, 1000]) == (33, 22, 11)
    recorder.capture(scene, camera, status)
    recorder.capture(scene, camera, {**status, "step": 2, "hit": False})
    recorder.finish_episode(complete=True)
    recorder.close()
    np.testing.assert_array_equal(camera, np.full_like(camera, (255, 0, 0)))
    np.testing.assert_array_equal(scene, np.full_like(scene, (11, 22, 33)))
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert manifest["layout"] == "robot_rgb_tracks_with_scene_inset"
    assert manifest["videos"][0]["remaining"] == 99
    capture = cv2.VideoCapture(str(recorder.directory / "env0_episode_001.mp4"))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(50.)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    ok, frame = capture.read()
    assert ok and frame.shape == (720, 1280, 3)
    capture.release()
