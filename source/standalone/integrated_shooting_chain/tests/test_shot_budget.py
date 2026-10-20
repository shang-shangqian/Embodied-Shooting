"""Exercise the real termination/reward methods on CPU, without Isaac Sim."""

import argparse
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


ENV_PATH = Path(__file__).resolve().parents[2] / "test7_2_weight/turret_aim_env_gt.py"


@pytest.fixture
def env():
    tree = ast.parse(ENV_PATH.read_text())
    names = {"_get_dones", "_get_rewards", "_compute_current_step_hit"}
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "quat_apply": lambda quat, vec: vec}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    count = 4
    cfg = SimpleNamespace(
        max_shots_per_episode=5, failure_height_threshold=0.1, failure_tilt_cos_threshold=0.5,
        enable_vision_supervision=False, target_out_of_view_grace_frames=3,
        r_first_shot_hit_bonus=50.0, r_repeat_fire_penalty=-4.0, hit_radius=0.1,
        r_no_fire_timeout_penalty=-5.0, w_yaw_align=0.1, w_pitch_align=0.1,
        w_ballistic=0.8, r_hit=100.0, r_fire_cost=-0.8, w_rate=-0.01,
        w_action_change=-0.01, r_time_penalty=-0.7, max_hit_distance=170.0,
    )
    instance = SimpleNamespace(
        cfg=cfg, device="cpu", num_envs=count, max_episode_length=500,
        turret=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.ones(count, 3),
                                                    root_quat_w=torch.zeros(count, 4))),
        _get_invalid_state_mask=lambda: torch.zeros(count, dtype=torch.bool),
        _episode_fire_count=torch.zeros(count, dtype=torch.long),
        _fire_command=torch.zeros(count), _aim_warmup_left_steps=torch.zeros(count, dtype=torch.long),
        _delta_pos=torch.zeros(count, 2), _actions=torch.zeros(count, 3),
        _prev_actions=torch.zeros(count, 3),
        _compute_muzzle_actual_yaw_pitch_in_yaw_only_base=lambda: (torch.zeros(count), torch.zeros(count)),
        _compute_muzzle_based_aim_refs=lambda: (torch.zeros(count),) * 3,
        _compute_soft_band_rewards=lambda **kwargs: (torch.zeros(count), torch.zeros(count)),
        _compute_angle_error=lambda: (None, torch.ones(count), None),
        _record_fire_ballistic_stats=lambda *args: None,
        _get_cached_ballistic_metrics=lambda: (torch.zeros(count), torch.tensor([False, True, False, False]), None),
        _collect_eval_details=True, _eval_samples_limit=10,
        _episode_start_target_pos_w=torch.zeros(count, 3), _eval_target_spawn_sum=torch.zeros(3),
        _eval_target_spawn_samples=[], _eval_episode_fire_counts=[], _eval_episode_hits=[],
        _vision_camera=None,
    )
    for name in ("_hit_flag", "_shot_budget_exhausted", "_target_out_of_view_terminated",
                 "_target_out_of_view_termination_mask", "_pending_target_resample"):
        setattr(instance, name, torch.zeros(count, dtype=torch.bool))
    for name in ("episode_length_buf", "_target_out_of_view_streak", "_vision_collect_warmup_left"):
        setattr(instance, name, torch.zeros(count, dtype=torch.long))
    for name in ("episodes", "hits", "timeouts", "fallen", "invalid", "target_out_of_view",
                 "shot_budget_exhausted"):
        setattr(instance, f"_eval_total_{name}", torch.zeros((), dtype=torch.long))
    instance._eval_target_spawn_count = torch.zeros((), dtype=torch.long)
    for name in names:
        setattr(instance, name, MethodType(namespace[name], instance))
    return instance


def test_fifth_shot_hit_is_success_and_fifth_miss_is_terminal(env):
    env._episode_fire_count[:] = torch.tensor([4, 5, 5, 6])
    env._fire_command[:] = 1
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [False, True, True, True]
    assert not truncated.any()
    assert env._hit_flag.tolist() == [False, True, False, False]
    assert env._shot_budget_exhausted.tolist() == [False, False, True, True]
    assert env._eval_total_hits == 1
    assert env._eval_total_shot_budget_exhausted == 2
    assert env._eval_episode_fire_counts == [5, 5, 6]
    assert env._eval_episode_hits == [1, 0, 0]


def test_fifth_shot_reward_survives_done_before_reward_order(env):
    env._episode_fire_count[:] = 5
    env._fire_command[:] = 1
    env._get_dones()  # DirectRLEnv evaluates dones before rewards.
    reward = env._get_rewards()
    assert reward[1] > 0
    assert reward[2] < 0
    env.cfg.max_shots_per_episode = 0
    torch.testing.assert_close(reward, env._get_rewards())


@pytest.mark.parametrize("cap", [0, 5])
def test_no_fire_still_times_out_and_ordinary_episodes_continue(env, cap):
    env.cfg.max_shots_per_episode = cap
    env.episode_length_buf[:] = torch.tensor([498, 499, 500, 0])
    terminated, truncated = env._get_dones()
    assert not terminated.any()
    assert truncated.tolist() == [False, True, True, False]
    assert not env._shot_budget_exhausted.any()


def test_disabled_budget_preserves_unlimited_evaluation(env):
    env.cfg.max_shots_per_episode = 0
    env._episode_fire_count[:] = 30
    terminated, truncated = env._get_dones()
    assert not terminated.any()
    assert not truncated.any()


def test_only_actual_fires_count_toward_budget(env):
    method = next(n for n in ast.walk(ast.parse(ENV_PATH.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "_pre_physics_step")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(ast.unparse(t) == "fire_now" for t in n.targets))
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.AugAssign)
               and ast.unparse(n.target) == "self._episode_fire_count") + 1
    env._eval_total_fires = torch.zeros((), dtype=torch.long)
    exec(compile(ast.Module(body=method.body[start:end], type_ignores=[]), str(ENV_PATH), "exec"), {
        "self": env, "torch": torch, "pred_ready": torch.tensor([True, False, True, True]),
        "aiming_enabled": torch.tensor([True, True, False, True]),
        "can_fire": torch.tensor([True, True, True, False]),
        "intent_confirmed": torch.ones(4, dtype=torch.bool),
    })
    assert env._episode_fire_count.tolist() == [1, 0, 0, 0]
    assert env._eval_total_fires == 1


def test_partial_reset_clears_only_selected_episode_budget(env):
    env._episode_fire_count[:] = torch.tensor([5, 2, 5, 3])
    env._get_dones()
    method = next(n for n in ast.walk(ast.parse(ENV_PATH.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx")
    statements = [n for n in method.body if isinstance(n, ast.Assign)
                  and any(ast.unparse(t) in ("self._episode_fire_count[env_ids_t]",
                                             "self._shot_budget_exhausted[env_ids_t]") for t in n.targets)]
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(ENV_PATH), "exec"),
         {"self": env, "env_ids_t": torch.tensor([0])})
    assert env._episode_fire_count.tolist() == [0, 2, 5, 3]
    assert env._shot_budget_exhausted.tolist() == [False, False, True, False]
    env.episode_length_buf[2] = 499
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [False, False, True, False]
    assert not truncated.any()  # Budget failure must not bootstrap as a timeout.


def test_fifth_shot_hit_at_time_limit_is_only_success(env):
    env._episode_fire_count[:] = 5
    env._fire_command[:] = 1
    env.episode_length_buf[:] = 499
    terminated, truncated = env._get_dones()
    assert terminated.all()
    assert not truncated.any()
    assert env._eval_total_hits == 1
    assert env._eval_total_shot_budget_exhausted == 3
    assert env._eval_total_timeouts == 0


@pytest.mark.parametrize("script", ["train_turret_mixed_gt.py", "eval_with_track_gt.py"])
def test_cli_budget_is_opt_in_and_reaches_environment(script):
    path = ENV_PATH.parent / script
    tree = ast.parse(path.read_text())
    arg = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and n.value.args and isinstance(n.value.args[0], ast.Constant)
               and n.value.args[0].value == "--max_shots_per_episode")
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[arg], type_ignores=[]), str(path), "exec"), {"parser": parser})
    assert parser.parse_args([]).max_shots_per_episode == 0
    args = parser.parse_args(["--max_shots_per_episode", "5"])
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(ast.unparse(t) == "env_cfg.max_shots_per_episode" for t in n.targets))
    cfg = SimpleNamespace()
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
         {"env_cfg": cfg, "args": args})
    assert cfg.max_shots_per_episode == 5


@pytest.fixture
def case_metrics(env):
    """Bind the actual evaluation accounting to the environment termination fixture."""
    path = ENV_PATH.parent / "eval_with_track_gt.py"
    tree = ast.parse(path.read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_install_mixed_training_logic")
    initializers = [n for n in install.body if isinstance(n, ast.Assign)
                    and any(ast.unparse(t).startswith("base_env._mix_eval_case_") for t in n.targets)]
    names = {"_record_completed_case_episodes", "get_mix_case_stats", "reset_eval_metrics_with_case_stats"}
    methods = [n for n in install.body if isinstance(n, ast.FunctionDef) and n.name in names]
    reset_calls = []
    namespace = {"torch": torch, "base_env": env,
                 "original_reset_eval_metrics": lambda: reset_calls.append(True)}
    exec(compile(ast.Module(body=initializers + methods, type_ignores=[]), str(path), "exec"), namespace)
    for name in names:
        setattr(env, name, MethodType(namespace[name], env))
    env._mix_case_id = torch.tensor([0, 1, 1, 0])
    env.turret.data.projected_gravity_b = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    return env, reset_calls


def test_eval_counts_fifth_shot_miss_hit_and_timeout_by_case(case_metrics):
    env, reset_calls = case_metrics
    env._episode_fire_count[:] = torch.tensor([5, 5, 5, 0])
    env._fire_command[:] = torch.tensor([1., 1., 1., 0.])
    env.episode_length_buf[:] = torch.tensor([100, 100, 100, 499])
    terminated, env.reset_time_outs = env._get_dones()
    completed = torch.nonzero(terminated | env.reset_time_outs).squeeze(-1)
    env._record_completed_case_episodes(completed)
    metrics = env.get_mix_case_stats()
    assert metrics["total"] == 4
    assert metrics["ss_episodes"] == metrics["sd_episodes"] == 2
    assert metrics["ss_shot_budget_exhausted"] == metrics["sd_shot_budget_exhausted"] == 1
    assert metrics["ss_shot_budget_exhausted_rate"] == metrics["sd_shot_budget_exhausted_rate"] == 0.5
    assert metrics["ss_timeouts"] == 1
    assert metrics["sd_timeouts"] == 0
    assert metrics["sd_hits"] == 1  # The fifth-shot hit is not a spent-budget failure.
    assert metrics["ss_fires"] == 5
    assert metrics["sd_fires"] == 10
    assert metrics["sd_accuracy_rate"] == 0.1
    assert metrics["dd_shot_budget_exhausted_rate"] == 0
    env.reset_eval_metrics_with_case_stats()
    assert reset_calls == [True]
    assert env.get_mix_case_stats()["total"] == 0
    assert all(not v.any() for k, v in vars(env).items() if k.startswith("_mix_eval_case_"))


def test_eval_skips_initial_reset_and_keeps_unlimited_budget_disabled(case_metrics):
    env, _ = case_metrics
    env.cfg.max_shots_per_episode = 0
    env._episode_fire_count[:] = 12
    env._fire_command[:] = torch.tensor([0., 1., 0., 0.])
    env.episode_length_buf[:] = torch.tensor([0, 499, 0, 0])
    _, env.reset_time_outs = env._get_dones()
    env._record_completed_case_episodes(torch.arange(4))
    metrics = env.get_mix_case_stats()
    assert metrics["total"] == 1
    assert metrics["sd_fires"] == 12
    assert metrics["sd_shot_budget_exhausted"] == metrics["ss_shot_budget_exhausted"] == 0
    assert metrics["ss_episodes"] == 0
