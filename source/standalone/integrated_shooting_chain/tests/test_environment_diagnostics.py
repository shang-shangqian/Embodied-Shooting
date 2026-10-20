import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ENV_PATH = Path(__file__).resolve().parents[2] / "test7_2_weight/turret_aim_env_gt.py"


@pytest.mark.parametrize("enabled", [False, True])
def test_lidar_debug_logging_does_not_disable_measured_range(enabled, capsys):
    tree = ast.parse(ENV_PATH.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "get_vision_supervision_batch")
    end = next(index for index, node in enumerate(method.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "refined_yaw" for target in node.targets))
    env = SimpleNamespace(
        cfg=SimpleNamespace(debug_diagnostics=enabled, enable_dynamic_target_raycast_debug=True),
        _vision_rgb_history=torch.zeros(1), num_envs=1, device="cpu",
        turret=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.zeros(1, 3))),
        _vision_distance_gt=torch.tensor([[100.0]]), _dynamic_target_raycast_sensor=object(),
        _raycast_hit_distance=torch.tensor([[70.0]]), _raycast_hit_valid=torch.tensor([True]),
    )
    namespace = {"self": env, "torch": torch}
    exec(compile(ast.Module(body=method.body[:end], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    assert namespace["lidar_distance"].item() == 70.0
    assert namespace["lidar_valid_mask"].item()
    output = capsys.readouterr().out
    assert ("[vision batch]" in output) is enabled
    assert env._printed_lidar_source_once is enabled


@pytest.mark.parametrize("enabled,expected", [(False, False), (True, True)])
def test_periodic_camera_raycast_diagnostics_have_separate_log_guard(enabled, expected):
    tree = ast.parse(ENV_PATH.read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and "_raycast_debug_counter % every" in ast.unparse(node.test))
    namespace = {"self": SimpleNamespace(cfg=SimpleNamespace(debug_diagnostics=enabled), _raycast_debug_counter=20),
                 "every": 20, "num_envs": 1}
    result = eval(compile(ast.Expression(guard.test), str(ENV_PATH), "eval"), namespace)
    assert result is expected


@pytest.mark.parametrize("script", ["train_turret_mixed_gt.py", "eval_with_track_gt.py"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_diagnostics_setting_keeps_full_chain_raycast_enabled(script, enabled):
    path = ENV_PATH.parent / script
    tree = ast.parse(path.read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) in ("env_cfg.debug_diagnostics",
                                                  "env_cfg.enable_dynamic_target_raycast_debug")
                           for target in node.targets)]
    assert len(assignments) == 2
    cfg = SimpleNamespace()
    namespace = {"env_cfg": cfg, "args": SimpleNamespace(debug_diagnostics=enabled), "full_chain": True}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert cfg.debug_diagnostics is enabled
    assert cfg.enable_dynamic_target_raycast_debug is True
