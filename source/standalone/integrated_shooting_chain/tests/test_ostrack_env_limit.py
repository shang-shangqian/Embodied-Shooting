"""Exercise CLI preflight without importing the Isaac Sim runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("script", ["eval_with_track_gt.py", "train_turret_mixed_gt.py"])
@pytest.mark.parametrize("num_envs", [4, 8])
def test_ostrack_multi_environment_preflight_warns_without_rejecting(script, num_envs, capsys):
    path = Path(__file__).resolve().parents[2] / "test7_2_weight" / script
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    advisory = next(node for node in main.body if isinstance(node, ast.If)
                    and "args.fine_tracking" in ast.unparse(node.test)
                    and "args.num_envs" in ast.unparse(node.test))
    namespace = {"full_chain": True, "args": SimpleNamespace(fine_tracking=True, num_envs=num_envs)}
    exec(compile(ast.Module(body=[advisory], type_ignores=[]), str(path), "exec"), namespace)
    output = capsys.readouterr().out
    assert f"{num_envs} envs" in output
    assert "one shared network and independent tracking state" in output
    assert "no hard environment limit" in output
