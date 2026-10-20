import argparse
import ast
from pathlib import Path
import sys

import pytest
import torch
from rl_games.common.schedulers import AdaptiveScheduler, IdentityScheduler


TRAINING_ROOT = Path(__file__).resolve().parents[2] / "test7_2_weight"
sys.path.insert(0, str(TRAINING_ROOT))

from ppo_training_config import add_ppo_arguments, make_learning_rate_restore_agent, resolve_ppo_settings


def parse_training_args(*options):
    path = TRAINING_ROOT / "train_turret_mixed_gt.py"
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    end = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets))
    namespace = {"argparse": argparse, "add_ppo_arguments": add_ppo_arguments}
    exec(compile(ast.Module(body=main.body[:end], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parser"].parse_args(["--locomotion_policy", "policy.pt", *options])


def test_full_chain_defaults_match_conservative_finetuning():
    args = parse_training_args("--training_mode", "full_chain")
    settings = resolve_ppo_settings(args)
    assert args.num_envs == 16
    assert args.reset_learning_rate is True
    assert args.num_envs * settings["horizon_length"] == 2048
    assert settings == {
        "horizon_length": 128, "minibatch_size": 1024, "mini_epochs": 2,
        "learning_rate": 3e-6, "lr_schedule": None, "seq_length": 8,
        "entropy_coef": 0.003, "e_clip": 0.1, "grad_norm": 0.5,
        "kl_threshold": 0.01, "save_frequency": 50, "save_best_after": 0,
    }


def test_gt_defaults_keep_original_training_settings():
    args = parse_training_args()
    settings = resolve_ppo_settings(args)
    assert args.num_envs == 64
    assert args.reset_learning_rate is False
    assert settings["horizon_length"] == 32
    assert settings["minibatch_size"] == 2048
    assert settings["mini_epochs"] == 4
    assert settings["learning_rate"] == 5e-5
    assert settings["lr_schedule"] == "adaptive"
    assert settings["save_frequency"] == 0
    assert settings["save_best_after"] == 11500


def test_explicit_cli_settings_override_full_chain_defaults():
    args = parse_training_args(
        "--training_mode", "full_chain", "--num_envs", "8", "--horizon_length", "256",
        "--minibatch_size", "512", "--mini_epochs", "1", "--learning_rate", "5e-6",
        "--lr_schedule", "adaptive", "--seq_length", "16", "--entropy_coef", "0.0003",
        "--e_clip", "0.05", "--grad_norm", "0.25", "--kl_threshold", "0.005",
        "--save_frequency", "25", "--save_best_after", "0", "--no-reset_learning_rate",
    )
    settings = resolve_ppo_settings(args)
    assert args.num_envs == 8
    assert args.reset_learning_rate is False
    assert settings == {
        "horizon_length": 256, "minibatch_size": 512, "mini_epochs": 1,
        "learning_rate": 5e-6, "lr_schedule": "adaptive", "seq_length": 16,
        "entropy_coef": 0.0003, "e_clip": 0.05, "grad_norm": 0.25,
        "kl_threshold": 0.005, "save_frequency": 25, "save_best_after": 0,
    }


@pytest.mark.parametrize("options", [
    ("--num_envs", "0"), ("--horizon_length", "0"), ("--horizon_length", "127"),
    ("--mini_epochs", "0"), ("--seq_length", "0"), ("--minibatch_size", "0"),
    ("--minibatch_size", "513"), ("--minibatch_size", "4096"),
    ("--learning_rate", "0"), ("--learning_rate", "nan"), ("--learning_rate", "inf"),
    ("--grad_norm", "0"), ("--kl_threshold", "0"), ("--e_clip", "1"),
    ("--entropy_coef=-0.1",), ("--save_frequency=-1",), ("--save_best_after=-1",),
])
def test_invalid_ppo_settings_fail_before_simulation(options):
    args = parse_training_args("--training_mode", "full_chain", *options)
    with pytest.raises(ValueError):
        resolve_ppo_settings(args)


@pytest.mark.parametrize("options,expected", [
    (("--num_envs", "1"), 128),
    (("--num_envs", "9"), 576),
    (("--horizon_length", "2048", "--seq_length", "2048"), 2048),
])
def test_automatic_minibatch_respects_sequence_and_batch_divisibility(options, expected):
    args = parse_training_args("--training_mode", "full_chain", *options)
    settings = resolve_ppo_settings(args)
    assert settings["minibatch_size"] == expected
    assert args.num_envs * settings["horizon_length"] % expected == 0
    assert expected % settings["seq_length"] == 0


class CheckpointAgent:
    def __init__(self, config):
        self.config = config
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.last_lr = config["learning_rate"]
        self.epoch_num = 0
        self.frame = 0

    def restore(self, checkpoint_path, set_epoch=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if set_epoch:
            self.epoch_num = checkpoint["epoch"]
            self.frame = checkpoint["frame"]
        return "restored"

    def update_lr(self, learning_rate):
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate


@pytest.mark.parametrize("reset_learning_rate", [False, True])
@pytest.mark.parametrize("schedule", [None, "adaptive"])
def test_restore_synchronizes_lr_before_first_update_and_preserves_checkpoint(
    tmp_path, reset_learning_rate, schedule,
):
    source_agent = CheckpointAgent({"learning_rate": 1.5e-6})
    source_agent.model(torch.ones(1, 1)).sum().backward()
    source_agent.optimizer.step()
    checkpoint_path = tmp_path / "original.pt"
    torch.save({
        "model": source_agent.model.state_dict(), "optimizer": source_agent.optimizer.state_dict(),
        "epoch": 114000, "frame": 1000000,
    }, checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()
    agent_class = make_learning_rate_restore_agent(CheckpointAgent, reset_learning_rate=reset_learning_rate)
    agent = agent_class({"learning_rate": 3e-6, "lr_schedule": schedule})
    assert agent.restore(checkpoint_path) == "restored"
    expected_rate = 3e-6 if reset_learning_rate else 1.5e-6
    assert agent.last_lr == expected_rate
    assert agent.optimizer.param_groups[0]["lr"] == expected_rate
    assert agent.epoch_num == 114000
    assert agent.frame == 1000000
    for name, value in source_agent.model.state_dict().items():
        torch.testing.assert_close(agent.model.state_dict()[name], value)
    for state_id, state in source_agent.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(agent.optimizer.state_dict()["state"][state_id][name], value)
    scheduler = IdentityScheduler() if schedule is None else AdaptiveScheduler(0.01)
    for _ in range(4):
        agent.optimizer.zero_grad()
        agent.model(torch.ones(1, 1)).sum().backward()
        agent.optimizer.step()
        agent.last_lr, _ = scheduler.update(agent.last_lr, 0.003, 114001, 0, 0.01)
        agent.update_lr(agent.last_lr)
        assert agent.optimizer.param_groups[0]["lr"] == expected_rate
    assert checkpoint_path.read_bytes() == original_bytes


def test_training_entrypoint_passes_settings_and_registers_restore_agent():
    tree = ast.parse((TRAINING_ROOT / "train_turret_mixed_gt.py").read_text())
    expanded = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                and any(key is None and isinstance(value, ast.Name) and value.id == "ppo_settings"
                        for key, value in zip(node.keys, node.values))]
    assert len(expanded) == 1
    assert {key.value for key in expanded[0].keys if isinstance(key, ast.Constant)}.isdisjoint({
        "horizon_length", "minibatch_size", "mini_epochs", "learning_rate", "lr_schedule",
        "seq_length", "save_frequency", "save_best_after", "entropy_coef", "e_clip", "grad_norm",
    })
    registrations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "register_builder"]
    assert any(node.args[0].value == "a2c_continuous" for node in registrations)
