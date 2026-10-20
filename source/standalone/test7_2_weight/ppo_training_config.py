import argparse
import math


def add_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--horizon_length", type=int, default=None)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--mini_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_schedule", choices=("none", "adaptive"), default=None)
    parser.add_argument("--seq_length", type=int, default=8)
    parser.add_argument("--e_clip", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.5)
    parser.add_argument("--kl_threshold", type=float, default=0.01)
    parser.add_argument("--save_frequency", type=int, default=None)
    parser.add_argument(
        "--reset_learning_rate", action=argparse.BooleanOptionalAction, default=None,
        help="After restoring a checkpoint, apply the configured learning rate instead of its optimizer rate. "
             "Defaults to enabled for full_chain and disabled for gt.",
    )


def resolve_ppo_settings(args: argparse.Namespace) -> dict:
    full_chain = args.training_mode == "full_chain"
    defaults = {
        "num_envs": 16 if full_chain else 64,
        "horizon_length": 128 if full_chain else 32,
        "mini_epochs": 2 if full_chain else 4,
        "learning_rate": 3e-6 if full_chain else 5e-5,
        "lr_schedule": "none" if full_chain else "adaptive",
        "save_frequency": 50 if full_chain else 0,
        "save_best_after": 0 if full_chain else 11500,
        "reset_learning_rate": full_chain,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    for name in ("num_envs", "horizon_length", "mini_epochs", "seq_length"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be > 0")
    for name in ("learning_rate", "grad_norm", "kl_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and > 0")
    if not math.isfinite(args.e_clip) or not 0 < args.e_clip < 1:
        raise ValueError("--e_clip must be finite and between 0 and 1")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("--entropy_coef must be finite and >= 0")
    for name in ("save_frequency", "save_best_after"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if args.horizon_length % args.seq_length != 0:
        raise ValueError("--horizon_length must be divisible by --seq_length")

    rollout_batch_size = args.num_envs * args.horizon_length
    if args.minibatch_size is None:
        args.minibatch_size = min(max(1024 if full_chain else 2048, args.seq_length), rollout_batch_size)
        while args.minibatch_size % args.seq_length or rollout_batch_size % args.minibatch_size:
            args.minibatch_size -= 1
    if not 0 < args.minibatch_size <= rollout_batch_size:
        raise ValueError("--minibatch_size must be > 0 and <= num_envs * horizon_length")
    if args.minibatch_size % args.seq_length or rollout_batch_size % args.minibatch_size:
        raise ValueError("--minibatch_size must divide the rollout batch and be divisible by --seq_length")

    return {
        "horizon_length": args.horizon_length,
        "minibatch_size": args.minibatch_size,
        "mini_epochs": args.mini_epochs,
        "learning_rate": args.learning_rate,
        "lr_schedule": None if args.lr_schedule == "none" else args.lr_schedule,
        "seq_length": args.seq_length,
        "entropy_coef": args.entropy_coef,
        "e_clip": args.e_clip,
        "grad_norm": args.grad_norm,
        "kl_threshold": args.kl_threshold,
        "save_frequency": args.save_frequency,
        "save_best_after": args.save_best_after,
    }


def make_learning_rate_restore_agent(agent_class, *, reset_learning_rate: bool):
    class LearningRateRestoreAgent(agent_class):
        def restore(self, checkpoint_path, set_epoch=True):
            result = super().restore(checkpoint_path, set_epoch=set_epoch)
            checkpoint_rate = float(self.optimizer.param_groups[0]["lr"])
            self.last_lr = float(self.config["learning_rate"]) if reset_learning_rate else checkpoint_rate
            self.update_lr(self.last_lr)
            print(
                f"[train-mixed] restored epoch={self.epoch_num} "
                f"checkpoint_lr={checkpoint_rate:.6g} effective_lr={self.last_lr:.6g} "
                f"lr_schedule={self.config.get('lr_schedule')}"
            )
            return result

    return LearningRateRestoreAgent
