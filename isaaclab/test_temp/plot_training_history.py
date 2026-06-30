from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to draw training history plots. "
            "Install it first, for example: pip install matplotlib"
        ) from exc


def resolve_history_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_dir():
        path = path / "history.json"
    if not path.is_file():
        raise FileNotFoundError(f"history.json not found: {path}")
    return path


def load_history(history_path: Path) -> list[dict[str, Any]]:
    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("history"), list):
        data = data["history"]

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{history_path} must contain a non-empty list of epoch records.")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"History item at index {idx} is not an object.")
        records.append(item)
    return records


def number_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def get_values(records: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = None
        for key in keys:
            if key in record:
                value = record[key]
                break
        values.append(number_or_nan(value))
    return values


def has_data(values: list[float]) -> bool:
    return any(math.isfinite(v) for v in values)


def plot_line(ax, epochs: list[float], values: list[float], label: str) -> bool:
    if not has_data(values):
        return False
    ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.6, label=label)
    return True


def mark_best_val_loss(ax, epochs: list[float], val_loss: list[float]) -> None:
    finite_pairs = [(epoch, value) for epoch, value in zip(epochs, val_loss) if math.isfinite(value)]
    if not finite_pairs:
        return

    best_epoch, best_loss = min(finite_pairs, key=lambda pair: pair[1])
    ax.scatter([best_epoch], [best_loss], s=55, color="#d62728", zorder=5, label=f"best val {best_loss:.6f}")
    ax.annotate(
        f"best epoch {int(best_epoch)}",
        xy=(best_epoch, best_loss),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )


def draw_history(records: list[dict[str, Any]], output_path: Path, dpi: int) -> None:
    plt = _load_pyplot()

    epochs = get_values(records, "epoch")
    if not has_data(epochs):
        epochs = [float(i) for i in range(1, len(records) + 1)]

    train_loss = get_values(records, "train_loss")
    val_loss = get_values(records, "val_loss", "loss")
    mae_x = get_values(records, "mae_x")
    mae_y = get_values(records, "mae_y")
    mae_z = get_values(records, "mae_z")
    rmse_x = get_values(records, "rmse_x")
    rmse_y = get_values(records, "rmse_y")
    rmse_z = get_values(records, "rmse_z")
    lr = get_values(records, "lr", "learning_rate")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("GRU Velocity Training History", fontsize=15)

    ax = axes[0][0]
    plotted = False
    plotted |= plot_line(ax, epochs, train_loss, "train_loss")
    plotted |= plot_line(ax, epochs, val_loss, "val_loss")
    mark_best_val_loss(ax, epochs, val_loss)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if plotted:
        ax.legend()

    ax = axes[0][1]
    plotted = False
    plotted |= plot_line(ax, epochs, mae_x, "mae_x")
    plotted |= plot_line(ax, epochs, mae_y, "mae_y")
    plotted |= plot_line(ax, epochs, mae_z, "mae_z")
    ax.set_title("Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("m/s")
    if plotted:
        ax.legend()

    ax = axes[1][0]
    plotted = False
    plotted |= plot_line(ax, epochs, rmse_x, "rmse_x")
    plotted |= plot_line(ax, epochs, rmse_y, "rmse_y")
    plotted |= plot_line(ax, epochs, rmse_z, "rmse_z")
    ax.set_title("Validation RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("m/s")
    if plotted:
        ax.legend()

    ax = axes[1][1]
    plotted = plot_line(ax, epochs, lr, "lr")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    if plotted:
        ax.set_yscale("log")
        ax.legend()

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from train_gru_velocity.py history.json.")
    parser.add_argument(
        "history",
        type=str,
        help="Path to history.json, or a run directory that contains history.json.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output png path. Defaults to <run_dir>/training_history.png.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    history_path = resolve_history_path(args.history)
    output_path = Path(args.out).expanduser() if args.out else history_path.parent / "training_history.png"

    records = load_history(history_path)
    draw_history(records, output_path=output_path, dpi=args.dpi)
    print(f"[plot] saved: {output_path}")


if __name__ == "__main__":
    main()
