"""Forward historical module and script paths to the organized packages."""

from importlib import import_module
import sys
from typing import Any


def forward_module(module_name: str, target: str, namespace: dict[str, Any]) -> None:
    implementation = import_module(f"speed_prediction.{target}")
    if module_name == "__main__":
        entry_point = getattr(implementation, "main", None)
        if entry_point is not None:
            entry_point()
    else:
        namespace.update(
            {name: value for name, value in vars(implementation).items() if not name.startswith("__")}
        )
        sys.modules[module_name] = implementation
