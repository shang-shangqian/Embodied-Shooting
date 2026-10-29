"""Small IO helpers shared by the velocity error validation tools.

The collector stores model inputs in ``*.taqstream`` files and the audit-only
fields may be emitted separately as CSV/JSONL.  Keeping the audit fields out
of the model payload makes accidental GT leakage easier to detect.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_NAMES = {0: "initial", 1: "flow", 2: "ostrack", 3: "prediction", 4: "lost"}

NON_AUDIT_REPORT_FILES = {
    "frame_metrics.csv",
    "episode_metrics.csv",
    "grouped_metrics.csv",
    "paired_effects.csv",
    "event_catalog.csv",
    "lagged_correlations.csv",
    "time_offset_scan.csv",
    "summary.json",
    "collection_manifest.json",
}


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text


def load_audit_rows(root: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Load audit rows keyed by ``(episode_uid, frame_index)``.

    CSV and JSONL are supported.  A JSON file may contain either a list of
    rows or ``{"rows": [...]}``.  Files without an episode UID are ignored;
    silently guessing the episode from a filename can merge unrelated data.
    """
    if root is None:
        return {}
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"audit path does not exist: {root}")
    paths = [root] if root.is_file() else sorted(
        [*root.rglob("*.csv"), *root.rglob("*.jsonl"), *root.rglob("*.json")]
    )
    if root.is_dir():
        paths = [path for path in paths if path.name not in NON_AUDIT_REPORT_FILES]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        try:
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    rows: Iterable[dict[str, Any]] = csv.DictReader(handle)
                    rows = [{key: parse_scalar(value) for key, value in row.items()} for row in rows]
            elif path.suffix.lower() == ".jsonl":
                rows = []
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"{path}:{line_number} must contain one JSON object"
                        )
                    rows.append(row)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload = payload.get("rows", payload.get("frames", []))
                if not isinstance(payload, list):
                    continue
                rows = [dict(row) for row in payload if isinstance(row, dict)]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load audit file {path}: {exc}") from exc
        for row in rows:
            episode_uid = row.get("episode_uid")
            frame_index = row.get("frame_index", row.get("frame", row.get("index")))
            if episode_uid is None or frame_index is None:
                continue
            try:
                key = (str(episode_uid), int(frame_index))
            except (TypeError, ValueError):
                continue
            if key in result:
                raise ValueError(
                    f"duplicate audit key episode_uid={key[0]!r}, frame_index={key[1]}"
                )
            result[key] = row
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float("inf") else default
