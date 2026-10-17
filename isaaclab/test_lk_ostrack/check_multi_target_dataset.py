from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_lk_ostrack.isaacsim_dataset import validate_collection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a collected multi-target dataset")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    summary = validate_collection_dataset(args.dataset_dir)
    print(
        "[dataset-check] valid",
        f"sequences={summary['sequences']}",
        f"frames={summary['frames']}",
        f"targets_per_sequence_sum={summary['targets_per_sequence_sum']}",
    )


if __name__ == "__main__":
    main()
