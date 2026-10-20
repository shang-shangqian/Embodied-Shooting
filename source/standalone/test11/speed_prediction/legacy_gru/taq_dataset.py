"""Adapt TAQ-v2 windows to the unchanged legacy GRU's 13-input contract."""

from collections import defaultdict
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..common.time_aware_velocity_data import (
    TAQ_LABEL_SEMANTICS,
    TAQ_SCHEMA_VERSION,
    TAQ_YAW_RATE_SEMANTICS,
    load_torch_payload,
)


def discover_chunks(data_dirs):
    files = set()
    for directory in data_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {root}")
        files.update(root.glob("*.pt"))
        files.update(root.rglob("taq_velocity_chunk_*.pt"))
        files.update(root.rglob("velocity_seq_chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"No velocity chunks found in: {data_dirs}")
    return sorted(files)


def chunk_identity(path):
    payload = load_torch_payload(path)
    metadata = payload.get("metadata", payload.get("meta", {}))
    schema = payload.get("schema_version", "legacy_sequence_data")
    if "sequence_features" in payload:
        if schema != TAQ_SCHEMA_VERSION:
            raise ValueError(f"{path}: only TAQ-v2 has supported physical velocity labels")
        semantics = payload.get("label_semantics", metadata.get("label_semantics"))
        if semantics != TAQ_LABEL_SEMANTICS:
            raise ValueError(f"{path}: incompatible velocity label semantics")
        if metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
            raise ValueError(f"{path}: incompatible heading-rate semantics")
        episode = metadata.get("episode_uid")
        if not episode:
            raise ValueError(f"{path}: TAQ-v2 requires episode_uid for leakage-free splitting")
    elif "sequence_data" in payload:
        episode = metadata.get("episode_uid")
        if not episode:
            episode = str(path.parent) if path.parent.name.startswith("episode_") else str(path)
    else:
        raise ValueError(f"{path}: not a velocity dataset chunk")
    scenario = metadata.get("scenario")
    if not scenario:
        scenario = next((part for part in reversed(path.parts) if part in {"ss", "sd", "ds", "dd"}), "unknown")
    return str(schema), str(scenario), str(episode)


def split_episode_files(files, train_ratio, test_ratio, seed):
    if not 0 < train_ratio < 1 or not 0 <= test_ratio < 1 or train_ratio + test_ratio >= 1:
        raise ValueError("Require 0 < train_ratio, 0 <= test_ratio and train_ratio + test_ratio < 1")
    groups = defaultdict(dict)
    identities = {}
    schemas = set()
    for path in sorted(files):
        schema, scenario, episode = chunk_identity(path)
        schemas.add(schema)
        if episode in identities and identities[episode] != scenario:
            raise ValueError(f"Episode {episode!r} occurs in multiple scenarios")
        identities[episode] = scenario
        groups[scenario].setdefault(episode, []).append(path)
    if len(schemas) != 1:
        raise ValueError("Do not mix legacy chunks and TAQ-v2 physical velocity labels in one run")
    randomizer = random.Random(seed)
    splits = {name: [] for name in ("train", "val", "test")}
    episode_counts = {name: {} for name in splits}
    for scenario, episodes in sorted(groups.items()):
        identifiers = sorted(episodes)
        randomizer.shuffle(identifiers)
        minimum = 3 if test_ratio > 0 else 2
        if len(identifiers) < minimum:
            raise ValueError(f"Scenario {scenario!r} needs at least {minimum} independent episodes")
        test_count = max(1, int(len(identifiers) * test_ratio)) if test_ratio > 0 else 0
        train_count = min(max(1, int(len(identifiers) * train_ratio)), len(identifiers) - test_count - 1)
        boundaries = {
            "train": identifiers[:train_count],
            "val": identifiers[train_count:len(identifiers) - test_count if test_count else None],
            "test": identifiers[len(identifiers) - test_count:] if test_count else [],
        }
        for split, selected in boundaries.items():
            episode_counts[split][scenario] = len(selected)
            splits[split].extend(path for identifier in selected for path in episodes[identifier])
    return splits, {"schema": next(iter(schemas)), "episode_counts": episode_counts}


def causal_smooth(position, window):
    if window < 1:
        raise ValueError("pos_smooth_window must be positive")
    if window == 1:
        return position.clone()
    return torch.stack([
        position[max(0, index + 1 - window):index + 1].mean(dim=0)
        for index in range(len(position))
    ])


class TAQLegacyGRUDataset(Dataset):
    """One last-supervised-frame sample per window, without padded input steps."""

    def __init__(self, file_path, pos_smooth_window=1):
        self.file_path = str(file_path)
        payload = load_torch_payload(file_path)
        self.metadata = payload.get("metadata", {})
        if payload.get("schema_version") != TAQ_SCHEMA_VERSION:
            raise ValueError(f"{file_path}: expected {TAQ_SCHEMA_VERSION}")
        if payload.get("label_semantics", self.metadata.get("label_semantics")) != TAQ_LABEL_SEMANTICS:
            raise ValueError(f"{file_path}: incompatible velocity label semantics")
        if self.metadata.get("angular_velocity_z_semantics") != TAQ_YAW_RATE_SEMANTICS:
            raise ValueError(f"{file_path}: incompatible heading-rate semantics")
        features = torch.as_tensor(payload["sequence_features"], dtype=torch.float32)
        valid = torch.as_tensor(payload["sequence_valid_mask"], dtype=torch.bool)
        labels = torch.as_tensor(payload["target_velocity_base"], dtype=torch.float32)
        supervised = torch.as_tensor(payload["target_valid_mask"], dtype=torch.bool)
        if features.ndim != 3 or features.shape[-1] != 24:
            raise ValueError(f"{file_path}: expected features [N,T,24]")
        if valid.shape != features.shape[:2] or supervised.shape != valid.shape or labels.shape != (*valid.shape, 3):
            raise ValueError(f"{file_path}: inconsistent window/label shapes")
        if torch.any(supervised & ~valid):
            raise ValueError(f"{file_path}: supervised padding frames")
        prefix = torch.arange(valid.shape[1])[None] < valid.sum(dim=1)[:, None]
        if not torch.equal(valid, prefix):
            raise ValueError(f"{file_path}: valid_mask must describe contiguous prefix padding")
        sequences = []
        targets = []
        self.skipped_windows = 0
        for index in range(len(features)):
            label_indices = torch.nonzero(supervised[index], as_tuple=False).flatten()
            if not len(label_indices) or int(label_indices[-1]) < 1:
                self.skipped_windows += 1
                continue
            end = int(label_indices[-1])
            current = features[index, :end + 1]
            position = causal_smooth(current[:, :3], pos_smooth_window)
            moving = (torch.linalg.vector_norm(current[1:, 6:8], dim=-1) > 0.05).float()[:, None]
            sequence = torch.cat(
                (position[1:], position[1:] - position[:-1], current[1:, 6:9], current[1:, 9:12], moving),
                dim=-1,
            )
            target = labels[index, end]
            if not torch.isfinite(sequence).all() or not torch.isfinite(target).all():
                raise ValueError(f"{file_path}: nonfinite model inputs or supervised target")
            sequences.append(sequence)
            targets.append(target)
        self.lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        self.x = pad_sequence(sequences, batch_first=True) if sequences else torch.empty(0, 0, 13)
        self.y = torch.stack(targets) if targets else torch.empty(0, 3)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index, :int(self.lengths[index])], self.y[index]


def collate_velocity_windows(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in inputs], dtype=torch.long)
    return pad_sequence(inputs, batch_first=True), torch.stack(targets), lengths


def predict_velocity_windows(model, inputs, lengths):
    output, _ = model.gru(model.normalize_input(inputs))
    indices = lengths.to(device=output.device) - 1
    return model.head(output[torch.arange(len(inputs), device=output.device), indices])
