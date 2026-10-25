"""Migration must not silently read the retained system-disk project copy."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speed_prediction.common import project_paths
from speed_prediction.common.time_aware_velocity_data import load_torch_payload


@pytest.fixture
def copies(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    new.mkdir()
    monkeypatch.setattr(project_paths, 'LEGACY_PROJECT_ROOT', old)
    monkeypatch.setattr(project_paths, 'PROJECT_ROOT', new)
    return old, new


def test_payload_uses_new_copy_when_original_still_exists(copies):
    old, new = copies
    torch.save({'copy': 'old'}, old/'model.pt')
    torch.save({'copy': 'new'}, new/'model.pt')
    assert load_torch_payload(old/'model.pt') == {'copy': 'new'}


def test_missing_migrated_payload_does_not_fall_back(copies):
    old, _ = copies
    torch.save({'copy': 'old'}, old/'model.pt')
    with pytest.raises(FileNotFoundError):
        load_torch_payload(old/'model.pt')


def test_manifest_normalizes_mixed_paths_without_rewriting_history(copies):
    old, new = copies
    original = {'files': {'train': [str(old/'episode.taqstream')],
                           'val': [str(new/'val.taqstream')]}, 'seed': 42}
    source = new/'manifest.json'
    source.write_text(json.dumps(original))
    before = source.read_bytes()
    loaded = project_paths.load_project_json(old/'manifest.json')
    assert loaded['files']['train'] == [str(new/'episode.taqstream')]
    assert loaded['files']['val'] == [str(new/'val.taqstream')]
    assert loaded['seed'] == 42
    assert source.read_bytes() == before


def test_external_storage_and_simulator_paths_stay_external(copies):
    for value in ('/mnt/A/hust_jby/isaaclab_storage/test11/data/episode.taqstream',
                  '/home/jby/Documents/isaacsim-5.1.0', 'relative/model.pt'):
        assert project_paths.resolve_project_path(value) == Path(value)


def test_prefix_sibling_is_not_part_of_old_project(copies):
    old, _ = copies
    sibling = Path(str(old) + '.before_migration')/'model.pt'
    assert project_paths.resolve_project_path(sibling) == sibling
