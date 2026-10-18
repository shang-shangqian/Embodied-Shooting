"""Physical-frame, causal, streaming and checkpoint compatibility regressions."""
from pathlib import Path
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test11.speed_prediction.common.time_aware_velocity_data import (
    ALIGNED_BASELINE_VERSION, LEGACY_BASELINE_VERSION, LEGACY_INPUT_TRANSFORM_VERSION,
    TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT, TAQ_LABEL_SEMANTICS, TAQ_YAW_RATE_SEMANTICS,
    build_time_aware_features, taq_model_config_from_checkpoint, weighted_velocity_baseline,
)
from test11.speed_prediction.taq_gru.time_aware_velocity_model import TimeAwareQualityGatedGRU
from test11.speed_prediction.taq_gru.online_taq_velocity import OnlineTAQVelocityEstimator, TAQObservation
from test11.speed_prediction.taq_gru.evaluate_taq_velocity_streaming import _segmented_baseline
from test11.speed_prediction.taq_gru.train_taq_velocity_episodes import load_taq_checkpoint, main


def rotate(p, angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*p[:, 0]-s*p[:, 1], s*p[:, 0]+c*p[:, 1], p[:, 2]), -1)


def scenario(dt=.01, n=240, oscillating=True, translating=False, target_velocity=(0., 0., 0.)):
    t = torch.arange(n, dtype=torch.float64)*dt
    yaw = .008*torch.sin(4*torch.pi*t) if oscillating else .03*t
    rate = .008*4*torch.pi*torch.cos(4*torch.pi*t) if oscillating else torch.full_like(t, .03)
    base = torch.stack((.2*torch.sin(.7*t), .1*torch.sin(.5*t), t*0), -1) if translating else torch.zeros(n, 3, dtype=torch.float64)
    base_v = torch.stack((.14*torch.cos(.7*t), .05*torch.cos(.5*t), t*0), -1) if translating else torch.zeros_like(base)
    target_v = torch.tensor(target_velocity, dtype=torch.float64).expand(n, 3)
    target = torch.tensor([100., 20., 5.])+target_v*t[:, None]
    position = rotate(target-base, -yaw)
    linear = rotate(base_v, -yaw)
    angular = torch.zeros_like(position); angular[:, 2] = rate
    sources = torch.ones(n, dtype=torch.long)
    quality = torch.ones(n, 7)
    features, _ = build_time_aware_features(timestamps_s=t, position_base_m=position,
        robot_linear_velocity_base_mps=linear, robot_angular_velocity_base_rps=angular,
        source_ids=sources, quality=quality)
    return t, features, sources, rotate(target_v-base_v, -yaw).float()


def test_oscillating_base_does_not_create_stationary_target_velocity():
    t, x, _, truth = scenario()
    valid = torch.ones(1, len(t), dtype=torch.bool)
    old = weighted_velocity_baseline(x[None], valid, baseline_version=LEGACY_BASELINE_VERSION)[0]
    new = weighted_velocity_baseline(x[None], valid, baseline_version=ALIGNED_BASELINE_VERSION)[0]
    steady = t >= .5
    assert (old[steady]-truth[steady]).norm(dim=-1).mean() > 5
    assert (new[steady]-truth[steady]).norm(dim=-1).mean() < .005


@pytest.mark.parametrize('target_velocity', [(0., 0., 0.), (2., .5, .2)])
def test_translation_is_retained_in_relative_velocity(target_velocity):
    t, x, _, truth = scenario(oscillating=False, translating=True, target_velocity=target_velocity)
    model = TimeAwareQualityGatedGRU()
    new = model.compute_velocity_baseline(x[None], torch.ones(1, len(t), dtype=torch.bool))[0]
    torch.testing.assert_close(new[t >= .5], truth[t >= .5], atol=3e-5, rtol=0)


def test_online_matches_full_history_at_high_frame_rate_and_during_missing_range():
    t, x, sources, _ = scenario(dt=.0075, n=180, translating=True, target_velocity=(1., -.3, .2))
    # Rebuild with a brief missing-range interval, preserving valid time steps.
    positions = x[:, :3].clone(); positions[85:90] = float('nan')
    ranges = torch.ones(len(t), dtype=torch.bool); ranges[85:90] = False
    sources[85:90] = 3
    x, _ = build_time_aware_features(timestamps_s=t, position_base_m=positions,
        robot_linear_velocity_base_mps=x[:, 6:9], robot_angular_velocity_base_rps=x[:, 9:12],
        source_ids=sources, quality=x[:, 15:22], range_valid_mask=ranges)
    model = TimeAwareQualityGatedGRU(hidden_dim=8, dropout=0).eval()
    expected = model.compute_velocity_baseline(x[None], torch.ones(1, len(t), dtype=torch.bool))[0]
    estimator = OnlineTAQVelocityEstimator(model, max_history_steps=32)
    actual = []
    for i, time in enumerate(t.tolist()):
        estimate = estimator.update(7, TAQObservation(timestamp_s=time,
            position_base_m=positions[i].tolist(), robot_linear_velocity_base_mps=x[i, 6:9].tolist(),
            robot_angular_velocity_base_rps=x[i, 9:12].tolist(), source=int(sources[i]),
            quality=x[i, 15:22].tolist(), range_valid=bool(ranges[i])))
        actual.append(estimate.velocity_baseline_mps)
    torch.testing.assert_close(torch.stack(actual), expected, atol=2e-5, rtol=0)
    assert len(estimator._states[7].observations) > 32
    estimator.reset(7)
    assert 7 not in estimator._states


def test_causal_output_padding_and_reset_isolation():
    _, x, _, _ = scenario(n=100)
    model = TimeAwareQualityGatedGRU()
    valid = torch.ones(1, len(x), dtype=torch.bool)
    expected = model.compute_velocity_baseline(x[None], valid)
    changed = x.clone(); changed[70:, :12] += 100
    actual = model.compute_velocity_baseline(changed[None], valid)
    torch.testing.assert_close(actual[:, :70], expected[:, :70], atol=0, rtol=0)
    padded = torch.cat((x, torch.full((20, 24), float('nan'))))[None]
    padded_valid = torch.cat((valid, torch.zeros(1, 20, dtype=torch.bool)), 1)
    result = model.compute_velocity_baseline(padded, padded_valid)
    torch.testing.assert_close(result[:, :100], expected, atol=0, rtol=0)
    assert torch.equal(result[:, 100:], torch.zeros_like(result[:, 100:]))
    reset = torch.zeros(len(x), dtype=torch.bool); reset[0] = reset[60] = True
    segmented = _segmented_baseline(x, valid[0], reset, model=model, window_s=.48)
    isolated = model.compute_velocity_baseline(x[None, 60:], valid[:, 60:])[0]
    torch.testing.assert_close(segmented[60:], isolated, atol=0, rtol=0)


@pytest.mark.parametrize('version', [LEGACY_BASELINE_VERSION, ALIGNED_BASELINE_VERSION])
def test_checkpoint_loads_matching_baseline_and_legacy_defaults(tmp_path, version):
    transform = LEGACY_INPUT_TRANSFORM_VERSION if version == LEGACY_BASELINE_VERSION else 'taq_robust_fd_v1'
    model = TimeAwareQualityGatedGRU(hidden_dim=8, dropout=0, baseline_version=version,
                                     input_transform_version=transform).eval()
    config = model.config
    if version == LEGACY_BASELINE_VERSION:
        config.pop('baseline_version'); config.pop('input_transform_version')
    payload = dict(schema_version=TAQ_SCHEMA_VERSION, label_semantics=TAQ_LABEL_SEMANTICS,
        baseline_version=version, model_config=config, model_state_dict=model.state_dict(),
        input_norm=dict(mean=torch.zeros(24), std=torch.ones(24)))
    path = tmp_path/'model.pt'; torch.save(payload, path)
    loaded = load_taq_checkpoint(path)
    assert loaded.baseline_version == version
    _, x, sources, _ = scenario(n=30)
    mask = torch.ones(1, len(x), dtype=torch.bool)
    for a, b in zip(model(x[None], sources[None], mask), loaded(x[None], sources[None], mask)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if version == LEGACY_BASELINE_VERSION:
        torch.testing.assert_close(loaded.compute_velocity_baseline(x[None], mask),
            weighted_velocity_baseline(x[None], mask), atol=0, rtol=0)
    payload['model_config']['baseline_version'] = ALIGNED_BASELINE_VERSION if version == LEGACY_BASELINE_VERSION else LEGACY_BASELINE_VERSION
    with pytest.raises(ValueError, match='conflicts'):
        taq_model_config_from_checkpoint(payload)
    payload['baseline_version'] = 'unknown'
    with pytest.raises(ValueError, match='unsupported'):
        taq_model_config_from_checkpoint(payload)


def test_checkpoint_window_controls_online_support_and_legacy_keeps_frame_cap():
    t, x, sources, _ = scenario(n=60)
    model = TimeAwareQualityGatedGRU(hidden_dim=8, dropout=0, baseline_window_s=.24).eval()
    estimator = OnlineTAQVelocityEstimator(model, history_window_s=.05, max_history_steps=2)
    legacy = OnlineTAQVelocityEstimator(TimeAwareQualityGatedGRU(
        hidden_dim=8, baseline_version=LEGACY_BASELINE_VERSION).eval(), history_window_s=.05, max_history_steps=2)
    actual = []
    for i, time in enumerate(t.tolist()):
        obs = TAQObservation(timestamp_s=time, position_base_m=x[i, :3].tolist(),
            robot_linear_velocity_base_mps=x[i, 6:9].tolist(),
            robot_angular_velocity_base_rps=x[i, 9:12].tolist(), source=int(sources[i]),
            quality=x[i, 15:22].tolist())
        actual.append(estimator.update(1, obs).velocity_baseline_mps)
        legacy.update(1, obs)
    expected = model.compute_velocity_baseline(x[None], torch.ones(1, len(t), dtype=torch.bool))[0]
    torch.testing.assert_close(torch.stack(actual), expected, atol=2e-5, rtol=0)
    assert len(estimator._states[1].observations) >= 25
    assert len(legacy._states[1].observations) == 2


@pytest.mark.parametrize('version', [LEGACY_BASELINE_VERSION, ALIGNED_BASELINE_VERSION])
def test_full_episode_training_saves_matching_checkpoint(tmp_path, monkeypatch, version):
    t, x, sources, truth = scenario(n=12)
    files = {}
    for split in ('train', 'val', 'test'):
        path = tmp_path/f'{split}.taqstream'
        reset = torch.zeros(len(t), dtype=torch.bool); reset[0] = True
        torch.save(dict(schema_version=TAQ_SCHEMA_VERSION, stream_format=TAQ_STREAM_FORMAT,
            label_semantics=TAQ_LABEL_SEMANTICS, sequence_features=x, sequence_source_ids=sources,
            sequence_valid_mask=torch.ones(len(t), dtype=torch.bool), target_velocity_base=truth,
            target_valid_mask=torch.ones(len(t), dtype=torch.bool), state_reset_mask=reset, timestamps_s=t,
            metadata=dict(episode_complete=True, angular_velocity_z_semantics=TAQ_YAW_RATE_SEMANTICS,
                measurement_gt_free_after_initialization=True, episode_uid=split, scenario='sd')), path)
        files[split] = [str(path)]
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(files=files)))
    monkeypatch.setattr(sys, 'argv', ['train', '--reference_manifest', str(manifest),
        '--save_dir', str(tmp_path/'runs'), '--device', 'cpu', '--epochs', '1', '--batch_size', '1',
        '--baseline_version', version, '--baseline_window_s', '.24'])
    main()
    checkpoint = next((tmp_path/'runs').glob('*/best_model.pt'))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['baseline_version'] == version
    assert payload['model_config']['baseline_version'] == version
    assert payload['model_config']['baseline_window_s'] == .24
    assert payload['train_args']['baseline_window_s'] == .24
    assert load_taq_checkpoint(checkpoint).baseline_window_s == .24
    assert load_taq_checkpoint(checkpoint).baseline_version == version
