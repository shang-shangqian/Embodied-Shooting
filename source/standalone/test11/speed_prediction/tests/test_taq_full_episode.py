"""Full-episode TAQ equivalence, gradient continuity and input isolation."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speed_prediction.common.time_aware_velocity_data import (
    TAQ_SCHEMA_VERSION, TAQ_STREAM_FORMAT, TAQ_LABEL_SEMANTICS,
    TAQ_YAW_RATE_SEMANTICS, build_time_aware_features,
)
from speed_prediction.taq_gru.train_taq_velocity_episodes import (
    EpisodeDataset, collate_episodes, forward_episode, load_reference_splits,
)
from speed_prediction.taq_gru.time_aware_velocity_model import TimeAwareQualityGatedGRU


def episode_file(tmp_path):
    t = torch.arange(8, dtype=torch.float64)*.02
    p = torch.stack((10+2*t, 1+t, t*0), -1).float()
    sources = torch.ones(8, dtype=torch.long)
    x, _ = build_time_aware_features(timestamps_s=t, position_base_m=p,
        robot_linear_velocity_base_mps=torch.zeros_like(p),
        robot_angular_velocity_base_rps=torch.zeros_like(p), source_ids=sources,
        quality=torch.ones(8,7))
    reset = torch.zeros(8,dtype=torch.bool)
    reset[0] = True
    payload = dict(schema_version=TAQ_SCHEMA_VERSION, stream_format=TAQ_STREAM_FORMAT,
        label_semantics=TAQ_LABEL_SEMANTICS, sequence_features=x,
        sequence_source_ids=sources, sequence_valid_mask=torch.ones(8,dtype=torch.bool),
        target_velocity_base=torch.ones(8,3), target_valid_mask=torch.ones(8,dtype=torch.bool),
        state_reset_mask=reset, timestamps_s=t, metadata=dict(episode_complete=True,
        angular_velocity_z_semantics=TAQ_YAW_RATE_SEMANTICS,
        measurement_gt_free_after_initialization=True, episode_uid='test', scenario='sd'))
    path = tmp_path/'episode.taqstream'
    torch.save(payload,path)
    return path,payload


def test_batched_encoding_matches_native_outputs_and_parameter_gradients(tmp_path):
    torch.manual_seed(3)
    model = TimeAwareQualityGatedGRU(dropout=0).eval()
    path,_ = episode_file(tmp_path)
    batch = collate_episodes(EpisodeDataset([path],model).episodes)
    expected = model(batch['features'],batch['source_ids'],batch['valid_mask'],
                     velocity_baseline=batch['baseline'])
    actual = forward_episode(model,batch)
    for a,b in zip(expected,actual):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    sum(v.square().sum() for v in expected).backward()
    expected_grad = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    sum(v.square().sum() for v in forward_episode(model,batch)).backward()
    for name,p in model.named_parameters():
        torch.testing.assert_close(p.grad,expected_grad[name],atol=2e-5,rtol=2e-4)


def test_reset_and_padding_match_step_interface(tmp_path):
    model = TimeAwareQualityGatedGRU(dropout=0).eval()
    path,payload = episode_file(tmp_path)
    payload['state_reset_mask'][4] = True
    torch.save(payload,path)
    batch = collate_episodes(EpisodeDataset([path],model).episodes)
    batch['valid_mask'][:,-1] = False
    actual = forward_episode(model,batch)
    hidden=None
    means,sigmas=[],[]
    for i in range(8):
        if batch['reset_mask'][0,i]: hidden=None
        mean,sigma,hidden,_=model.step(batch['features'][:,i],batch['source_ids'][:,i],
            hidden=hidden,valid_mask=batch['valid_mask'][:,i],velocity_baseline=batch['baseline'][:,i])
        means.append(mean);sigmas.append(sigma)
    for a,b in zip(actual,(torch.stack(means,1),torch.stack(sigmas,1))):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    assert torch.equal(batch['baseline'][0,4],torch.zeros(3))


def test_complete_bptt_keeps_gradient_to_early_observations(tmp_path):
    model=TimeAwareQualityGatedGRU(dropout=0).eval()
    path,_=episode_file(tmp_path)
    batch=collate_episodes(EpisodeDataset([path],model).episodes)
    batch['features'].requires_grad_()
    mean,_=forward_episode(model,batch)
    mean[:,-1].sum().backward()
    assert batch['features'].grad[0,0].abs().sum()>0


def test_dataset_rejects_oracle_and_isolates_targets(tmp_path):
    model=TimeAwareQualityGatedGRU(dropout=0).eval()
    path,payload=episode_file(tmp_path)
    a=EpisodeDataset([path],model)[0]
    payload['target_velocity_base']+=1000
    torch.save(payload,path)
    b=EpisodeDataset([path],model)[0]
    for key in ('features','source_ids','valid_mask','baseline','reset_mask'):
        assert torch.equal(a[key],b[key])
    payload['metadata']['validation_gt_range']=True
    torch.save(payload,path)
    with pytest.raises(ValueError,match='GT/oracle'):
        EpisodeDataset([path],model)


def test_reference_split_rejects_shared_episode(tmp_path):
    import json
    path,_=episode_file(tmp_path)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'files':{s:[str(path)] for s in ('train','val','test')}}))
    with pytest.raises(ValueError,match='Duplicate or cross-split'):
        load_reference_splits(manifest)
