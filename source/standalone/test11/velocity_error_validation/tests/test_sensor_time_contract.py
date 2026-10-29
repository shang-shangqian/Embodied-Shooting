"""Timing errors, causality, asynchronous pairing, and CV replay regressions."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from n0_time_replay import (NoiseBackend, delivery_schedule, feature_history,
                            replay, cv_step, initial_state)
from optimization_components import LearnedCVFilter
from speed_prediction.sensor_time_contract import (
    Delivery, DeliveryQueue, MatchedMeasurements, position_from_pair, yaw_rotation, capture_packet,
)
from speed_prediction.time_aware_velocity_data import build_time_aware_features


def packets(n=45, moving=False):
    result = []
    for i in range(n):
        t = i*.02; yaw = .3*t if moving else 0.
        robot = np.array([.4*t, -.2*t, 0]) if moving else np.zeros(3)
        relative = yaw_rotation(-yaw)@(np.array([100, 2, 1])+[2*t, -t, .2*t]-robot)
        # Identity optical->body calibration for this analytical fixture.
        distance = np.linalg.norm(relative)
        direction_yaw = np.arctan2(relative[0], relative[2])
        direction_pitch = np.arcsin(-relative[1]/distance)
        result.append(dict(sample_id=i, capture_time_s=t, direction_yaw_pitch=[direction_yaw,direction_pitch],
                           camera_rotation_base=np.eye(3).tolist(), camera_origin_base=[0,0,0],
                           range_m=distance,range_valid=True,source_id=1,quality=[1]*7,
                           linear_velocity_base_mps=(yaw_rotation(-yaw)@np.array([.4,-.2,0])).tolist() if moving else [0,0,0],
                           angular_velocity_base_rps=[0,0,.3 if moving else 0]))
    return result


def test_transport_does_not_mutate_capture_clock():
    raw=packets(); original=copy.deepcopy(raw)
    _,clock,rows=delivery_schedule(raw,'combined',123)
    assert raw==original
    assert all(r['arrival_time_s']>=r['sample_time_s'] for r in rows)
    assert any(r['arrival_time_s']>r['sample_time_s'] for r in rows)
    assert not np.array_equal(clock,np.arange(len(raw))*.02)


def test_queue_releases_only_arrived_packets_and_handles_reordering():
    raw=packets(); q=DeliveryQueue()
    q.push(Delivery('direction',0,0,.10,raw[0]));q.push(Delivery('direction',1,.02,.04,raw[1]))
    assert q.arrived(.03)==[]
    assert [p.sample_id for p in q.arrived(.04)]==[1]
    assert [p.sample_id for p in q.arrived(.10)]==[0]
    with pytest.raises(ValueError,match='Duplicate'):q.push(Delivery('direction',0,0,.11,raw[0]))
    with pytest.raises(ValueError,match='precedes'):q.push(Delivery('range',1,.02,0,raw[1]))


def test_async_pairing_never_combines_different_exposures():
    raw=packets(); m=MatchedMeasurements()
    assert m.receive(Delivery('range',1,.02,.02,raw[1])) is None
    assert m.receive(Delivery('direction',0,0,.04,raw[0])) is None
    pair=m.receive(Delivery('range',0,0,.04,raw[0]))
    np.testing.assert_allclose(pair['position_base_m'],[100,2,1],atol=1e-10)
    assert len(m.pending)==1
    with pytest.raises(ValueError,match='consumed'):m.receive(Delivery('direction',0,0,.05,raw[0]))
    with pytest.raises(ValueError,match='identity'):position_from_pair(raw[0],raw[1])
    altered=dict(raw[0],capture_time_s=.001)
    with pytest.raises(ValueError,match='time'):position_from_pair(raw[0],altered)


def test_pending_unpaired_components_expire():
    raw=packets();m=MatchedMeasurements()
    m.receive(Delivery('range',0,0,0,raw[0]));assert m.expire(.21)==1
    assert not m.pending


def test_feature_constructor_matches_existing_24_feature_contract():
    n=20;t=np.arange(n)*.02;raw=packets(n)
    pos=np.array([position_from_pair(p,p) for p in raw],np.float32)
    lin=np.zeros((n,3));ang=lin.copy();src=np.ones(n,int);quality=np.ones((n,7))
    # All-valid input keeps physical missing propagation inactive.
    x,_=feature_history(pos,lin,ang,src,quality,t)
    original,_=build_time_aware_features(timestamps_s=t,position_base_m=pos,
        robot_linear_velocity_base_mps=lin,robot_angular_velocity_base_rps=ang,
        source_ids=src,quality=quality,range_valid_mask=np.ones(n,bool))
    np.testing.assert_allclose(x,original.numpy(),atol=2e-6)


def test_sync_replay_equals_original_filter_equations():
    raw=packets(25,moving=True);n=len(raw)
    pos=np.array([position_from_pair(p,p) for p in raw],np.float32)
    lin=np.array([p['linear_velocity_base_mps'] for p in raw]);ang=np.array([p['angular_velocity_base_rps'] for p in raw])
    src=np.ones(n,int);quality=np.ones((n,7));t=np.arange(n)*.02
    x,extra=feature_history(pos,lin,ang,src,quality,t)
    model=LearnedCVFilter(None,False,.1,.1)
    with torch.inference_mode():
        pred,_,_=model.forward_segment(torch.tensor(x)[None],torch.tensor(src)[None],
            torch.tensor(extra)[None],torch.ones(1,n,dtype=torch.bool))
    corrected=replay(raw,'sync',1,NoiseBackend())
    np.testing.assert_allclose(corrected['prediction'],pred[0].numpy(),atol=5e-4)
    legacy=replay(raw,'sync',1,NoiseBackend(),corrected=False)
    np.testing.assert_array_equal(corrected['prediction'],legacy['prediction'])


def test_delayed_replay_uses_each_measurement_once_and_improves_cv_motion():
    raw=packets(55,moving=True)
    correct=replay(raw,'image_delay',2,NoiseBackend())
    legacy=replay(raw,'image_delay',2,NoiseBackend(),corrected=False)
    truth=np.array([yaw_rotation(-.3*p['capture_time_s'])@np.array([1.6,-.8,.2]) for p in raw])
    assert np.mean(np.linalg.norm(correct['prediction'][20:]-truth[20:],axis=1)) < np.mean(np.linalg.norm(legacy['prediction'][20:]-truth[20:],axis=1))
    assert correct['stats']['paired_measurements']==correct['stats']['unique_pairs']
    assert correct['stats']['max_replay_age_s']<=.20+1e-9
    assert np.linalg.eigvalsh(correct['covariance']).min()>0


@pytest.mark.parametrize('profile',['image_delay','proprio_delay','combined'])
def test_future_sensor_values_cannot_change_past_outputs(profile):
    raw=packets(30); changed=copy.deepcopy(raw)
    for p in changed[18:]:p['range_m']*=3;p['linear_velocity_base_mps']=[20,20,20]
    a=replay(raw,profile,8,NoiseBackend());b=replay(changed,profile,8,NoiseBackend())
    np.testing.assert_array_equal(a['prediction'][:18],b['prediction'][:18])


def test_real_long_dt_is_fully_propagated():
    state=list(initial_state());state[0][3:]=[2,0,0]
    x=np.zeros(24);x[12]=1.2
    result,_=cv_step(tuple(state),x,np.array([.1]*6))
    np.testing.assert_allclose(result[0][:3],[2.4,0,0])
    assert result[1][0,0]>25+25*1.2**2


def test_capture_export_has_no_world_pose_and_preserves_camera_offset():
    p=capture_packet(sample_id=1,sample_time_s=.02,yaw_pitch=[0,0],range_m=10,
        range_valid=True,source_id=1,quality=[1]*7,linear=[0,0,0],angular=[0,0,0],
        camera_position_world=[7,8,10],camera_quaternion_world=[1,0,0,0],
        base_position_world=[7,8,9],base_quaternion_world=[1,0,0,0],tracker_reset=False)
    assert not any('world' in key or 'target' in key for key in p)
    np.testing.assert_allclose(position_from_pair(p,p),[0,0,11])
    assert p['direction_sample_time_s']==p['range_sample_time_s']==p['proprio_sample_time_s']


def test_expired_samples_cannot_rewrite_old_state():
    from n0_time_replay import TimeContractReplay
    raw=packets(20);engine=TimeContractReplay(raw,NoiseBackend(),lag_s=.1)
    _,clock,_=delivery_schedule(raw,'sync',4)
    for i in range(18):engine.update(i,[],clock)
    before=[s[0].copy() for s in engine.states]
    event=Delivery('direction',0,0,.36,raw[0])
    engine.update(18,[event],clock)
    assert engine.expired==1
    for a,b in zip(before,engine.states):np.testing.assert_array_equal(a,b[0])


def test_cached_features_equal_full_prefix_with_late_changes_and_missing():
    from n0_time_replay import FeatureCache
    raw=packets(38,moving=True);cache=FeatureCache()
    pos=np.array([position_from_pair(p,p) for p in raw],np.float32)
    linear=np.array([p['linear_velocity_base_mps'] for p in raw]);angular=np.array([p['angular_velocity_base_rps'] for p in raw])
    src=np.ones(len(raw),int);q=np.ones((len(raw),7));times=np.arange(len(raw))*.02
    pos[12:19]=np.nan;src[12:19]=4
    for end in range(1,len(raw)+1):
        start=end-1
        if end==23:
            pos[17]=position_from_pair(raw[17],raw[17]);src[17]=1;start=17
        tail,extra=cache.build(pos[:end],linear[:end],angular[:end],src[:end],q[:end],times[:end],start)
        expected,expected_extra=feature_history(pos[:end],linear[:end],angular[:end],src[:end],q[:end],times[:end])
        np.testing.assert_allclose(tail,expected[start:],atol=2e-6)
        np.testing.assert_allclose(extra,expected_extra[start:],atol=2e-6)


def test_actual_sampling_gate_changes_measurement_intervals_not_physics_time():
    from speed_prediction.sensor_time_contract import PhysicalSamplingGate
    gate=PhysicalSamplingGate(.2,9)
    physical=np.arange(100)*.02;sampled=np.array([gate.take(i) for i in range(100)])
    assert sampled[0] and 0<sampled.sum()<100
    assert np.diff(physical[sampled]).max()>.02
    np.testing.assert_allclose(np.diff(physical),.02)


def test_laser_return_without_a_target_direction_is_not_a_target_position():
    p=packets(1)[0]
    missing_direction=dict(p,source_id=4)
    assert position_from_pair(missing_direction,p) is None


@pytest.mark.parametrize('profile',['sync','image_delay','combined'])
def test_live_packet_api_requires_no_future_timeline_and_matches_replay(profile):
    from n0_time_replay import CausalSensorTimeEstimator
    raw=packets(40,moving=True);queue,_,_=delivery_schedule(raw,profile,9)
    online=CausalSensorTimeEstimator(NoiseBackend());pred=[]
    for p in raw:
        result=online.advance(p['sample_id'],p['capture_time_s'],queue.arrived(p['capture_time_s']))
        pred.append(result['velocity_base_mps'])
    expected=replay(raw,profile,9,NoiseBackend())
    np.testing.assert_array_equal(pred,expected['prediction'])
    online.reset();assert len(online.engine.times)==0


def test_delivery_metadata_cannot_relabel_a_future_sample_as_past():
    raw=packets(3);queue=DeliveryQueue()
    with pytest.raises(ValueError,match='metadata disagree'):
        queue.push(Delivery('direction',0,0,.05,raw[2]))
    from n0_time_replay import CausalSensorTimeEstimator
    online=CausalSensorTimeEstimator(NoiseBackend())
    with pytest.raises(ValueError,match='mismatch'):
        online.advance(0,0,[Delivery('direction',0,0,0,raw[2])])
