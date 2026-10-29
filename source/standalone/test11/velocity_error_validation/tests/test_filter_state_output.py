"""Physical output and forecasting checks independent of learned performance."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from speed_prediction.filter_state_output import filter_state_output, forecast_local, predict_cv_state
from n0_time_replay import CausalSensorTimeEstimator, NoiseBackend, delivery_schedule
from test_sensor_time_contract import packets


def output(**kw):
    defaults=dict(state_time_s=2.,heading_local_rad=.7,base_position_local_m=[1,2,3],
                  base_velocity_base_mps=[.4,-.2,.1],process_intensity=[.1,.2,.3],
                  initialized=True,reference_epoch_s=0.)
    defaults.update(kw)
    return filter_state_output(np.arange(6,dtype=float),np.eye(6)*25,**defaults)


def test_cross_covariance_is_included_in_forecast():
    p=np.eye(6);p[:3,3:]=p[3:,:3]=np.eye(3)*.4
    _,c=predict_cv_state(np.arange(6),p,[.2]*3,.2)
    np.testing.assert_allclose(c[:3,:3],np.eye(3)*(1+2*.2*.4+.2**2+.2*.2**3/3),atol=1e-14)
    np.testing.assert_allclose(c[:3,3:],np.eye(3)*(.4+.2+.2*.2**2/2),atol=1e-14)


def test_body_relative_state_and_joint_covariance_rotate_together():
    rng=np.random.default_rng(3);a=rng.normal(size=(6,6));p=a@a.T
    o=filter_state_output(np.arange(6),p,state_time_s=1,heading_local_rad=np.pi/2,
        base_position_local_m=[1,2,3],base_velocity_base_mps=[.4,-.2,.1],
        process_intensity=[0,0,0],initialized=True,reference_epoch_s=0)
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_allclose(o['position_base_m'],rot.T@(np.arange(3)-[1,2,3]),atol=1e-14)
    np.testing.assert_allclose(o['velocity_base_mps'],rot.T@np.arange(3,6)-[.4,-.2,.1],atol=1e-14)
    np.testing.assert_allclose(o['covariance_base'][:3,3:],rot.T@p[:3,3:]@rot,atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(o['covariance_base']),np.linalg.eigvalsh(p),atol=1e-13)


def test_output_gap_and_forecast_use_full_elapsed_time():
    o=output(output_time_s=2.8,last_direct_measurement_time_s=1.5,last_observation_local_m=[4,5,6])
    assert o['posterior_time_s']==2 and o['state_time_s']==o['output_time_s']==2.8
    assert o['measurement_age_s']==pytest.approx(1.3)
    np.testing.assert_allclose(o['position_local_m'],np.arange(3)+.8*np.arange(3,6))
    f=forecast_local(o,.7)
    expected,c=predict_cv_state(np.arange(6),np.eye(6)*25,[.1,.2,.3],1.5)
    np.testing.assert_allclose(f['position_local_m'],expected[:3],atol=1e-14)
    np.testing.assert_allclose(f['covariance_local'],c,atol=1e-13)
    assert f['prediction_time_s']==3.5


@pytest.mark.parametrize('gap',[-.01,np.nan,np.inf])
def test_invalid_future_interval_is_rejected(gap):
    with pytest.raises(ValueError):forecast_local(output(),gap)


def test_uncertainty_keeps_unclipped_physical_covariance():
    o=filter_state_output(np.zeros(6),np.eye(6)*10000,state_time_s=0,heading_local_rad=0,
       base_position_local_m=[0]*3,base_velocity_base_mps=[0]*3,process_intensity=[0]*3,
       initialized=False,reference_epoch_s=0)
    np.testing.assert_array_equal(o['sigma_base_mps'],[100]*3)
    assert o['status']=='uninitialized'
    assert o['last_direct_measurement_time_s'] is None


def test_snapshot_changes_cannot_mutate_filter_or_other_outputs():
    raw=packets(8);runner=CausalSensorTimeEstimator(NoiseBackend());q,_,_=delivery_schedule(raw,'sync',1)
    first=runner.advance(0,0,q.arrived(0));before=copy.deepcopy(first)
    for p in raw[1:]:runner.advance(p['sample_id'],p['capture_time_s'],q.arrived(p['capture_time_s']))
    np.testing.assert_array_equal(first['state_local'],before['state_local'])
    first['covariance_local'][:]=0;first['position_local_m'][:]=999
    assert np.linalg.eigvalsh(runner.engine.states[0][1]).min()>0
    assert runner.engine.states[0][0][0]<200


def test_delayed_observation_time_is_distinct_from_state_time():
    raw=packets(12,moving=True);q,_,_=delivery_schedule(raw,'image_delay',1)
    runner=CausalSensorTimeEstimator(NoiseBackend())
    for p in raw:o=runner.advance(p['sample_id'],p['capture_time_s'],q.arrived(p['capture_time_s']))
    assert o['state_time_s']==o['output_time_s']==raw[-1]['capture_time_s']
    assert o['last_direct_measurement_time_s']<o['state_time_s']
    assert o['measurement_age_s']>=.04
    np.testing.assert_allclose(o['state_local'],runner.engine.states[-1][0])
    assert o['covariance_position_velocity_local'].shape==(3,3)


def test_future_sensor_values_cannot_change_published_position_or_covariance():
    raw=packets(25,moving=True);changed=copy.deepcopy(raw)
    for p in changed[15:]:p['range_m']*=2;p['linear_velocity_base_mps']=[10,20,30]
    values=[]
    for stream in (raw,changed):
        q,_,_=delivery_schedule(stream,'combined',4);runner=CausalSensorTimeEstimator(NoiseBackend());out=[]
        for p in stream:out.append(runner.advance(p['sample_id'],p['capture_time_s'],q.arrived(p['capture_time_s'])))
        values.append(out)
    for a,b in zip(values[0][:15],values[1][:15]):
        for k in ('state_local','covariance_local','covariance_base'):np.testing.assert_array_equal(a[k],b[k])


def test_real_identity_reset_creates_new_reference_generation():
    runner=CausalSensorTimeEstimator(NoiseBackend());raw=packets(5)
    q,_,_=delivery_schedule(raw,'sync',1)
    for p in raw:a=runner.advance(p['sample_id'],p['capture_time_s'],q.arrived(p['capture_time_s']))
    runner.reset();b=runner.advance(0,0,[])
    assert b['reference_generation']==a['reference_generation']+1
    assert not b['initialized'] and b['last_observation_local_m'] is None


def test_legacy_candidate_exposes_joint_state_without_changing_speed():
    from online_experimental_velocity import CausalVelocityCandidate
    x=np.zeros(24,np.float32);x[:3]=[90,2,1];x[22:24]=1;x[15]=1
    runner=CausalVelocityCandidate(device='cpu',fixed_cv={'q':.1,'r':.1})
    o=runner.update(x,1,5)
    assert o['state_time_s']==o['output_time_s']==o['reference_epoch_s']==5
    np.testing.assert_array_equal(o['velocity_base_mps'],np.zeros(3))
    np.testing.assert_allclose(o['covariance_local'],runner.state[1][0].numpy(),atol=1e-8)
    x[12]=.02
    o2=runner.update(x,1,5.02,tracker_reset=True)
    assert o2['reference_generation']==o['reference_generation']


def test_legacy_feature_timestamp_mismatch_is_rejected_before_state_changes():
    from online_experimental_velocity import CausalVelocityCandidate
    x=np.zeros(24,np.float32);x[:3]=[90,2,1];x[22:24]=1;x[15]=1
    runner=CausalVelocityCandidate(device='cpu',fixed_cv={'q':.1,'r':.1})
    runner.update(x,1,0)
    x[12]=.04
    with pytest.raises(ValueError,match='dt must match'):runner.update(x,1,.02)
    assert len(runner.timestamps)==1
    x[12]=.02
    assert runner.update(x,1,.02)['state_time_s']==.02


def test_legacy_feature_api_cannot_silently_misdate_a_clipped_long_gap():
    from online_experimental_velocity import CausalVelocityCandidate
    x=np.zeros(24,np.float32);x[:3]=[90,2,1];x[22:24]=1;x[15]=1
    runner=CausalVelocityCandidate(device='cpu',fixed_cv={'q':.1,'r':.1})
    runner.update(x,1,0);x[12]=.7
    with pytest.raises(ValueError,match='physical packet API'):runner.update(x,1,.7)
    assert len(runner.timestamps)==1


def test_uninitialized_forecast_requires_explicit_prior_request():
    o=output(initialized=False)
    assert not o['position_available'] and not o['forecast_valid']
    with pytest.raises(ValueError,match='uninitialized'):forecast_local(o,.1)
    prior=forecast_local(o,.1,allow_uninitialized=True)
    assert not prior['valid'] and not prior['initialized']
