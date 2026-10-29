"""Numerical and causality checks that protect the scientific comparisons."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from optimization_components import fit_baseline,rotate,propagated_inputs,ExperimentalGRU,CausalTCN
from train_flow_ablations import gru_segment
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU

def motion(n=60):
    t=np.arange(n)*.02;x=np.zeros((n,24),np.float32);x[:,12]=.02;x[0,12]=0;x[:,15]=x[:,22]=x[:,23]=1
    x[:,:3]=np.array([90,2,1])+t[:,None]*np.array([2,-1,.5]);x[1:,3:6]=[2,-1,.5]
    r=np.zeros(n,bool);r[0]=True
    return x,t,r

def test_aligned_fit_matches_constant_velocity_in_rotating_translating_frame():
    x,t,r=motion();yaw=.3*t;base=t[:,None]*[.4,-.2,.1]
    target=np.array([90,2,1])+t[:,None]*[2,-1,.5]
    x[:,:3]=rotate(target-base,-yaw);x[:,11]=.3;x[:,6:9]=rotate(np.tile([.4,-.2,.1],(len(t),1)),-yaw)
    b,_=fit_baseline(x,t,r,.48,aligned=True,fd_weight=False)
    truth=rotate(np.tile([1.6,-.8,.4],(len(t),1)),-yaw)
    np.testing.assert_allclose(b[5:],truth[5:],atol=3e-3)

def test_baseline_prefix_invariance_and_reset_boundaries():
    x,t,r=motion();b,_=fit_baseline(x,t,r,aligned=True)
    changed=x.copy();changed[30:,:3]+=1000
    c,_=fit_baseline(changed,t,r,aligned=True)
    np.testing.assert_array_equal(b[:30],c[:30])
    r[30]=True;c,_=fit_baseline(changed,t,r,aligned=True)
    np.testing.assert_array_equal(c[30],np.zeros(3))

def test_missing_propagation_is_causal_and_keeps_observation_flags():
    x,t,r=motion();x[30:40,:6]=0;x[30:40,22:24]=0
    source=np.ones(len(t),int);source[30:40]=4
    xp,rp=propagated_inputs(x,source,t)
    np.testing.assert_array_equal(xp[:,22:24],x[:,22:24]);assert rp.sum()==1
    assert np.linalg.norm(xp[35,:3])>80
    xf=x.copy();xf[40:,:3]*=100
    truncated,_=propagated_inputs(xf,source,t)
    np.testing.assert_array_equal(xp[:40],truncated[:40])

def test_gru_segment_and_step_agree_including_reset_and_padding():
    torch.manual_seed(1);model=ExperimentalGRU(TimeAwareQualityGatedGRU(dropout=0).config).eval()
    x=torch.randn(2,15,24);x[:,:,12]=.02
    batch=dict(x=x,source=torch.ones(2,15,dtype=torch.long),valid=torch.ones(2,15,dtype=torch.bool),reset=torch.zeros(2,15,dtype=torch.bool),base=torch.randn(2,15,3),extra=torch.zeros(2,15,24))
    batch['reset'][:,0]=True;batch['reset'][0,8]=True;batch['valid'][1,12:]=False
    with torch.inference_mode():
        pred,sigma,h=gru_segment(model,batch,0,15);state=None;values=[]
        for i in range(15):
            if state is not None:state=torch.where(batch['reset'][:,i,None],0,state)
            p,_,state,_=model.step(x[:,i],batch['source'][:,i],hidden=state,valid_mask=batch['valid'][:,i],velocity_baseline=batch['base'][:,i]);values.append(p)
        torch.testing.assert_close(pred,torch.stack(values,1),atol=1e-6,rtol=1e-6)
        a,_,ha=gru_segment(model,batch,0,7);b,_,hb=gru_segment(model,batch,7,15,ha.detach())
        torch.testing.assert_close(pred,torch.cat((a,b),1),atol=1e-6,rtol=1e-6)

def test_tcn_is_causal_and_matches_29_frame_history():
    torch.manual_seed(2);normalizer=TimeAwareQualityGatedGRU(dropout=0);model=CausalTCN(normalizer).eval()
    x=torch.randn(2,50,24);s=torch.ones(2,50,dtype=torch.long);e=torch.randn(2,50,24);b=torch.randn(2,50,3)
    with torch.inference_mode():
        full,_=model(x,s,e,b)
        changed=x.clone();changed[:,31:]=100
        future,_=model(changed,s,e,b)
        torch.testing.assert_close(full[:,:31],future[:,:31])
        last,_=model(x[:,-29:],s[:,-29:],e[:,-29:],b[:,-29:])
        torch.testing.assert_close(full[:,-1],last[:,-1],atol=1e-6,rtol=1e-6)

def test_physical_filter_stream_segments_and_missing_are_causal():
    from optimization_components import LearnedCVFilter
    model=LearnedCVFilter(TimeAwareQualityGatedGRU(dropout=0),learned=True)
    x,t,_=motion(50);x[20:30,22:24]=0;x=torch.tensor(x)[None]
    sources=torch.ones(1,50,dtype=torch.long);extra=torch.zeros(1,50,24);valid=torch.ones(1,50,dtype=torch.bool)
    full,sigma,state=model.forward_segment(x,sources,extra,valid)
    left,_,state_left=model.forward_segment(x[:,:23],sources[:,:23],extra[:,:23],valid[:,:23])
    right,_,_=model.forward_segment(x[:,23:],sources[:,23:],extra[:,23:],valid[:,23:],tuple(v.detach() for v in state_left))
    torch.testing.assert_close(full,torch.cat((left,right),1),atol=1e-6,rtol=1e-6)
    changed=x.clone();changed[:,30:,:3]+=100
    future,_,_=model.forward_segment(changed,sources,extra,valid)
    torch.testing.assert_close(full[:,:30],future[:,:30])
    (full.square().mean()+sigma.square().mean()).backward()
    assert torch.isfinite(model.noise[-1].weight.grad).all()
    assert (torch.linalg.eigvalsh(state[1].detach())>0).all()

def test_separate_position_gate_does_not_suppress_position_for_large_fd():
    torch.manual_seed(6);model=ExperimentalGRU(TimeAwareQualityGatedGRU(dropout=0).config,split_gate=True).eval()
    x=torch.randn(1,24);x[:,12]=.02;x[:,22:24]=1;source=torch.ones(1,dtype=torch.long)
    large=x.clone();large[:,3:6]=3000
    with torch.inference_mode():
        a,_,_=model._encode_step(x,source);b,_,_=model._encode_step(large,source)
    torch.testing.assert_close(a[:,:32],b[:,:32])

def test_physical_memory_advances_through_prediction_before_complete_loss():
    x,t,r=motion(50);source=np.ones(len(t),int)
    x[30:40,22]=0;source[30:40]=3
    x[40:45,:6]=0;x[40:45,22:24]=0;source[40:45]=4
    propagated,_=propagated_inputs(x,source,t)
    truth=np.array([90,2,1])+t[:,None]*np.array([2,-1,.5])
    np.testing.assert_allclose(propagated[40:45,:3],truth[40:45],atol=1e-4)
    # Real vision-module prediction positions remain the actual inputs.
    np.testing.assert_array_equal(propagated[30:40,:3],x[30:40,:3])

def test_gt_free_online_filter_matches_batched_causal_stream():
    from online_experimental_velocity import CausalVelocityCandidate
    from optimization_components import LearnedCVFilter
    x,t,r=motion(45);source=np.ones(len(t),int);x[20:30,22]=0;source[20:30]=3;x[30:35,:6]=0;x[30:35,22:24]=0;source[30:35]=4
    runner=CausalVelocityCandidate(device='cpu',fixed_cv={'q':1,'r':.01})
    stream=[]
    for i in range(len(t)):stream.append(runner.update(x[i],int(source[i]),float(t[i]),tracker_reset=bool(r[i]))['velocity_base_mps'])
    xp,_=propagated_inputs(x,source,t)
    model=LearnedCVFilter(None,False,1,.01).eval()
    with torch.inference_mode():
        batch,_,_=model.forward_segment(torch.tensor(xp)[None],torch.tensor(source)[None],torch.zeros(1,len(t),24),torch.ones(1,len(t),dtype=torch.bool))
    np.testing.assert_allclose(np.array(stream),batch[0].numpy(),atol=1e-6)
    assert len(runner.features)==len(t)
    runner.reset();assert len(runner.features)==0 and runner.state is None
