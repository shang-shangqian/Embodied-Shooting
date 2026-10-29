"""GT-free causal reference API for the campaign's candidate backends.

One instance per target. Reset on actual target/episode change; tracker_reset
is distinct from target reset. This implementation retains the causal prefix
for replay consistency. Cached-input benchmark is not its end-to-end budget.
"""
from pathlib import Path
import numpy as np
import torch
from optimization_components import LearnedCVFilter,ExperimentalGRU,CausalTCN,TimeAwareQualityGatedGRU,fit_baseline,propagated_inputs,rotate,WIDTHS
from train_flow_ablations import make_model
from analyze_error_sources import _segmented_baseline
from speed_prediction.time_aware_velocity_data import load_torch_payload
from speed_prediction.filter_state_output import filter_state_output

class CausalVelocityCandidate:
    def __init__(self,checkpoint=None,*,device='cpu',fixed_cv=None,baseline=None):
        self.device=torch.device(device)
        if fixed_cv is not None:
            self.cfg=dict(kalman=True,missing=True,aligned=True)
            self.scale=1;self.model=LearnedCVFilter(None,False,float(fixed_cv['q']),float(fixed_cv['r'])).to(self.device)
        elif baseline is not None:
            self.cfg=dict(baseline_only=True,aligned=True,missing=bool(baseline['missing']))
            self.scale=int(baseline['scale']);self.model=torch.nn.Identity()
        else:
            cp=load_torch_payload(checkpoint);self.cfg=cp['variant_config'];self.scale=cp['scale']
            normalizer=TimeAwareQualityGatedGRU(**cp['original_config'])
            if self.cfg.get('kalman'):self.model=LearnedCVFilter(normalizer)
            elif self.cfg.get('tcn'):self.model=CausalTCN(normalizer)
            else:self.model=ExperimentalGRU(cp['original_config'],self.cfg.get('quality',False),self.cfg.get('split_gate',False))
            self.model=self.model.to(self.device);self.model.load_state_dict(cp['state_dict'])
        self.model.eval();self.reset()
    def reset(self):
        self.features=[];self.sources=[];self.timestamps=[];self.resets=[];self.state=None
        self.last_observation_local=None;self.last_direct_time=None
        self.reference_generation=getattr(self,'reference_generation',-1)+1
    @torch.inference_mode()
    def update(self,features,source_id,timestamp_s,*,tracker_reset=False,episode_reset=False):
        if episode_reset:self.reset()
        feature=np.asarray(torch.as_tensor(features).cpu(),dtype=np.float32).reshape(24)
        if not np.isfinite(feature).all():raise ValueError('Features must be finite; represent missing via masks/source')
        if not np.isfinite(timestamp_s):raise ValueError('Timestamp must be finite')
        if self.timestamps and timestamp_s<=self.timestamps[-1]:raise ValueError('Timestamps must increase; reset on a new episode')
        elapsed=float(timestamp_s)-self.timestamps[-1] if self.timestamps else 0.
        if abs(float(feature[12])-elapsed)>max(1e-6,abs(elapsed)*1e-6):
            raise ValueError('Feature dt must match timestamp elapsed time; first tick must have zero dt')
        if self.cfg.get('kalman') and elapsed>.5+1e-9:
            raise ValueError('Legacy feature CV supports dt <= 0.5 s; use the physical packet API for longer gaps')
        if not 0<=int(source_id)<5:raise ValueError('Unknown tracker source')
        self.features.append(feature.copy());self.sources.append(int(source_id));self.timestamps.append(float(timestamp_s));self.resets.append(bool(tracker_reset) or len(self.features)==1)
        raw=np.stack(self.features);source=np.array(self.sources);times=np.array(self.timestamps);resets=np.array(self.resets)
        if self.cfg.get('missing'):x,resets=propagated_inputs(raw,source,times)
        else:x=raw
        bases=[];extras=[]
        for width in WIDTHS:
            base,extra=fit_baseline(x,times,resets,width,aligned=self.cfg.get('aligned',False),fd_weight=not self.cfg.get('aligned',False))
            if self.cfg.get('aligned') and self.cfg.get('missing'):
                for i in range(1,len(base)):
                    if x[i,23]<.5:base[i]=rotate(base[i-1]+x[i-1,6:9],np.array(-x[i,11]*(times[i]-times[i-1])))-x[i,6:9]
                extra[:,:3]=base/10
            bases.append(base);extras.append(extra)
        xd=torch.tensor(x,device=self.device);sd=torch.tensor(source,device=self.device);ed=torch.tensor(np.concatenate(extras,1),device=self.device)
        if self.cfg.get('aligned'):base=torch.tensor(bases[self.scale],device=self.device)
        else:
            base=_segmented_baseline(xd,torch.ones(len(x),dtype=torch.bool,device=self.device),torch.tensor(resets,device=self.device),self.model,.48)
            ed[:,16:19]=base/10
        if self.cfg.get('baseline_only'):
            return dict(velocity_base_mps=base[-1].cpu().numpy(),sigma_mps=None)
        if self.cfg.get('kalman'):
            mean,sigma,self.state=self.model.forward_segment(xd[None,-1:],sd[None,-1:],ed[None,-1:],torch.ones(1,1,dtype=torch.bool,device=self.device),self.state)
            mean=mean[0,-1];sigma=sigma[0,-1]
        elif self.cfg.get('tcn'):
            mean,sigma=self.model(xd[None,-29:],sd[None,-29:],ed[None,-29:],base[None,-29:]);mean=mean[0,-1];sigma=sigma[0,-1]
        else:
            if tracker_reset and not self.cfg.get('missing'):self.state=None
            self.model.current_extra=ed[-1:]
            mean,sigma,self.state,_=self.model.step(xd[-1:],sd[-1:],hidden=self.state,velocity_baseline=base[-1:]);mean=mean[0];sigma=sigma[0]
        if self.cfg.get('kalman'):
            state_mean,cov,yaw,base,robot,_,started=self.state
            turn=float(yaw[0].cpu())
            direct=bool(feature[22]>.5 and feature[23]>.5)
            if direct:
                self.last_direct_time=float(timestamp_s)
                self.last_observation_local=rotate(feature[:3].astype(float),np.array(turn))+base[0].cpu().numpy()
            if self.model.learned:
                normalized,_=self.model.normalizer._prepare_input(xd[-1:])
                noise=torch.nn.functional.softplus(self.model.noise(torch.cat((normalized,self.model.embedding(sd[-1:]),ed[-1:]),-1)))+1e-4
                q=noise[0,:3].cpu().numpy()
            else:q=np.full(3,self.model.q)
            output=filter_state_output(state_mean[0].cpu().numpy(),cov[0].cpu().numpy(),
                state_time_s=timestamp_s,heading_local_rad=turn,base_position_local_m=base[0].cpu().numpy(),
                base_velocity_base_mps=x[-1,6:9],process_intensity=q,initialized=bool(started[0].cpu()),
                reference_epoch_s=self.timestamps[0],reference_generation=self.reference_generation,
                last_direct_measurement_time_s=self.last_direct_time,last_observation_local_m=self.last_observation_local,
                direct_update=direct)
            # Preserve legacy float32 speed and clipped training sigma fields.
            output.update(velocity_base_mps=mean.cpu().numpy(),sigma_mps=torch.exp(sigma).cpu().numpy())
            return output
        return dict(velocity_base_mps=mean.cpu().numpy(),sigma_mps=torch.exp(sigma).cpu().numpy())

    @classmethod
    def from_selection(cls,runs,*,device='cpu'):
        import json
        runs=Path(runs);choice=json.loads((runs/'evaluation/frozen_selection.json').read_text())
        family=choice['recommended_family']
        if family=='fixed_cv':return cls(device=device,fixed_cv=choice['fixed_cv_validation_choice'])
        if family=='aligned_baseline':return cls(device=device,baseline=choice['aligned_baseline_validation_choice'])
        return cls(runs/choice['run']/'best_model.pt',device=device)
