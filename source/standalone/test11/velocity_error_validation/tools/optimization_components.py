"""Causal experimental components. Audit GT never enters deployable features."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_error_sources import _stream_arrays, _load_model, _segmented_baseline
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU
from speed_prediction.time_aware_velocity_data import load_torch_payload, build_time_aware_features

CHECKPOINT=HERE.parents[1]/'speed_prediction/stage1_v2_runs/09-17-17-16/best_model.pt'
WIDTHS=(.12,.24,.48)

def rotate(p,yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    result=p.copy(); result[...,0]=c*p[...,0]-s*p[...,1]; result[...,1]=s*p[...,0]+c*p[...,1]
    return result

def audit_rows(path):
    rows=[json.loads(x) for x in path.with_name('velocity_error_audit.jsonl').read_text().splitlines()]
    return sorted(rows,key=lambda r:r['frame_index'])

def rebuild(x,sources,times,position,resets):
    # Keep source/quality/age/mask identical; only FD and position are changed.
    p=position.copy(); p[x[:,23]<.5]=np.nan
    features,_=build_time_aware_features(timestamps_s=times,position_base_m=p,
       robot_linear_velocity_base_mps=x[:,6:9],robot_angular_velocity_base_rps=x[:,9:12],
       source_ids=sources,quality=x[:,15:22],range_valid_mask=x[:,23]>.5)
    result=x.copy(); result[:,:6]=features.numpy()[:,:6]; result[resets,3:6]=0
    return result

def oracle_features(x,sources,times,resets,rows,kind):
    gt=np.array([r['position_gt_base_m'] for r in rows],float)
    yaw=np.array([r['base_roll_pitch_yaw_rad'][2] for r in rows])
    offset=rotate(np.array([r['camera_position_world_m'] for r in rows])-np.array([r['base_position_world_m'] for r in rows]),-yaw)
    measured=x[:,:3]-offset; exact=gt-offset
    mnorm=np.maximum(np.linalg.norm(measured,axis=1),1e-8)
    gnorm=np.maximum(np.linalg.norm(exact,axis=1),1e-8)
    if kind=='range_oracle': pos=offset+measured/mnorm[:,None]*gnorm[:,None]
    elif kind=='direction_oracle': pos=offset+exact/gnorm[:,None]*mnorm[:,None]
    elif kind=='position_oracle': pos=gt
    else: raise ValueError(kind)
    return rebuild(x,sources,times,pos,resets)

def fit_baseline(x,times,resets,width=.48,mode='linear',aligned=False,pose=None,fd_weight=True):
    """Float64, true-timestamp causal fit, O(T*history); inclusive 1 ns tolerance.

    aligned=True uses only integrated yaw rate + base velocity. pose is an
    explicitly labelled GT-pose diagnostic. Current-frame yaw term is applied
    only to the unaligned relative-position regression.
    """
    n=len(x); k=min(n,int(np.ceil(width/max(np.min(np.diff(times)) if n>1 else .02,.001)))+3)
    indices=np.arange(n)[:,None]-np.arange(k)[None,:]
    index=indices.clip(0)
    last_reset=np.maximum.accumulate(np.where(resets,np.arange(n),0))
    tau=times[index]-times[:,None]
    causal=(indices>=last_reset[:,None])&(tau>=-width-1e-9)&(tau<=0)&(indices>=0)
    weights=x[:,23]*(.05+.95*x[:,22])*(.1+.9*x[:,15])
    if fd_weight: weights*=np.minimum(1,30/np.maximum(np.linalg.norm(x[:,3:6],axis=1),1e-6))
    w=weights[index]*causal
    if mode in ('recent','quadratic'): w*=np.exp(tau/max(width/3,.01))
    yaw=np.zeros(n); base=np.zeros((n,3)); pos=x[:,:3].astype(float)
    if aligned:
        if pose is None:
            dt=np.diff(times,prepend=times[0]); yaw=np.cumsum(.5*(x[:,11]+np.r_[x[0,11],x[:-1,11]])*dt)
            world_v=rotate(x[:,6:9],yaw)
            base=np.cumsum(.5*(world_v+np.vstack((world_v[0],world_v[:-1])))*dt[:,None],axis=0)
        else: yaw,base=pose
        pos=rotate(pos,yaw)+base
    p=pos[index]; count=(w>0).sum(1); sw=np.maximum(w.sum(1),1e-12)
    mt=(w*tau).sum(1)/sw
    mp=(w[:,:,None]*p).sum(1)/sw[:,None]
    centered=tau-mt[:,None]; variance=(w*centered**2).sum(1)
    slope=(w[:,:,None]*centered[:,:,None]*(p-mp[:,None,:])).sum(1)/np.maximum(variance[:,None],1e-6)
    if mode=='quadratic':
        design=np.stack((np.ones_like(tau),tau,.5*tau**2),axis=-1)
        for _ in range(2):
            gram=np.einsum('nt,nti,ntj->nij',w,design,design)
            gram+=np.diag([1e-8,1e-6,1e-3])[None]
            rhs=np.einsum('nt,nti,ntd->nid',w,design,p)
            coeff=np.linalg.solve(gram,rhs)
            residual=np.linalg.norm(np.einsum('nti,nid->ntd',design,coeff)-p,axis=-1)
            w=w*np.minimum(1,.05/np.maximum(residual,1e-8))
        slope=np.where((count>=5)[:,None],coeff[:,1],slope)
    slope[count<2]=0
    fitted=mp[:,None,:]+centered[:,:,None]*slope[:,None,:]
    rmse=np.sqrt((w*np.linalg.norm(p-fitted,axis=-1)**2).sum(1)/sw)
    span=np.maximum(0,-np.min(np.where(w>0,tau,0),axis=1))
    if aligned: velocity=rotate(slope,-yaw)-x[:,6:9]
    else:
        velocity=slope.copy(); velocity[:,0]-=x[:,11]*x[:,1]; velocity[:,1]+=x[:,11]*x[:,0]
    velocity[count<2]=0
    last_direct=np.maximum.accumulate(np.where((x[:,22]>.5)&(x[:,23]>.5),np.arange(n),-1))
    age=times-times[last_direct.clip(0)]
    extra=np.column_stack((velocity/10,span/width,count/25,rmse,variance/np.maximum(sw*width**2,1e-9),age))
    return velocity.astype(np.float32),np.nan_to_num(extra).clip(-20,20).astype(np.float32)

def propagated_inputs(x,sources,times):
    """Soft outages: retain memory and propagate last observed causal velocity.
    A new file always represents a true episode reset. Reacquisition within
    that file is an observation update, not a new target identity.
    """
    x=x.copy(); resets=np.zeros(len(x),bool); resets[0]=True
    prior,_=fit_baseline(x,times,resets,aligned=True)
    active=False; p=np.zeros(3); v=np.zeros(3); last_robot=np.zeros(3)
    for i in range(len(x)):
        if x[i,23]>.5 and x[i,22]>.5:
            p=x[i,:3].copy(); v=prior[i].copy(); active=True
        elif active:
            dt=times[i]-times[i-1]; turn=x[i,11]*dt
            v=rotate(v+last_robot,np.array(-turn))-x[i,6:9]
            p=rotate(p,np.array(-turn))+v*dt
            if x[i,23]<.5:x[i,:3]=p; x[i,3:6]=0
        last_robot=x[i,6:9].copy()
    return x,resets

class ExperimentalGRU(TimeAwareQualityGatedGRU):
    def __init__(self,config,quality=False,split_gate=False):
        super().__init__(**config); self.use_quality=quality; self.split_gate=split_gate
        if quality:
            self.baseline_encoder=nn.Sequential(nn.Linear(24,48),nn.SiLU(),nn.Linear(48,self.context_dim))
            nn.init.zeros_(self.baseline_encoder[-1].weight); nn.init.zeros_(self.baseline_encoder[-1].bias)
        if split_gate:
            self.position_gate=nn.Sequential(nn.Linear(18+self.source_embedding_dim,32),nn.SiLU(),nn.Linear(32,1),nn.Sigmoid())
            self.position_encoder=nn.Sequential(nn.Linear(3,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
            self.fd_encoder=nn.Sequential(nn.Linear(3,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
    def _encode_step(self,features,source_ids):
        encoded,gate,dt=super()._encode_step(features,source_ids)
        if self.split_gate:
            normalized,reliability=self._prepare_input(features)
            # Independent position gate omits FD and FD reliability.
            position_context=torch.cat((normalized[:,6:12],normalized[:,12:15],normalized[:,15:22],normalized[:,22:24],self.source_embedding(source_ids)),-1)
            position_gate=self.position_gate(position_context)*features[:,23,None]
            measurement=torch.cat((self.position_encoder(normalized[:,:3])*position_gate,
                                   self.fd_encoder(normalized[:,3:6])*gate),-1)
            encoded=torch.cat((measurement,encoded[:,self.measurement_dim:]),-1)
        if self.use_quality:
            encoded=torch.cat((encoded[:,:self.measurement_dim],encoded[:,self.measurement_dim:]+self.baseline_encoder(self.current_extra)),-1)
        return encoded,gate,dt

class CausalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__(); self.dilation=dilation
        self.a=nn.Conv1d(64,64,3,dilation=dilation); self.b=nn.Conv1d(64,64,3,dilation=dilation)
    def forward(self,x):
        y=F.silu(self.a(F.pad(x,(2*self.dilation,0))))
        y=F.silu(self.b(F.pad(y,(2*self.dilation,0))))
        return x+y

class CausalTCN(nn.Module):
    receptive_field=29
    def __init__(self,normalizer):
        super().__init__(); self.normalizer=normalizer
        self.embedding=nn.Embedding(5,4); self.project=nn.Conv1d(52,64,1)
        self.blocks=nn.Sequential(*(CausalBlock(d) for d in (1,2,4)))
        self.head=nn.Conv1d(64,6,1)
    def forward(self,x,sources,extra,baseline):
        normalized,_=self.normalizer._prepare_input(x)
        inputs=torch.cat((normalized,self.embedding(sources),extra),-1).transpose(1,2)
        output=self.head(self.blocks(F.silu(self.project(inputs)))).transpose(1,2)
        return baseline+output[...,:3],output[...,3:].clamp(-5,2)

class LearnedCVFilter(nn.Module):
    """CV state in locally integrated odometry; predict positive diagonal Q/R.
    Only direct valid observations update. Missing observations only propagate.
    """
    def __init__(self,normalizer,learned=True,q=2.,r=.05):
        super().__init__();self.normalizer=normalizer;self.learned=learned;self.q=q;self.r=r
        self.embedding=nn.Embedding(5,4)
        self.noise=nn.Sequential(nn.Linear(52,48),nn.SiLU(),nn.Linear(48,6))
        nn.init.zeros_(self.noise[-1].weight)
        self.noise[-1].bias.data[:3].fill_(float(np.log(np.expm1(q))))
        self.noise[-1].bias.data[3:].fill_(float(np.log(np.expm1(r))))
        if not learned:
            for p in self.parameters():p.requires_grad_(False)
    def forward_segment(self,x,sources,extra,valid,state=None):
        batch,steps,_=x.shape;dtype=x.dtype;device=x.device
        if state is None:
            mean=torch.zeros(batch,6,device=device);cov=torch.eye(6,device=device)[None].repeat(batch,1,1)*25
            yaw=torch.zeros(batch,device=device);base=torch.zeros(batch,3,device=device);prev_robot=torch.zeros_like(base);prev_omega=yaw.clone();started=torch.zeros(batch,dtype=torch.bool,device=device)
        else:mean,cov,yaw,base,prev_robot,prev_omega,started=state
        if self.learned:
            normalized,_=self.normalizer._prepare_input(x)
            noise=F.softplus(self.noise(torch.cat((normalized,self.embedding(sources),extra),-1)))+1e-4
        else:noise=torch.cat((torch.full_like(x[...,:3],self.q),torch.full_like(x[...,:3],self.r)),-1)
        eye=torch.eye(6,device=device,dtype=dtype)[None].expand(batch,-1,-1)
        identity3=torch.eye(3,device=device,dtype=dtype)[None].expand(batch,-1,-1)
        outputs=[];sigmas=[]
        for i in range(steps):
            dt=x[:,i,12].clamp(0,.5)*valid[:,i];omega=x[:,i,11]
            yaw=yaw+.5*(omega+prev_omega)*dt
            c,s=torch.cos(yaw),torch.sin(yaw)
            rot=torch.zeros(batch,3,3,device=device);rot[:,0,0]=c;rot[:,0,1]=-s;rot[:,1,0]=s;rot[:,1,1]=c;rot[:,2,2]=1
            robot=torch.bmm(rot,x[:,i,6:9,None]).squeeze(-1)
            base=base+.5*(robot+prev_robot)*dt[:,None]
            transition=eye.clone();transition[:,:3,3:]=identity3*dt[:,None,None]
            q=noise[:,i,:3];process=torch.zeros(batch,6,6,device=device)
            process[:,:3,:3]=torch.diag_embed(q*dt[:,None]**3/3)
            process[:,:3,3:]=process[:,3:,:3]=torch.diag_embed(q*dt[:,None]**2/2)
            process[:,3:,3:]=torch.diag_embed(q*dt[:,None])
            mean=torch.bmm(transition,mean[:,:,None]).squeeze(-1);cov=transition@cov@transition.transpose(1,2)+process
            observed=(x[:,i,22]>.5)&(x[:,i,23]>.5)&valid[:,i]
            position=torch.bmm(rot,x[:,i,:3,None]).squeeze(-1)+base
            initial=observed&~started;mean=torch.cat((torch.where(initial[:,None],position,mean[:,:3]),mean[:,3:]),-1)
            innovation=position-mean[:,:3]
            measurement_cov=torch.diag_embed(noise[:,i,3:])
            innovation_cov=cov[:,:3,:3]+measurement_cov+identity3*1e-6
            gain=torch.linalg.solve(innovation_cov,cov[:,:3,:]).transpose(1,2)
            updated=mean+torch.bmm(gain,innovation[:,:,None]).squeeze(-1)
            joseph=eye.clone();joseph[:,:,:3]=joseph[:,:,:3]-gain
            updated_cov=joseph@cov@joseph.transpose(1,2)+gain@measurement_cov@gain.transpose(1,2)
            mean=torch.where(observed[:,None],updated,mean);cov=torch.where(observed[:,None,None],updated_cov,cov)
            started=started|observed
            velocity=torch.bmm(rot.transpose(1,2),mean[:,3:,None]).squeeze(-1)-x[:,i,6:9]
            velocity_cov=rot.transpose(1,2)@cov[:,3:,3:]@rot
            outputs.append(velocity);sigmas.append(.5*torch.log(torch.diagonal(velocity_cov,dim1=1,dim2=2).clamp_min(1e-6)))
            prev_robot=robot;prev_omega=omega
        return torch.stack(outputs,1),torch.stack(sigmas,1).clamp(-5,2),(mean,cov,yaw,base,prev_robot,prev_omega,started)
