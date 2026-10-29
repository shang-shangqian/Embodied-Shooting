"""Check whether old float32 time-window membership explains CPU/CUDA drift."""
import json
from pathlib import Path
import torch
from optimization_components import CHECKPOINT
from analyze_error_sources import _load_model,_stream_arrays
from speed_prediction.time_aware_velocity_data import load_torch_payload,weighted_velocity_baseline,robust_finite_difference

def fit_explicit(f,absolute):
    elapsed=absolute[:,None]-absolute[None,:];mask=(elapsed>=0)&(elapsed<=.48)
    reliability=robust_finite_difference(f,clip_mps=30)[1]
    weight=f[:,23]*(.05+.95*f[:,22])*(.1+.9*f[:,15])*reliability
    w=mask.float()*weight[None,:];sw=w.sum(-1,keepdim=True).clamp_min(1e-6)
    mean_t=(w*absolute[None]).sum(-1,keepdim=True)/sw
    centered=absolute[None]-mean_t;mp=(w[:,:,None]*f[None,:,:3]).sum(1)/sw
    slope=(w[:,:,None]*centered[:,:,None]*(f[None,:,:3]-mp[:,None,:])).sum(1)/(w*centered.square()).sum(-1,keepdim=True).clamp_min(1e-6)
    omega=torch.zeros_like(f[:,:3]);omega[:,2]=f[:,11]
    result=slope+torch.cross(omega,f[:,:3],dim=-1)
    return torch.where(((w>0).sum(-1)>=2)[:,None],result,0),mask

def main():
    root=Path('source/standalone/test11/velocity_error_validation/outputs/pilot_fixed_camera_v2_warmup_flowgate')
    paths=sorted(root.rglob('*.taqstream'));path=paths[0];f,source,v,y,m,r,t=_stream_arrays(load_torch_payload(path),path)
    indices=torch.nonzero(r).flatten().tolist()+[len(f)];start,end=max(zip(indices[:-1],indices[1:]),key=lambda pair:pair[1]-pair[0]);f=f[start:end]
    cpu_t=torch.cumsum(f[:,12].clamp_min(0),0);gpu_t=torch.cumsum(f[:,12].cuda().clamp_min(0),0).cpu()
    cpu,cpu_mask=fit_explicit(f,cpu_t);swapped,gpu_mask=fit_explicit(f,gpu_t)
    gpu=weighted_velocity_baseline(f.cuda()[None],torch.ones(1,len(f),dtype=torch.bool,device='cuda:0'),window_s=.48)[0].cpu()
    summary=dict(path=str(path),segment=[start,end],mask_membership_differences=int((cpu_mask!=gpu_mask).sum()),time_max_difference_s=float((cpu_t-gpu_t).abs().max()),cpu_cuda_mean_vector_difference=float(torch.linalg.vector_norm(cpu-gpu,dim=-1).mean()),cpu_with_cuda_timestamps_mean_vector_difference=float(torch.linalg.vector_norm(swapped-gpu,dim=-1).mean()),max_axis_after_timestamp_swap=float((swapped-gpu).abs().max()))
    out=Path('source/standalone/test11/velocity_error_validation/reports/pilot_fixed_camera_v2_warmup_flowgate_e2_cuda_20261007/device_boundary_check.json');out.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
