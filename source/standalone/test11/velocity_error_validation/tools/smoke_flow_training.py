"""One real-data CUDA update per architecture before starting expensive fits."""
import time
from pathlib import Path
import torch
from optimization_components import CHECKPOINT
from train_flow_ablations import VARIANTS,make_model,gru_segment,tcn_segment
from analyze_error_sources import _load_model,_stream_arrays,_segmented_baseline
from speed_prediction.time_aware_velocity_data import load_torch_payload
from speed_prediction.taq_velocity_training import masked_taq_velocity_loss

torch.set_num_threads(4)
path=next(Path('source/standalone/test11/velocity_error_validation/outputs/optimization_20261007/train/ss_clean').rglob('*.taqstream'))
f,source,valid,target,score,reset,times=_stream_arrays(load_torch_payload(path),path)
device=torch.device('cuda:0');template,_=_load_model(CHECKPOINT,device)
base=_segmented_baseline(f.to(device),valid.to(device),reset.to(device),template,.48)
batch={k:value[None,:100].repeat(64,*([1]*(value.ndim))).to(device) for k,value in dict(x=f,source=source,valid=valid,target=target,score=score,reset=reset,base=base,extra=torch.zeros(len(f),24)).items()}
for variant,cfg in VARIANTS.items():
    model=make_model(cfg,device).train();torch.cuda.synchronize();start=time.time()
    if cfg.get('kalman'):mean,sigma,_=model.forward_segment(batch['x'],batch['source'],batch['extra'],batch['valid'])
    elif cfg.get('tcn'):mean,sigma,_=tcn_segment(model,batch,0,100)
    else:mean,sigma,_=gru_segment(model,batch,0,100)
    loss,_=masked_taq_velocity_loss(mean,sigma,batch['target'],batch['score'])
    loss.backward();torch.cuda.synchronize()
    if not torch.isfinite(loss):raise ValueError(f'Invalid loss {variant}')
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError(f'Invalid gradient {variant}')
    print(f'{variant} loss={float(loss):.4f} B64/T100 forward+backward={time.time()-start:.3f}s peakMB={torch.cuda.max_memory_allocated()/1e6:.1f}',flush=True)
    del model,mean,sigma,loss;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
