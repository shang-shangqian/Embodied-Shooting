"""Target-device causal replay costs, excluding visual frontend."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from train_flow_ablations import make_model,gru_segment,tcn_segment,batchify,write_csv
from optimization_components import fit_baseline
from speed_prediction.time_aware_velocity_data import load_torch_payload

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda:0');data=load_torch_payload(args.runs/'prepared_flows.pt');rows=[]
    for run in sorted(args.runs.glob('*_seed41')):
        cp=load_torch_payload(run/'best_model.pt');cfg=cp['variant_config'];model=make_model(cfg,device).eval();model.load_state_dict(cp['state_dict'])
        for b in (1,8,32):
            batch=batchify([data['test'][0]]*b,cfg,device,cp['scale']);length=batch['x'].shape[1];start=max(0,length-1);state=None
            # Warm recurrent state without measuring it.
            if cfg.get('kalman'):
                _,_,state=model.forward_segment(batch['x'][:,:start],batch['source'][:,:start],batch['extra'][:,:start],batch['valid'][:,:start])
            elif not cfg.get('tcn'):_,_,state=gru_segment(model,batch,0,start)
            times=[]
            for i in range(110):
                torch.cuda.synchronize();begin=time.perf_counter()
                if cfg.get('kalman'):model.forward_segment(batch['x'][:,start:],batch['source'][:,start:],batch['extra'][:,start:],batch['valid'][:,start:],state)
                elif cfg.get('tcn'):tcn_segment(model,batch,start,length)
                else:gru_segment(model,batch,start,length,state)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
                if i>=10:times.append(elapsed)
            rows.append(dict(variant=cp['variant'],targets=b,component='cached-input model update including Python/CUDA sync',p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),peak_allocated_MB=torch.cuda.max_memory_allocated()/1e6))
        del model;torch.cuda.empty_cache()
    r=data['test'][0];x=r['soft_x'].numpy();ts=r['time'].numpy();keep=ts>=ts[-1]-.55;x=x[keep];ts=ts[keep];reset=np.zeros(len(x),bool);reset[0]=True
    times=[]
    for i in range(210):
        begin=time.perf_counter()
        for width in (.12,.24,.48):fit_baseline(x,ts,reset,width,aligned=True,fd_weight=False)
        if i>=10:times.append((time.perf_counter()-begin)*1000)
    rows.append(dict(variant='all aligned quality candidates',targets=1,component='CPU three-scale baseline+quality on 0.55s buffer',p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),peak_allocated_MB=0))
    write_csv(args.runs/'evaluation/latency.csv',rows)
    (args.runs/'evaluation/latency_limitations.json').write_text(json.dumps(dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,excludes=['camera rendering','LK/OSTrack','full feature construction','host-to-device transfer','production target management'],tcn='recompute causal 29-frame history per update',other_jobs='present; measurements are shared-device wall times, not isolated GPU upper bounds'),indent=2))
if __name__=='__main__':main()
