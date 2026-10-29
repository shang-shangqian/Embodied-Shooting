"""Separate frame alignment and FD reliability, sharing E2 scoring frames."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from optimization_components import fit_baseline
from analyze_inference_context import prepare_pairs
from analyze_observation_baselines import metrics
from train_flow_ablations import write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);streams,_=prepare_pairs(sorted(args.data.rglob('*.taqstream')),0);stats=defaultdict(list);start=time.time()
    for si,s in enumerate(streams):
        f,source,valid,y,labelmask,resets,times=s.arrays
        for aligned in (False,True):
            for fd in (False,True):
                for width in (.12,.24,.48):
                    b,_=fit_baseline(f.numpy(),times.numpy(),resets.numpy(),width,aligned=aligned,fd_weight=fd)
                    e=np.linalg.norm(b-y.numpy(),axis=1)
                    for filt,mask in (('unfiltered',s.score),('center_above_ground',s.score&(s.world_z>=0))):
                        if mask.any():stats[(s.condition,aligned,fd,width,filt)].append(e[mask])
        if (si+1)%128==0:print(f'factorial {si+1}/{len(streams)} {time.time()-start:.1f}s',flush=True)
    args.out.mkdir(exist_ok=True,parents=True)
    write_csv(args.out/'baseline_factorial.csv',[dict(condition=k[0],aligned=k[1],fd_weight=k[2],width=k[3],filter=k[4],**metrics(v)) for k,v in stats.items()])
if __name__=='__main__':main()
