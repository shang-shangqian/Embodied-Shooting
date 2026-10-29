"""E2 same-frame oracle and no-learning baseline diagnostics."""
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from optimization_components import *
from analyze_inference_context import prepare_pairs,run_sequences
from analyze_error_sources import _load_model, _segmented_baseline


def metrics(e):
    e=np.concatenate(e); return {'n':len(e),'mean':float(e.mean()),'p95':float(np.quantile(e,.95)),'p99':float(np.quantile(e,.99)),'over_1':float((e>1).mean()),'over_2':float((e>2).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--max-pairs',type=int,default=0); args=ap.parse_args()
    torch.set_num_threads(4); model,_=_load_model(CHECKPOINT,torch.device(args.device))
    streams,total=prepare_pairs(sorted(args.data.rglob('*.taqstream')),args.max_pairs)
    stats=defaultdict(list); episodes=[]; numerical=[]; start=time.time()
    for chunk_start in range(0,len(streams),8):
        chunk=streams[chunk_start:chunk_start+8]; tasks=[]
        for s in chunk:
            f,source,valid,y,labelmask,resets,times=s.arrays
            x=f.numpy();ts=times.numpy();rm=resets.numpy();rows=audit_rows(s.path)
            pose=(np.array([r['base_roll_pitch_yaw_rad'][2] for r in rows]),np.array([r['base_position_world_m'] for r in rows]))
            for kind in ('original','range_oracle','direction_oracle','position_oracle'):
                xf=x if kind=='original' else oracle_features(x,source.numpy(),ts,rm,rows,kind)
                fd=torch.tensor(xf,device=args.device);vd=valid.to(args.device);rd=resets.to(args.device)
                old=_segmented_baseline(fd,vd,rd,model,.48).cpu().numpy()
                tasks.append((s,kind,xf,old,pose))
                if kind=='original':
                    cpu=_segmented_baseline(f,valid,resets,model,.48).numpy()
                    numerical.append({'uid':s.uid,'mean_cpu_cuda_vector_difference':float(np.linalg.norm(cpu-old,axis=1).mean()),'max_axis_difference':float(np.abs(cpu-old).max())})
        length=max(len(task[2]) for task in tasks);batch=len(tasks)
        fb=torch.zeros(batch,length,24,device=args.device);sb=torch.zeros(batch,length,dtype=torch.long,device=args.device)
        vb=torch.zeros(batch,length,dtype=torch.bool,device=args.device);rb=vb.clone()
        for i,(s,kind,xf,old,pose) in enumerate(tasks):
            n=len(xf);fb[i,:n]=torch.tensor(xf,device=args.device);sb[i,:n]=s.arrays[1].to(args.device)
            vb[i,:n]=s.arrays[2].to(args.device);rb[i,:n]=s.arrays[5].to(args.device)
        residuals=run_sequences(model,fb,sb,vb,rb)[0].cpu().numpy()
        for i,(s,kind,xf,old,pose) in enumerate(tasks):
            f,source,valid,y,labelmask,resets,times=s.arrays
            x=f.numpy();ts=times.numpy();rm=resets.numpy();truth=y.numpy()
            preds={'checkpoint_stream':old+residuals[i,:len(x)],'legacy_baseline':old}
            for width in (.12,.24,.48):
                for mode in ('linear','recent','quadratic'):
                    preds[f'{mode}_{width:.2f}']=fit_baseline(xf,ts,rm,width,mode)[0]
                for name,p in (('odometry_aligned',None),('gt_pose_aligned',pose)):
                    preds[f'{name}_{width:.2f}']=fit_baseline(xf,ts,rm,width,'linear',True,p,False)[0]
            for name,pred in preds.items():
                error=np.linalg.norm(pred-truth,axis=1)
                for filt,mask in (('unfiltered',s.score),('center_above_ground',s.score&(s.world_z>=0))):
                    for group,gmask in [('all',np.ones(len(x),bool)),('direct',source.numpy()<=2),('prediction',source.numpy()==3),('lost',source.numpy()==4)]:
                        score=mask&gmask
                        if score.any():stats[(s.condition,kind,name,filt,group)].append(error[score])
                    if mask.any():episodes.append({'uid':s.uid,'condition':s.condition,'oracle':kind,'method':name,'filter':filt,**metrics([error[mask]])})
        print(f'E2 {chunk_start+len(chunk)}/{len(streams)} elapsed={time.time()-start:.1f}s',flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    summary=[dict(condition=k[0],oracle=k[1],method=k[2],filter=k[3],group=k[4],**metrics(v)) for k,v in stats.items()]
    for name,rows in [('summary',summary),('episode_metrics',episodes),('cpu_cuda_baseline',numerical)]:
        with (args.out/f'{name}.csv').open('w') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'manifest.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT),'device':args.device,'data':str(args.data),'pairs':len(streams)//4,'available_pairs':total,'elapsed_s':time.time()-start,'oracle_preserves_source_quality_masks':True,'aligned_odometry_uses':'integrated existing yaw-rate and base linear velocity; no GT pose','gt_pose_aligned':'diagnostic only; never training features','baseline_boundaries':'original resets; timestamps float64 and 1ns tolerance'},indent=2))
if __name__=='__main__':main()
