"""Controlled complete-flow TBPTT campaign, independent seeds, frozen test.

Experimental outputs are isolated from the production model. Sequence slots
hold whole episodes; segment transitions detach numerical hidden state.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time,copy,os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from optimization_components import (CHECKPOINT,WIDTHS,ExperimentalGRU,CausalTCN,LearnedCVFilter,audit_rows,fit_baseline,propagated_inputs,rotate)
from analyze_error_sources import _load_model,_stream_arrays,_segmented_baseline
from speed_prediction.time_aware_velocity_data import load_torch_payload
from speed_prediction.taq_velocity_training import masked_taq_velocity_loss

VARIANTS={
 'cold_segments':dict(cold=True),
 'cold_0p48_state':dict(cold_period=24),
 'flow_original':{},
 'flow_aligned':dict(aligned=True),
 'flow_quality':dict(quality=True),
 'flow_split_gate':dict(split_gate=True),
 'flow_missing':dict(missing=True),
 'flow_combined':dict(aligned=True,quality=True,split_gate=True,missing=True),
 'flow_combined_scratch':dict(aligned=True,quality=True,split_gate=True,missing=True,scratch=True),
 'flow_combined_tail':dict(aligned=True,quality=True,split_gate=True,missing=True,tail=True),
 'tcn_combined':dict(aligned=True,quality=True,missing=True,tcn=True),
 'learned_filter':dict(aligned=True,quality=True,missing=True,kalman=True),
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_record(path,template,split="development"):
    payload=load_torch_payload(path);meta=payload['metadata'];f,s,v,y,m,r,t=_stream_arrays(payload,path)
    if meta['condition']!='D' or meta.get('gt_direction_used_for_model_input',False):raise ValueError('Real input only')
    rows=audit_rows(path)
    if any(row.get('gt_direction_used_for_model_input') or row.get('gt_range_used_for_model_input') for row in rows):raise ValueError('Oracle input contamination')
    if len(rows)!=len(f):raise ValueError('Incomplete audit')
    x=f.numpy();ts=t.numpy();resets=r.numpy(); xp,rp=propagated_inputs(x,s.numpy(),ts)
    old=_segmented_baseline(f.cuda(),v.cuda(),r.cuda(),template,.48).cpu().numpy()
    soft_old=_segmented_baseline(torch.tensor(xp,device='cuda:0'),v.cuda(),torch.tensor(rp,device='cuda:0'),template,.48).cpu().numpy()
    baselines=[];extras=[];soft_bases=[];soft_extras=[];legacy_extras=[];soft_legacy_extras=[]
    for width in WIDTHS:
        _,le=fit_baseline(x,ts,resets,width,aligned=False,fd_weight=True);legacy_extras.append(le)
        _,le=fit_baseline(xp,ts,rp,width,aligned=False,fd_weight=True);soft_legacy_extras.append(le)
        b,e=fit_baseline(x,ts,resets,width,aligned=True,fd_weight=False);baselines.append(b);extras.append(e)
        b,e=fit_baseline(xp,ts,rp,width,aligned=True,fd_weight=False);soft_bases.append(b);soft_extras.append(e)
    # During complete missing use last observed velocity with robot motion
    # propagation rather than a zero/ill-conditioned fit.
    for bases,xx in ((soft_bases,xp),):
        for b in bases:
            for i in range(1,len(b)):
                if xx[i,23]<.5:b[i]=rotate(b[i-1]+xx[i-1,6:9],np.array(-xx[i,11]*(ts[i]-ts[i-1])))-xx[i,6:9]
    for b,e in zip(soft_bases,soft_extras):e[:,:3]=b/10
    legacy_extras[-1][:,:3]=old/10;soft_legacy_extras[-1][:,:3]=soft_old/10
    z=np.array([row['target_position_world_m'][2] for row in rows]); eligible=v.numpy()&m.numpy()
    obs=np.flatnonzero((x[:,22]>.5)&(x[:,23]>.5))
    if len(obs):eligible &= ts>=ts[obs[0]]
    else:eligible[:]=False
    record=dict(path=str(path),uid=meta['episode_uid'],scenario=meta['scenario'],profile=meta.get('degradation',{}).get('profile','unknown'),trajectory_uid=meta['trajectory_uid'],seed=meta.get('environment_seed',meta.get('seed')),x=f,soft_x=torch.tensor(xp),source=s,valid=v,target=y,score=torch.tensor(eligible),ground=torch.tensor(z>=0),audit_score_only=dict(world_target=np.array([row['target_position_world_m'] for row in rows]),world_base=np.array([row['base_position_world_m'] for row in rows]),yaw=np.array([row['base_roll_pitch_yaw_rad'][2] for row in rows]),physics_time=np.array([row['physics_timestamp_s'] for row in rows])),reset=r,soft_reset=torch.tensor(rp),time=t,old=torch.tensor(old),soft_old=torch.tensor(soft_old),aligned=torch.tensor(np.stack(baselines,1)),soft_aligned=torch.tensor(np.stack(soft_bases,1)),legacy_extra=torch.tensor(np.concatenate(legacy_extras,1)),soft_legacy_extra=torch.tensor(np.concatenate(soft_legacy_extras,1)),extra=torch.tensor(np.concatenate(extras,1)),soft_extra=torch.tensor(np.concatenate(soft_extras,1)))
    label_world_v=rotate(y.numpy()+np.array([row['robot_linear_velocity_base_mps'] for row in rows]),record['audit_score_only']['yaw'])
    label_fd=np.diff(record['audit_score_only']['world_target'],axis=0)/np.diff(record['audit_score_only']['physics_time'])[:,None]
    kinematic_error=np.linalg.norm(label_fd-.5*(label_world_v[1:]+label_world_v[:-1]),axis=1)
    record['audit_score_only']['world_target_velocity']=label_world_v
    record['kinematic_residual']=torch.tensor(np.r_[0,kinematic_error])
    return record, dict(split=split,path=str(path),sha256=sha(path),audit_sha256=sha(path.with_name('velocity_error_audit.jsonl')),uid=record['uid'],scenario=record['scenario'],seed=record['seed'],frames=len(f),score_frames=int(eligible.sum()),ground_score_frames=int((eligible&(z>=0)).sum()),lost_frames=int((s==4).sum()),prediction_frames=int((s==3).sum()),reset_frames=int(r.sum()),label_valid_frames=int(m.sum()),post_start_lost_frames=int(((s.numpy()==4)&eligible).sum()),post_start_kinematic_residual_over2=int(((kinematic_error>2)&eligible[1:]).sum()),kinematic_residual_p95=float(np.quantile(kinematic_error,.95)),kinematic_residual_max=float(kinematic_error.max()),initialization_delay_s=float(ts[obs[0]]-ts[0]) if len(obs) else None)


def prepare(root,out,include_test=True):
    out.mkdir(parents=True,exist_ok=True); template,_=_load_model(CHECKPOINT,torch.device('cuda:0')); torch.set_num_threads(4)
    data={}; manifest=[]
    splits=('train','val','test') if include_test else ('train','val')
    for split in splits:
        if len(list((root/split).glob('*/COLLECTION_COMPLETE')))!=4:raise RuntimeError(f'{split} collection is incomplete')
        streams=[]
        for path in sorted((root/split).rglob('*.taqstream')):
            record,row=prepare_record(path,template,split)
            streams.append(record);manifest.append(row)
        if not streams:raise ValueError(f'No streams in {split}')
        data[split]=streams;print(f'Prepared {split}: {len(streams)} episodes',flush=True)
    # Trajectories must be disjoint even if run IDs differ.
    identities=[{item['seed'] for item in data[split]} for split in splits]
    if any(identities[i]&identities[j] for i in range(len(splits)) for j in range(i)):raise ValueError('Split overlap')
    (out/('data_manifest.json' if include_test else 'data_manifest_trainval.json')).write_text(json.dumps(manifest,indent=2));torch.save(data,out/('prepared_flows.pt' if include_test else 'prepared_trainval.pt'))
    return data


def batchify(records,cfg,device,scale=1):
    n=max(len(r['x']) for r in records); b=len(records)
    fields={'x':(24,),'extra':(24,),'base':(3,),'target':(3,),'source':(),'valid':(),'reset':(),'score':(),'ground':(),'time':()}
    batch={k:torch.zeros((b,n,*shape),dtype=torch.long if k=='source' else torch.bool if k in ('valid','reset','score','ground') else torch.float32,device=device) for k,shape in fields.items()}
    for i,r in enumerate(records):
        length=len(r['x']);soft=cfg.get('missing',False)
        mapping={'x':r['soft_x' if soft else 'x'],'extra':r[('soft_' if soft else '')+('extra' if cfg.get('aligned') else 'legacy_extra')],'base':r['soft_aligned' if soft else 'aligned'][:,scale] if cfg.get('aligned') else r['soft_old' if soft else 'old'],'reset':r['soft_reset' if soft else 'reset']}
        for k in fields:batch[k][i,:length]=mapping[k] if k in mapping else r[k]
    return batch


def make_model(cfg,device):
    original,_=_load_model(CHECKPOINT,torch.device('cpu'))
    if cfg.get('tcn') or cfg.get('kalman'):
        for p in original.parameters():p.requires_grad_(False)
        model=LearnedCVFilter(original) if cfg.get("kalman") else CausalTCN(original)
    else:
        model=ExperimentalGRU(original.config,cfg.get('quality',False),cfg.get('split_gate',False))
        if cfg.get('scratch'):
            model.set_input_norm_stats(original.input_mean.flatten(),original.input_std.flatten())
        else:model.load_state_dict(original.state_dict(),strict=False)
        if cfg.get('split_gate'):
            if not cfg.get('scratch'):
                # Warm-start each independent branch from its corresponding
                # input/output columns; gates then train independently.
                with torch.no_grad():
                    model.position_gate[0].weight.copy_(torch.cat((original.reliability_gate[0].weight[:,:18],original.reliability_gate[0].weight[:,19:]),-1))
                    model.position_gate[0].bias.copy_(original.reliability_gate[0].bias)
                    model.position_gate[2].load_state_dict(original.reliability_gate[2].state_dict())
                    for encoder,col,row in ((model.position_encoder,slice(0,3),slice(0,32)),(model.fd_encoder,slice(3,6),slice(32,64))):
                        encoder[0].weight.copy_(original.measurement_encoder[0].weight[:,col]);encoder[0].bias.copy_(original.measurement_encoder[0].bias)
                        encoder[2].weight.copy_(original.measurement_encoder[3].weight[row]);encoder[2].bias.copy_(original.measurement_encoder[3].bias[row])
            # Remove the unused original encoder from trainable parameter count.
            for p in model.measurement_encoder.parameters():p.requires_grad_(False)
    return model.to(device)


def gru_segment(model,batch,start,end,hidden=None,cold=False,cold_period=0):
    x=batch['x'][:,start:end];src=batch['source'][:,start:end];b,t,_=x.shape
    model.current_extra=batch['extra'][:,start:end].reshape(-1,24)
    encoded,gate,dt=model._encode_step(x.reshape(-1,24),src.reshape(-1))
    encoded=encoded.reshape(b,t,-1);dt=dt.reshape(b,t)
    if hidden is None or cold:hidden=model.initial_hidden(b,device=x.device)
    tau=F.softplus(model.log_time_constant)+1e-3; states=[]
    for i in range(t):
        if cold_period and (start+i)%cold_period==0:hidden=torch.zeros_like(hidden)
        hidden=torch.where(batch['reset'][:,start+i,None],torch.zeros_like(hidden),hidden)
        candidate=model.gru_cell(encoded[:,i],hidden*torch.exp(-dt[:,i,None]/tau))
        hidden=torch.where(batch['valid'][:,start+i,None],candidate,hidden);states.append(hidden)
    head=model.head(torch.stack(states,1))
    return batch['base'][:,start:end]+model.velocity_delta_head(head),model.log_sigma_head(head).clamp(-5,2),hidden


def tcn_segment(model,batch,start,end):
    left=max(0,start-28)
    mean,sigma=model(batch['x'][:,left:end],batch['source'][:,left:end],batch['extra'][:,left:end],batch['base'][:,left:end])
    return mean[:,start-left:],sigma[:,start-left:],None


def metric_arrays(errors):
    if not len(errors):return {'n':0}
    return dict(n=len(errors),mean=float(np.mean(errors)),p95=float(np.quantile(errors,.95)),p99=float(np.quantile(errors,.99)),over_1=float((errors>1).mean()),over_2=float((errors>2).mean()))

@torch.inference_mode()
def evaluate(model,records,cfg,device,batch_size=64,scale=1,return_rows=False):
    model.eval(); groups=defaultdict(list);episode_rows=[];saved=[]
    for offset in range(0,len(records),batch_size):
        recs=records[offset:offset+batch_size];batch=batchify(recs,cfg,device,scale);hidden=None;pred=[];sigma=[]
        for start in range(0,batch['x'].shape[1],100):
            end=min(start+100,batch['x'].shape[1])
            if cfg.get('baseline_only'):p=batch['base'][:,start:end];s=torch.full_like(p,float('nan'));hidden=None
            elif cfg.get('kalman'):p,s,hidden=model.forward_segment(batch['x'][:,start:end],batch['source'][:,start:end],batch['extra'][:,start:end],batch['valid'][:,start:end],hidden)
            elif cfg.get('tcn'):p,s,hidden=tcn_segment(model,batch,start,end)
            else:p,s,hidden=gru_segment(model,batch,start,end,hidden)
            pred.append(p);sigma.append(s)
        pred=torch.cat(pred,1).cpu().numpy();sig=torch.exp(torch.cat(sigma,1)).cpu().numpy()
        for i,r in enumerate(recs):
            n=len(r['x']);p=pred[i,:n];truth=r['target'].numpy();e=np.linalg.norm(p-truth,axis=1)
            score=r['score'].numpy();source=r['source'].numpy()
            last_direct=np.maximum.accumulate(np.where(source<=2,np.arange(n),0))
            # Recovery is first 200ms after a non-direct stretch, not all direct.
            event=np.zeros(n,bool);direct=(r['x'][:,22].numpy()>.5)&(r['x'][:,23].numpy()>.5)
            clock=r['audit_score_only']['physics_time'];first=np.flatnonzero(direct)
            for index in np.flatnonzero(direct&~np.r_[True,direct[:-1]]):
                if len(first) and index>first[0]:event[index:np.searchsorted(clock,clock[index]+.2-1e-8)]=True
            for filt,mask in (('unfiltered',score),('center_above_ground',score&r['ground'].numpy())):
                for group,gmask in [('all',np.ones(n,bool)),('direct',source<=2),('prediction',source==3),('lost',source==4),('range_unavailable',r['x'][:,23].numpy()<.5),('direct_available',direct),('recovery_200ms',event),('age_gt5s',r['time'].numpy()-float(r['time'][0])>5)]:
                    scored=mask&gmask
                    if scored.any():groups[(r['scenario'],filt,group)].append(e[scored])
                if mask.any():
                    coverage=np.mean(np.all(np.abs(p[mask]-truth[mask])<=1.96*sig[i,:n][mask],axis=1))
                    episode_rows.append(dict(uid=r['uid'],scenario=r['scenario'],profile=r['profile'],filter=filt,coverage_95_box=None if cfg.get('baseline_only') else float(coverage),coverage_95_marginal=None if cfg.get('baseline_only') else float(np.mean(np.abs(p[mask]-truth[mask])<=1.96*sig[i,:n][mask])),**metric_arrays(e[mask])))
            if return_rows:saved.append(dict(uid=r['uid'],prediction=p,sigma=sig[i,:n],error=e,score=score))
    summary=[dict(scenario=k[0],filter=k[1],group=k[2],**metric_arrays(np.concatenate(v))) for k,v in groups.items()]
    lookup={r['scenario']:r['p95'] for r in summary if r['filter']=='unfiltered' and r['group']=='all'}
    metric=.4*lookup.get('ss',float('inf'))+.6*lookup.get('sd',float('inf'))
    return metric,summary,episode_rows,saved


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def train_one(data,out,variant,seed,args):
    cfg=VARIANTS[variant]; run=out/f'{variant}_seed{seed}';run.mkdir(exist_ok=True)
    if (run/'COMPLETE').exists():return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);model=make_model(cfg,device);optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    best=float('inf');log=[];start_time=time.time();rng=random.Random(seed)
    for epoch in range(1,args.epochs+1):
        model.train();order=list(data['train']);rng.shuffle(order);losses=[]
        for offset in range(0,len(order),args.batch_size):
            batch=batchify(order[offset:offset+args.batch_size],cfg,device,args.scale);hidden=None
            # Cold control uses the same frames, optimizer updates and loss;
            # only numerical state is cleared at each TBPTT boundary.
            segment=100
            for start in range(0,batch['x'].shape[1],segment):
                end=min(start+segment,batch['x'].shape[1]);optim.zero_grad(set_to_none=True)
                if cfg.get('kalman'):mean,sigma,hidden=model.forward_segment(batch['x'][:,start:end],batch['source'][:,start:end],batch['extra'][:,start:end],batch['valid'][:,start:end],hidden)
                elif cfg.get('tcn'):mean,sigma,hidden=tcn_segment(model,batch,start,end)
                else:mean,sigma,hidden=gru_segment(model,batch,start,end,hidden,cfg.get('cold',False),cfg.get('cold_period',0))
                mask=batch['score'][:,start:end];target=batch['target'][:,start:end]
                if mask.any():
                    loss,_=masked_taq_velocity_loss(mean,sigma,target,mask)
                    if cfg.get('tail'):
                        error=torch.linalg.vector_norm(mean-target,dim=-1)[mask]
                        loss=loss+.05*torch.topk(error,max(1,int(len(error)*.1))).values.mean()
                    if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step();losses.append(float(loss.detach()))
                if hidden is not None:hidden=tuple(h.detach() for h in hidden) if isinstance(hidden,tuple) else hidden.detach()
        metric,summary,ep,_=evaluate(model,data['val'],cfg,device,args.batch_size,args.scale)
        log.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_macro_p95=metric,elapsed_s=time.time()-start_time))
        if metric<best:
            best=metric;torch.save(dict(state_dict=model.state_dict(),original_config=_load_model(CHECKPOINT,torch.device('cpu'))[0].config,variant=variant,variant_config=cfg,seed=seed,epoch=epoch,scale=args.scale,metric=metric,args=vars(args)),run/'best_model.pt')
            write_csv(run/'validation_summary.csv',summary);write_csv(run/'validation_episodes.csv',ep)
        write_csv(run/'training_history.csv',log)
        print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} valP95={metric:.4f} best={best:.4f} elapsed={time.time()-start_time:.1f}s',flush=True)
    (run/'COMPLETE').write_text('Training complete; independent test not used for selection.\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--train-val-only',action='store_true');ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--scale',type=int,default=1);ap.add_argument('--variants',nargs='+',default=list(VARIANTS));ap.add_argument('--seeds',nargs='+',type=int,default=[41,42,43]);args=ap.parse_args()
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/('prepared_trainval.pt' if args.train_val_only else 'prepared_flows.pt');data=load_torch_payload(cache) if cache.exists() else prepare(args.data,args.out,not args.train_val_only)
    if args.prepare_only:return
    (args.out/'protocol.json').write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'checkpoint_sha':sha(CHECKPOINT),'tbptt_steps':100,'true_resets':'original except missing ablation; always new episode reset','training_inputs':'source, quality, observed position, robot motion; no GT audit input','selection':'.4 SS P95+.6 SD P95, unfiltered val only','test_used_for_training':False,'frozen_baseline_scale_s':WIDTHS[args.scale]},indent=2))
    for variant in args.variants:
        for seed in args.seeds:train_one(data,args.out,variant,seed,args)
if __name__=='__main__':main()
