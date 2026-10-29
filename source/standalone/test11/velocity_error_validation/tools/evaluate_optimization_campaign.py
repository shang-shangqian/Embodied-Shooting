"""Frozen validation selection followed by common-frame independent test."""
import argparse,csv,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from train_flow_ablations import (VARIANTS,make_model,evaluate,batchify,metric_arrays,write_csv)
from optimization_components import CHECKPOINT,WIDTHS,LearnedCVFilter,rotate
from speed_prediction.time_aware_velocity_data import load_torch_payload,weighted_velocity_baseline
from analyze_error_sources import _load_model

@torch.inference_mode()
def original_rolling(model,records,device,width):
    saved=[]
    for r in records:
        x=r['x'];time=r['time'];reset=r['reset'];last=np.maximum.accumulate(np.where(reset.numpy(),np.arange(len(x)),0));tasks=[];pred=np.zeros((len(x),3),np.float32);sigma=pred.copy()
        for i in range(len(x)):
            start=max(int(torch.searchsorted(time,time[i]-width-1e-8)),int(last[i]));tasks.append((start,i+1))
        for offset in range(0,len(tasks),128):
            rows=tasks[offset:offset+128];maxn=max(e-s for s,e in rows);f=torch.zeros(len(rows),maxn,24,device=device);sources=torch.zeros(len(rows),maxn,dtype=torch.long,device=device);valid=torch.zeros(len(rows),maxn,dtype=torch.bool,device=device)
            for j,(start,end) in enumerate(rows):
                n=end-start;f[j,:n]=x[start:end].to(device);f[j,0,3:6]=0;f[j,0,12]=0;sources[j,:n]=r['source'][start:end].to(device);valid[j,:n]=True
            baseline=weighted_velocity_baseline(f,valid,window_s=.48);p,s=model(f,sources,valid,velocity_baseline=baseline)
            for j,(start,end) in enumerate(rows):pred[offset+j]=p[j,end-start-1].cpu().numpy();sigma[offset+j]=torch.exp(s[j,end-start-1]).cpu().numpy()
        saved.append(dict(uid=r['uid'],prediction=pred,sigma=sigma,error=np.linalg.norm(pred-r['target'].numpy(),axis=1),score=r['score'].numpy()))
    return saved


def summarize_saved(records,saved):
    groups=defaultdict(list);episodes=[]
    for r,s in zip(records,saved):
        for filt,mask in [('unfiltered',r['score'].numpy()),('center_above_ground',r['score'].numpy()&r['ground'].numpy())]:
            if mask.any():
                groups[(r['scenario'],filt,'all')].append(s['error'][mask]);episodes.append(dict(uid=r['uid'],scenario=r['scenario'],filter=filt,**metric_arrays(s['error'][mask])))
    return [dict(scenario=k[0],filter=k[1],group=k[2],**metric_arrays(np.concatenate(v))) for k,v in groups.items()],episodes


def forecast_metrics(records,saved,cfg):
    groups=defaultdict(list)
    for r,s in zip(records,saved):
        audit=r['audit_score_only'];times=audit['physics_time'];yaw=audit['yaw'];n=len(times);score=r['score'].numpy();x=r['soft_x' if cfg.get('missing') else 'x'].numpy();observed=(r['x'][:,23].numpy()>.5)&(r['x'][:,22].numpy()>.5)
        world_pos=rotate(x[:,:3],yaw)+audit['world_base'];world_vel=rotate(s['prediction']+x[:,6:9],yaw)
        for horizon in (.1,.2):
            end=np.searchsorted(times,times+horizon-1e-7);ok=(end<n)&score
            clipped=np.minimum(end,n-1);dt=times[clipped]-times
            # True future coordinates are evaluation labels only. No future
            # robot pose is required by this constant-velocity extrapolation.
            displacement=world_vel*dt[:,None]-(audit['world_target'][clipped]-audit['world_target'])
            absolute=world_pos+world_vel*dt[:,None]-audit['world_target'][clipped]
            for filt,fmask in [('unfiltered',np.ones(n,bool)),('center_above_ground',r['ground'].numpy())]:
                for kind,error,mask in [('velocity_displacement',displacement,ok&fmask),('absolute_position_direct',absolute,ok&fmask&observed)]:
                    if mask.any():groups[(r['scenario'],filt,horizon,kind)].append(np.linalg.norm(error[mask],axis=1))
    return [dict(scenario=k[0],filter=k[1],horizon_s=k[2],kind=k[3],unit='m',**metric_arrays(np.concatenate(v))) for k,v in groups.items()]


def paired_bootstrap(records,reference,candidate,repeats=500):
    rows=[];rng=np.random.default_rng(20261007)
    for scenario in ('ss','sd'):
        for filt in ('unfiltered','center_above_ground'):
            errors=[]
            for r,a,b in zip(records,reference,candidate):
                if r['scenario']!=scenario:continue
                mask=r['score'].numpy()
                if filt=='center_above_ground':mask=mask&r['ground'].numpy()
                if mask.any():errors.append((a['error'][mask],b['error'][mask]))
            if not errors:continue
            samples={'mean':[],'p95':[]}
            for _ in range(repeats):
                idx=rng.integers(len(errors),size=len(errors));a=np.concatenate([errors[i][0] for i in idx]);b=np.concatenate([errors[i][1] for i in idx])
                samples['mean'].append(b.mean()-a.mean());samples['p95'].append(np.quantile(b,.95)-np.quantile(a,.95))
            for metric,values in samples.items():rows.append(dict(scenario=scenario,filter=filt,metric=metric,delta_ci_low=float(np.quantile(values,.025)),delta_ci_high=float(np.quantile(values,.975)),episodes=len(errors),bootstrap_unit='episode; fixed trained seed'))
    return rows


def stratified_metrics(records,predictions):
    result=[]
    groups=defaultdict(list)
    for r,p in zip(records,predictions):
        raw=r['x'].numpy();ts=r['time'].numpy();n=len(raw)
        distance=np.linalg.norm(r['audit_score_only']['world_target']-r['audit_score_only']['world_base'],axis=1)
        speed=np.linalg.norm(r['audit_score_only']['world_target_velocity'],axis=1)
        last=np.maximum.accumulate(np.where((raw[:,22]>.5)&(raw[:,23]>.5),np.arange(n),0));age=ts-ts[last]
        fields={'distance_m':(distance,[0,100,120,np.inf]),'target_world_speed_mps':(speed,[0,1,3,np.inf]),'gyro_input_abs_rps':(np.abs(raw[:,11]),[0,.1,.5,np.inf]),'sensor_dt_s':(raw[:,12],[0,.019,.021,np.inf]),'observation_age_s':(age,[0,.1,.5,1,np.inf])}
        for field,(value,edges) in fields.items():
            for low,high in zip(edges[:-1],edges[1:]):
                for filt,score in [('unfiltered',r['score'].numpy()),('center_above_ground',r['score'].numpy()&r['ground'].numpy())]:
                    mask=score&(value>=low)&(value<high)
                    if mask.any():groups[(r['scenario'],filt,field,low,high)].append(p['error'][mask])
    return [dict(scenario=k[0],filter=k[1],field=k[2],low=k[3],high=k[4],**metric_arrays(np.concatenate(v))) for k,v in groups.items()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(4);data=load_torch_payload(args.runs/'prepared_flows.pt');device=torch.device(args.device);report=args.runs/'evaluation';report.mkdir(exist_ok=True)
    selection=[];all_rows=[];bootstrap=[];forecast=[];run_stats=[]
    # Validation records alone determine the architecture choice. Evaluation
    # of all frozen architectures on test is diagnostic, not a new selection.
    for run in sorted(args.runs.glob('*_seed*')):
        if not (run/'COMPLETE').exists():continue
        cp=load_torch_payload(run/'best_model.pt');cfg=cp['variant_config'];selection.append(dict(run=run.name,variant=cp['variant'],seed=cp['seed'],epoch=cp['epoch'],validation_macro_p95=cp['metric']))
    architecture=defaultdict(list)
    for row in selection:architecture[row['variant']].append(row['validation_macro_p95'])
    winner=min(architecture,key=lambda k:np.mean(architecture[k]));chosen=min([row for row in selection if row['variant']==winner],key=lambda row:row['validation_macro_p95'])
    frozen=dict(selection_rule='lowest mean validation macro P95 across training seeds; checkpoint seed then lowest validation P95',variant=winner,run=chosen['run'],architecture_validation_means={k:float(np.mean(v)) for k,v in architecture.items()})
    # Ordinary CV controls do not learn sensor noise. Q/R are selected using
    # validation only and share the learned filter's physical propagation.
    val_scores=[]
    for q in (.1,1.,5.):
        for r in (.01,.1,1.):
            normalizer,_=_load_model(CHECKPOINT,device);fixed=LearnedCVFilter(normalizer,False,q,r).to(device)
            score,_,_,_=evaluate(fixed,data['val'],dict(missing=True,aligned=True,kalman=True),device)
            val_scores.append(dict(q=q,r=r,val_macro_p95=score))
    fixed_choice=min(val_scores,key=lambda row:row['val_macro_p95'])
    baseline_scan=[]
    for missing in (False,True):
        for scale,width in enumerate(WIDTHS):
            cfg=dict(aligned=True,baseline_only=True,missing=missing)
            score,_,_,_=evaluate(torch.nn.Identity(),data['val'],cfg,device,scale=scale)
            baseline_scan.append(dict(missing=missing,scale=scale,width_s=width,val_macro_p95=score))
    baseline_choice=min(baseline_scan,key=lambda row:row['val_macro_p95'])
    family_scores={'trained_'+winner:float(np.mean(architecture[winner])),'fixed_cv':fixed_choice['val_macro_p95'],'aligned_baseline':baseline_choice['val_macro_p95']}
    frozen['recommended_family']=min(family_scores,key=family_scores.get)
    frozen['family_validation_scores']=family_scores
    frozen['aligned_baseline_validation_choice']=baseline_choice
    frozen['fixed_cv_validation_choice']=fixed_choice
    (report/'frozen_selection.json').write_text(json.dumps(frozen,indent=2));write_csv(report/'validation_selection.csv',selection)
    model,_=_load_model(CHECKPOINT,device)
    _,summary,episodes,reference=evaluate(model,data['test'],{},device,return_rows=True)
    write_csv(report/'original_stream_summary.csv',summary);write_csv(report/'original_stream_episodes.csv',episodes);write_csv(report/'original_stream_forecast.csv',forecast_metrics(data['test'],reference,{}))
    for item in selection:
        run=args.runs/item['run'];cp=load_torch_payload(run/'best_model.pt');cfg=cp['variant_config'];model=make_model(cfg,device);model.load_state_dict(cp['state_dict']);scale=cp['scale']
        _,summary,episodes,saved=evaluate(model,data['test'],cfg,device,scale=scale,return_rows=True)
        write_csv(run/'test_summary.csv',summary);write_csv(run/'test_episodes.csv',episodes)
        torch.save(saved,run/'test_predictions.pt')
        if run.name==chosen['run']:selected_saved=saved
        all_rows.extend(dict(run=run.name,variant=cp['variant'],seed=cp['seed'],**row) for row in summary)
        bootstrap.extend(dict(run=run.name,variant=cp['variant'],seed=cp['seed'],**row) for row in paired_bootstrap(data['test'],reference,saved))
        forecast.extend(dict(run=run.name,variant=cp['variant'],seed=cp['seed'],**row) for row in forecast_metrics(data['test'],saved,cfg))
        run_stats.append(dict(run=run.name,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_sha256=hashlib.sha256((run/'best_model.pt').read_bytes()).hexdigest()))
        print(f"Tested {run.name}",flush=True)
    # Predeclared finite-history anchors.
    model,_=_load_model(CHECKPOINT,device)
    for width in (.16,.32):
        saved=original_rolling(model,data['test'],device,width);summary,ep=summarize_saved(data['test'],saved)
        write_csv(report/f'original_rolling_{width:.2f}_summary.csv',summary);write_csv(report/f'original_rolling_{width:.2f}_episodes.csv',ep)
    normalizer,_=_load_model(CHECKPOINT,device);fixed=LearnedCVFilter(normalizer,False,fixed_choice['q'],fixed_choice['r']).to(device)
    _,summary,ep,saved=evaluate(fixed,data['test'],dict(missing=True,aligned=True,kalman=True),device,return_rows=True)
    fixed_saved=saved
    write_csv(report/'fixed_cv_test_summary.csv',summary)
    write_csv(report/'fixed_cv_test_episodes.csv',ep)
    bootstrap.extend(dict(run='fixed_cv',variant='fixed_cv',seed=0,**row) for row in paired_bootstrap(data['test'],reference,saved))
    forecast.extend(dict(run='fixed_cv',variant='fixed_cv',seed=0,**row) for row in forecast_metrics(data['test'],saved,dict(missing=True)))
    baseline_rows=[]
    for choice in baseline_scan:
        cfg=dict(aligned=True,baseline_only=True,missing=choice['missing'])
        _,summary,ep,saved=evaluate(torch.nn.Identity(),data['test'],cfg,device,scale=choice['scale'],return_rows=True)
        baseline_rows.extend(dict(width_s=choice['width_s'],missing=choice['missing'],**row) for row in summary)
    write_csv(report/'aligned_baseline_validation_scan.csv',baseline_scan);write_csv(report/'aligned_baseline_test_summary.csv',baseline_rows)
    write_csv(report/'fixed_cv_validation_scan.csv',val_scores)
    write_csv(report/'test_summary.csv',all_rows);write_csv(report/'paired_bootstrap.csv',bootstrap);write_csv(report/'forecast.csv',forecast);write_csv(report/'model_artifacts.csv',run_stats)
    strata=[]
    for name,saved in [('original_stream',reference),(chosen['run'],selected_saved),('fixed_cv',fixed_saved)]:strata.extend(dict(method=name,**row) for row in stratified_metrics(data['test'],saved))
    write_csv(report/'stratified_test.csv',strata)
    development=load_torch_payload(args.runs/'development_inputs.pt')
    chosen_cp=load_torch_payload(args.runs/chosen['run']/'best_model.pt')
    selected_model=make_model(chosen_cp['variant_config'],device);selected_model.load_state_dict(chosen_cp['state_dict'])
    original_model,_=_load_model(CHECKPOINT,device)
    development_rows=[]
    for dataset,records in development.items():
        for name,m,cfg in [('original_stream',original_model,{}),('selected_'+chosen['run'],selected_model,chosen_cp['variant_config']),('fixed_cv',fixed,dict(missing=True,aligned=True,kalman=True))]:
            _,summary,ep,saved=evaluate(m,records,cfg,device,scale=chosen_cp['scale'],return_rows=True)
            development_rows.extend(dict(dataset=dataset,method=name,**row) for row in summary)
        print(f'Development regression {dataset}',flush=True)
    write_csv(report/'development_regression.csv',development_rows)
    (report/'COMPLETE').write_text('Frozen-test evaluation complete.\n')
if __name__=='__main__':main()
