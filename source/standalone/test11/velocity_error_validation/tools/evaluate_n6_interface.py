"""Frozen N6 interface diagnostic: paired position forecasts and joint uncertainty.

GT world positions/poses and future coordinates enter scoring only. Estimators
accept current capture ticks and already arrived packets, never audit labels.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import chi2
from n0_time_replay import CausalSensorTimeEstimator, NoiseBackend, delivery_schedule, PROFILES
from optimization_components import LearnedCVFilter, rotate
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU

BASE=Path('source/standalone/test11/velocity_error_validation')
CHECKPOINT=BASE/'reports/optimization_flow_runs_20261007/learned_filter_seed41/best_model.pt'
SOURCE_RUNS=(BASE/'reports/n0_time_contract_20261007',BASE/'reports/n0_actual_sampling_20261007')
HORIZONS=(0.,.1,.2)
FIELDS=('dataset','scenario','profile','backend','filter','event','horizon_s','method')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(a):
    return dict(n=len(a),mean=float(np.mean(a)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),maximum=float(np.max(a)))


def uncertainty_arrays(error,covariance,device):
    """Full ellipsoid coverage; GPU float64 solves, with no covariance clipping."""
    e=torch.as_tensor(error,dtype=torch.float64,device=device)
    p=torch.as_tensor(covariance,dtype=torch.float64,device=device)
    eig=torch.linalg.eigvalsh(p)
    if not bool(torch.isfinite(p).all()) or float(eig.min())<=0:
        raise ValueError('Nonfinite or non-positive joint forecast covariance')
    nees=(e*torch.linalg.solve(p,e[...,None])[...,0]).sum(-1)
    axis=(e.abs()<=1.959963984540054*torch.sqrt(torch.diagonal(p,dim1=-2,dim2=-1))).double().mean(-1)
    dim=e.shape[-1];threshold=float(chi2.ppf(.95,dim))
    logdet=torch.linalg.slogdet(p)[1]
    volume=math.pi**(dim/2)/math.gamma(dim/2+1)*threshold**(dim/2)*torch.exp(.5*logdet)
    return dict(nees=nees.cpu().numpy(),coverage95=(nees<=threshold).double().cpu().numpy(),
                axis_coverage95=axis.cpu().numpy(),volume=volume.cpu().numpy(),
                min_eigenvalue=eig[:,0].cpu().numpy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE/'reports/n6_interface_20261007')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'COMPLETE').exists():raise ValueError('Use a new output directory; completed experiment is immutable')
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=LearnedCVFilter(TimeAwareQualityGatedGRU(**cp['original_config'])).to(args.device).eval()
    model.load_state_dict(cp['state_dict'])
    backends={'frozen_learned_filter':NoiseBackend(model,args.device),'fixed_cv':NoiseBackend(q=.1,r=.1)}
    sources=[Path(__file__),BASE/'tools/n0_time_replay.py',BASE/'tools/online_experimental_velocity.py',BASE/'tools/optimization_components.py',BASE.parent/'speed_prediction/filter_state_output.py',BASE/'tools/run_n6_interface_campaign.sh',BASE/'tests/test_filter_state_output.py']
    protocol=dict(work='2 / N6 interface diagnostic',checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
        device=args.device,source_hashes={str(p):digest(p) for p in sources},
        candidate_training=False,parameter_search=False,fixed_cv_q=.1,fixed_cv_r=.1,replay_lag_s=.20,
        horizons_s=HORIZONS,profiles=PROFILES,
        comparison='raw acquired position propagated over measurement age + horizon versus current posterior position + horizon, using exactly the same target absolute local velocity',
        direct_definition='valid direct direction/range exposure pair newly incorporated at this output tick',
        prediction_definition='initialized outputs with no newly incorporated direct pair; uninitialized outputs scored separately',
        score_mask='unchanged N0 score; ground means target above ground at forecast origin; forecast endpoint must exist within same episode',
        forecast_model='CV; current learned Q held constant throughout forecast; includes Ppv/Pvp and full integrated Q',
        covariance_scope='conditional target model; excludes estimated base pose/velocity/clock uncertainty',
        gt_scope='world anchor, label velocities and future world positions only in scoring, never estimator inputs',
        input_environment='isaac_lab_232 + Isaac Sim setup; existing complete N0 acquisitions, no new training/collection')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    forecast_pool=defaultdict(list);group_pool=defaultdict(list);uncertainty_pool=defaultdict(list);bootstrap_pool=defaultdict(lambda:defaultdict(list))
    episodes=[];consistency=[];stored=[];manifest_out=[];runtime=[]
    wall=time.monotonic();stream_count=0
    for source_run in SOURCE_RUNS:
        if not (source_run/'COMPLETE').exists():raise ValueError('N0 reference incomplete')
        records=json.loads((source_run/'data_manifest.json').read_text())
        saved=torch.load(source_run/'predictions.pt',map_location='cpu',weights_only=False)
        reference={(r['uid'],r['profile'],r['backend']):r for r in saved if r['method']=='sample_time_replay'}
        dataset='actual_sparse' if 'actual_sampling' in source_run.name else 'dense'
        if args.smoke:
            groups=sorted({r['group'] for r in records});records=[next(r for r in records if r['group']==g) for g in groups]
        for meta in records:
            path=Path(meta['packet_file'])
            if digest(path)!=meta['packet_sha256']:raise ValueError('Capture packet hash changed')
            if digest(path.with_name('taq_velocity_episode.taqstream'))!=meta['stream_sha256'] or digest(path.with_name('velocity_error_audit.jsonl'))!=meta['audit_sha256']:
                raise ValueError('Scoring stream/audit hash changed')
            packets=torch.load(path,map_location='cpu',weights_only=False)['packets']
            uid=meta['uid'];times=np.array([p['capture_time_s'] for p in packets]);n=len(times)
            stream=torch.load(path.with_name('taq_velocity_episode.taqstream'),map_location='cpu',weights_only=False)
            audit=[json.loads(r) for r in path.with_name('velocity_error_audit.jsonl').read_text().splitlines()]
            # Evaluation-only world/local alignment at the episode's first physical tick.
            world_target=np.array([r['target_position_world_m'] for r in audit])
            true_yaw=np.array([r['base_roll_pitch_yaw_rad'][2] for r in audit])
            origin=np.array(audit[0]['base_position_world_m']);yaw0=true_yaw[0]
            truth_local=rotate(world_target-origin,np.array(-yaw0))
            true_velocity_world=rotate(stream['target_velocity_base'].numpy()+np.array([r['robot_linear_velocity_base_mps'] for r in audit]),true_yaw)
            true_velocity_local=rotate(true_velocity_world,np.array(-yaw0))
            seed=meta['seed']+10000+100*int(path.parents[1].name.split('_')[1])+int(path.parent.name.split('_')[1])
            profiles=('sync',) if dataset=='actual_sparse' else (('sync','combined') if args.smoke else PROFILES)
            for profile in profiles:
                for backend_name,backend in backends.items():
                    old=reference[uid,profile,backend_name]
                    queue,_,_=delivery_schedule(packets,profile,seed)
                    runner=CausalSensorTimeEstimator(backend);outputs=[];latency=[]
                    for packet in packets:
                        available=queue.arrived(packet['capture_time_s'])
                        before=time.perf_counter()
                        out=runner.advance(packet['sample_id'],packet['capture_time_s'],available)
                        latency.append((time.perf_counter()-before)*1000);outputs.append(out)
                    prediction=np.array([o['velocity_base_mps'] for o in outputs])
                    delta=np.linalg.norm(prediction-old['prediction'],axis=1)
                    if delta.max()>1e-10:raise AssertionError(f'Work 2 changed frozen N0 velocity: {uid} {profile} {backend_name}: {delta.max()}')
                    mean=np.array([o['state_local'] for o in outputs]);cov=np.array([o['covariance_local'] for o in outputs])
                    noise=np.array([o['process_intensity_local'] for o in outputs])
                    direct=np.array([o['direct_update'] for o in outputs]);initialized=np.array([o['initialized'] for o in outputs])
                    raw=np.array([o['last_observation_local_m'] if o['last_observation_local_m'] is not None else [np.nan]*3 for o in outputs])
                    age=np.array([o['measurement_age_s'] if o['measurement_age_s'] is not None else np.nan for o in outputs])
                    score=np.asarray(old['score']);ground=np.asarray(old['ground'])
                    states_time=np.array([o['state_time_s'] for o in outputs])
                    np.testing.assert_array_equal(states_time,times)
                    symmetry=float(np.max(np.abs(cov-np.swapaxes(cov,-1,-2))))
                    min_eig=float(np.linalg.eigvalsh(cov).min())
                    consistency.append(dict(uid=uid,group=meta['group'],dataset=dataset,profile=profile,backend=backend_name,frames=n,
                        velocity_max_difference_mps=float(delta.max()),covariance_min_eigenvalue=min_eig,
                        covariance_max_asymmetry=symmetry,state_output_time_max_difference_s=float(np.max(np.abs(states_time-times))),
                        direct_arrivals=int(direct.sum()),prediction_only=int((initialized&~direct).sum()),uninitialized=int((~initialized).sum()),
                        output_coverage=1.,max_measurement_age_s=float(np.nanmax(age)) if np.isfinite(age).any() else 0.))
                    for lo,hi in ((0,100),(100,300),(300,n)):
                        hi=min(hi,n)
                        if hi>lo:runtime.append(dict(uid=uid,dataset=dataset,profile=profile,backend=backend_name,frame_start=lo,frame_end=hi,
                            n=hi-lo,median_ms=float(np.median(latency[lo:hi])),p95_ms=float(np.quantile(latency[lo:hi],.95))))
                    events={'direct_arrival':direct&initialized,'prediction_only':initialized&~direct,
                            'uninitialized':~initialized,'all_outputs':np.ones(n,bool)}
                    for horizon in HORIZONS:
                        endpoint=np.searchsorted(times,times+horizon-1e-8)
                        endpoint_clip=np.minimum(endpoint,n-1)
                        exists=(endpoint<n)&(np.abs(times[endpoint_clip]-times-horizon)<1e-7)
                        dt=times[endpoint_clip]-times
                        posterior=mean[:,:3]+dt[:,None]*mean[:,3:]
                        raw_forecast=raw+(age+dt)[:,None]*mean[:,3:]
                        target=truth_local[endpoint_clip]
                        post_error=posterior-target;raw_error=raw_forecast-target
                        p_forecast=cov[:,:3,:3]+dt[:,None,None]*(cov[:,:3,3:]+cov[:,3:,:3])+dt[:,None,None]**2*cov[:,3:,3:]+np.array([np.diag(q)*h**3/3 for q,h in zip(noise,dt)])
                        pos_uncertainty=uncertainty_arrays(post_error,p_forecast,args.device)
                        for filt,fmask in (('all_frames',score),('ground',score&ground)):
                            for event,emask in events.items():
                                chosen=exists&fmask&emask
                                if not chosen.any():continue
                                meta_key=(dataset,meta['scenario'],profile,backend_name,filt,event,horizon)
                                post_norm=np.linalg.norm(post_error[chosen],axis=1)
                                key=meta_key+('posterior_position_with_same_velocity',)
                                forecast_pool[key].append(post_norm)
                                group_pool[(meta['group'],)+key].append(post_norm)
                                episodes.append(dict(uid=uid,group=meta['group'],block=meta['block'],**dict(zip(FIELDS,key)),**metrics(post_norm)))
                                uncertainty_pool[meta_key+('forecast_position',)].append({k:v[chosen] for k,v in pos_uncertainty.items()})
                                if event=='direct_arrival':
                                    if not np.isfinite(raw_forecast[chosen]).all():raise AssertionError('Direct paired baseline unavailable')
                                    raw_norm=np.linalg.norm(raw_error[chosen],axis=1)
                                    rawkey=meta_key+('raw_position_with_same_velocity',)
                                    forecast_pool[rawkey].append(raw_norm)
                                    group_pool[(meta['group'],)+rawkey].append(raw_norm)
                                    episodes.append(dict(uid=uid,group=meta['group'],block=meta['block'],**dict(zip(FIELDS,rawkey)),**metrics(raw_norm)))
                                    bootstrap_pool[meta_key][meta['block']].append((raw_norm,post_norm))
                        if horizon==0:
                            for kind,err,mat in (('current_velocity',mean[:,3:]-true_velocity_local,cov[:,3:,3:]),
                                                ('current_joint_state',mean-np.c_[truth_local,true_velocity_local],cov)):
                                values=uncertainty_arrays(err,mat,args.device)
                                for filt,fmask in (('all_frames',score),('ground',score&ground)):
                                    for event,emask in events.items():
                                        chosen=fmask&emask
                                        if chosen.any():uncertainty_pool[(dataset,meta['scenario'],profile,backend_name,filt,event,horizon,kind)].append({k:v[chosen] for k,v in values.items()})
                    stored.append(dict(uid=uid,group=meta['group'],dataset=dataset,profile=profile,backend=backend_name,
                        times=times,state_local=mean,covariance_local=cov,process_intensity_local=noise,
                        velocity_base_mps=prediction,last_observation_local_m=raw,measurement_age_s=age,
                        direct_update=direct,initialized=initialized,score=score,ground=ground,
                        heading_local_rad=np.array([o['heading_local_rad'] for o in outputs]),
                        base_position_local_m=np.array([o['base_position_local_m'] for o in outputs])))
            manifest_out.append(dict(**meta,dataset=dataset));stream_count+=1
            print(f'Completed interface {stream_count}: {dataset}/{meta["group"]} {uid}; elapsed={time.monotonic()-wall:.1f}s',flush=True)
    summary=[dict(zip(FIELDS,k),**metrics(np.concatenate(v))) for k,v in forecast_pool.items()]
    grouped=[dict(zip(('group',)+FIELDS,k),**metrics(np.concatenate(v))) for k,v in group_pool.items()]
    unc=[]
    for key,parts in uncertainty_pool.items():
        joined={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        unc.append(dict(zip(FIELDS[:-1]+('kind',),key),n=len(joined['nees']),
            nees_mean=float(np.mean(joined['nees'])),nees_p95=float(np.quantile(joined['nees'],.95)),
            ellipsoid_coverage95=float(np.mean(joined['coverage95'])),axis_coverage95=float(np.mean(joined['axis_coverage95'])),
            mean_ellipsoid_volume=float(np.mean(joined['volume'])),min_covariance_eigenvalue=float(np.min(joined['min_eigenvalue']))))
    intervals=[];rng=np.random.default_rng(26100792)
    for key,blocks in bootstrap_pool.items():
        if key[2] not in ('sync','combined'):continue
        pairs=[(np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts])) for parts in blocks.values()]
        means=[];p95=[]
        for _ in range(500):
            idx=rng.integers(len(pairs),size=len(pairs));a=np.concatenate([pairs[i][0] for i in idx]);b=np.concatenate([pairs[i][1] for i in idx])
            means.append(b.mean()-a.mean());p95.append(np.quantile(b,.95)-np.quantile(a,.95))
        for metric,values in (('mean',means),('p95',p95)):
            intervals.append(dict(zip(FIELDS[:-1],key),metric=metric,blocks=len(pairs),
                delta_ci_low=float(np.quantile(values,.025)),delta_ci_high=float(np.quantile(values,.975)),
                bootstrap_unit='collection seed x environment; posterior minus raw; frozen model'))
    write_csv(args.out/'forecast_metrics.csv',summary);write_csv(args.out/'forecast_episodes.csv',episodes)
    write_csv(args.out/'grouped_forecast_metrics.csv',grouped)
    write_csv(args.out/'uncertainty_metrics.csv',unc);write_csv(args.out/'paired_bootstrap.csv',intervals)
    write_csv(args.out/'interface_validation.csv',consistency);write_csv(args.out/'runtime_diagnostic.csv',runtime)
    torch.save(stored,args.out/'state_outputs.pt')
    (args.out/'data_manifest.json').write_text(json.dumps(manifest_out,indent=2))
    (args.out/'execution.json').write_text(json.dumps(dict(torch_version=torch.__version__,device=args.device,
        gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None,
        elapsed_s=time.monotonic()-wall,streams=stream_count,output_groups=len(stored)),indent=2))
    for path,expected in protocol['source_hashes'].items():
        if digest(path)!=expected:raise ValueError('Scored source changed while evaluation was running')
    (args.out/'COMPLETE').write_text('Frozen N6 interface/forecast diagnostic complete.\n')
    print('N6 interface evaluation complete:',args.out,flush=True)


if __name__=='__main__':main()
