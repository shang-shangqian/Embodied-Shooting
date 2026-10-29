"""Evaluate N0 frozen filters on paired real-vision acquisition trajectories."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from n0_time_replay import (NoiseBackend, PROFILES, TimeContractReplay, delivery_schedule,
                            feature_history, replay)
from optimization_components import LearnedCVFilter
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU
from speed_prediction.sensor_time_contract import position_from_pair, TIME_CONTRACT_VERSION

BASE = Path('source/standalone/test11/velocity_error_validation')
CHECKPOINT = BASE/'reports/optimization_flow_runs_20261007/learned_filter_seed41/best_model.pt'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(errors):
    return dict(n=len(errors), mean=float(np.mean(errors)),
                p95=float(np.quantile(errors, .95)), p99=float(np.quantile(errors, .99)),
                over2_rate=float(np.mean(errors > 2)))


def clock_only_replay(packets, profile, seed, backend):
    queue, _, schedule = delivery_schedule(packets, profile, seed)
    times = np.array([p['capture_time_s'] for p in packets])
    engine = TimeContractReplay(packets, backend, corrected=False)
    predictions = []
    for i, p in enumerate(packets):
        engine.current_packet = p
        velocity, _ = engine.update(i, queue.arrived(times[i]), times)
        predictions.append(velocity.copy())
    return dict(prediction=np.array(predictions), schedule=schedule, stats=engine_stats(engine))


def engine_stats(engine):
    return dict(expired_packets=engine.expired, paired_measurements=engine.paired,
                unique_pairs=len(engine.unique_updates), replayed_steps=engine.replayed_steps,
                max_replay_age_s=engine.max_replay_age_s, sensor_clock_elapsed_error_s=0.)


@torch.inference_mode()
def validate_frozen_equations(packets, model, corrected, device):
    positions = np.array([position_from_pair(p, p) if p['range_valid'] and p['source_id']<4 else [np.nan]*3 for p in packets])
    linear = np.array([p['linear_velocity_base_mps'] for p in packets])
    angular = np.array([p['angular_velocity_base_rps'] for p in packets])
    sources = np.array([p['source_id'] for p in packets])
    quality = np.array([p['quality'] for p in packets])
    times = np.array([p['capture_time_s'] for p in packets])
    x, extra = feature_history(positions, linear, angular, sources, quality, times)
    p, _, _ = model.forward_segment(torch.tensor(x, device=device)[None],
        torch.tensor(sources, device=device)[None], torch.tensor(extra, device=device)[None],
        torch.ones(1, len(x), dtype=torch.bool, device=device))
    difference = np.linalg.norm(p[0].cpu().numpy()-corrected['prediction'], axis=1)
    maximum = float(difference.max())
    if maximum > .005:
        raise AssertionError(f'Frozen filter numerical mismatch: {maximum}')
    return dict(frames=len(x), max_vector_difference_mps=maximum,
                mean_vector_difference_mps=float(difference.mean()), tolerance_mps=.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=BASE/'reports/n0_time_contract_20261007')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--actual-sparse', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'delivery_schedule.jsonl').write_text('')
    cp = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = LearnedCVFilter(TimeAwareQualityGatedGRU(**cp['original_config'])).to(args.device)
    model.load_state_dict(cp['state_dict']); model.eval()
    backends = {'fixed_cv':NoiseBackend(q=.1, r=.1),
                'frozen_learned_filter':NoiseBackend(model, args.device)}
    protocol = dict(version=TIME_CONTRACT_VERSION, checkpoint=str(CHECKPOINT),
                    checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    fixed_cv_q=.1, fixed_cv_r=.1, replay_lag_s=.20, seed_base=26110780,
                    profiles=list(PROFILES), candidate_training=False, parameter_search=False,
                    timing_clock='shared monotonic capture reference; device reported clock isolated',
                    sampling_scope='packet thinning after genuine clean LK/OSTrack; tracker itself not rerun on sparse images',
                    latency_scope='simulated result/transport delivery, not camera/frontend wall-clock latency',
                    timestamp_noise_scope='legacy cumulative device time distortion; corrected uses recorded acquisition reference, not estimated unknown clock skew',
                    control_scope='arrival-time composition on identical acquired measurements; not a full rerun of legacy delayed-image tracking',
                    filters='unchanged frozen learned Q/R and preselected fixed q=.1,r=.1',
                    scores='existing label burn-in & first raw direct-valid observation; same frames for all methods, delayed startup not excluded')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    paths = sorted(args.data.rglob('timed_sensor_packets.pt'))
    paths = [p for p in paths if '_incomplete_capture_archive' not in str(p)
             and ('smoke_' in str(p)) == args.smoke
             and ('sd_sparse_' in str(p)) == args.actual_sparse]
    if not paths:
        raise ValueError('No completed acquisition traces')
    if not args.smoke and len(list(args.data.glob('*/COLLECTION_COMPLETE'))) < 7:
        raise ValueError('Six formal groups plus smoke marker required')
    profiles=('sync',) if args.actual_sparse else PROFILES
    paired_capture_audit=[]
    if args.actual_sparse:
        protocol['profiles']=['actual_sparse_tracking_sync_transport']
        protocol['sampling_scope']='actual camera consumption skipped BEFORE LK/OSTrack; physical simulation continues every 20 ms'
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    pooled = defaultdict(list); episodes = []; stored = []; manifest = []; audits = []; equations = []
    bootstrap_inputs = defaultdict(lambda:defaultdict(dict)); start = time.monotonic()
    for stream_index, path in enumerate(paths):
        if not args.smoke and not (path.parents[3]/'capture_validation.json').exists():
            raise ValueError('Acquisition frame-count validation missing')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        packets = payload['packets']; uid = payload['episode_uid']
        stream = torch.load(path.with_name('taq_velocity_episode.taqstream'), map_location='cpu', weights_only=False)
        audit = [json.loads(row) for row in path.with_name('velocity_error_audit.jsonl').read_text().splitlines()]
        if payload['version'] != TIME_CONTRACT_VERSION or len(packets) != len(audit):
            raise ValueError('Incorrect acquisition contract')
        forbidden = {'target_velocity_base','target_position_world_m','base_position_world_m','label','ground'}
        assert not any(forbidden.intersection(p) for p in packets)
        times = np.array([p['capture_time_s'] for p in packets])
        np.testing.assert_allclose(times, stream['timestamps_s'].numpy(), atol=1e-10)
        np.testing.assert_allclose(times, [r['physics_timestamp_s'] for r in audit], atol=1e-10)
        truth = stream['target_velocity_base'].numpy()
        mask = stream['target_valid_mask'].numpy().copy()
        direct = np.array([p['source_id']<=2 and p['range_valid'] for p in packets])
        first = np.flatnonzero(direct)
        if len(first):
            mask &= times >= times[first[0]]
            startup = (times-times[first[0]]>=-1e-9) & (times-times[first[0]]<.5-1e-8)
        else:
            mask[:] = False; startup = np.zeros(len(mask), bool)
        ground = np.array([r['target_position_world_m'][2]>=0 for r in audit])
        folder = path.parents[3].name
        group = folder.rsplit('_', 1)[0]; scenario = stream['metadata']['scenario']
        block = folder+':'+path.parents[1].name
        seed = int(stream['metadata']['environment_seed'])+10000
        # Env/episode seed changes degradation only; raw physical trace unchanged.
        replay_seed = seed+100*int(path.parents[1].name.split('_')[1])+int(path.parent.name.split('_')[1])
        if args.actual_sparse:
            dense_path=args.data/folder.replace('sd_sparse_','sd_',1)/path.relative_to(path.parents[3])
            dense_stream=torch.load(dense_path.with_name('taq_velocity_episode.taqstream'),map_location='cpu',weights_only=False)
            dense_packets=torch.load(dense_path,map_location='cpu',weights_only=False)['packets']
            dense_audit=[json.loads(row) for row in dense_path.with_name('velocity_error_audit.jsonl').read_text().splitlines()]
            if len(dense_packets)!=len(packets):raise ValueError('Sparse/dense physics stream lengths differ')
            position_delta=np.linalg.norm(np.array([r['target_position_world_m'] for r in audit])-np.array([r['target_position_world_m'] for r in dense_audit]),axis=1)
            label_delta=np.linalg.norm(truth-dense_stream['target_velocity_base'].numpy(),axis=1)
            dense_first=np.flatnonzero([p['source_id']<=2 and p['range_valid'] for p in dense_packets])
            # Include sparse startup from the same dense reference start.
            mask=dense_stream['target_valid_mask'].numpy().copy()
            if len(dense_first):mask &= times>=times[dense_first[0]]
            else:mask[:]=False
            if len(dense_first):startup=(times-times[dense_first[0]]>=-1e-9)&(times-times[dense_first[0]]<.5-1e-8)
            paired_capture_audit.append(dict(uid=uid,frames=len(packets),
                target_trajectory_max_difference_m=float(position_delta.max()),
                label_max_difference_mps=float(label_delta.max()),
                physical_time_max_difference_s=float(np.max(np.abs(times-dense_stream['timestamps_s'].numpy()))),
                skipped_images=sum(not p.get('image_sampled',True) for p in packets),
                max_tracker_input_dt_s=max([p['tracker_input_dt_s'] for p in packets if p.get('tracker_input_dt_s') is not None],default=0.),
                trajectory_matches_1mm=bool(position_delta.max()<.001)))
        reconstruction = []
        for p, row in zip(packets, audit):
            if p['source_id']<4 and p['range_valid']:
                reconstruction.append(np.linalg.norm(position_from_pair(p,p)-row['position_observed_base_m']))
        manifest.append(dict(uid=uid, group=group, scenario=scenario, block=block, seed=seed-10000,
                             packet_file=str(path), packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             stream_sha256=hashlib.sha256(path.with_name('taq_velocity_episode.taqstream').read_bytes()).hexdigest(),
                             audit_sha256=hashlib.sha256(path.with_name('velocity_error_audit.jsonl').read_bytes()).hexdigest(),
                             frames=len(packets), score_frames=int(mask.sum()), ground_frames=int((mask&ground).sum()),
                             reconstruction_max_m=max(reconstruction,default=0.),
                             yaw_rate_abs_p95=float(np.quantile(np.abs([p['angular_velocity_base_rps'][2] for p in packets]),.95))))
        for profile in profiles:
            for name, backend in backends.items():
                corrected = replay(packets, profile, replay_seed, backend, corrected=True)
                legacy = replay(packets, profile, replay_seed, backend, corrected=False)
                clock_only = clock_only_replay(packets, profile, replay_seed, backend) if profile=='combined' else None
                if name=='frozen_learned_filter' and profile=='sync':
                    equations.append(dict(uid=uid,**validate_frozen_equations(packets, model, corrected, args.device)))
                results={'arrival_time_control':legacy, 'sample_time_replay':corrected}
                if clock_only is not None:
                    results['physical_clock_only']=clock_only
                for method, result in results.items():
                    error = np.linalg.norm(result['prediction']-truth, axis=1)
                    for filt, valid in [('all_frames',mask),('ground',mask&ground)]:
                        for event, subset in [('all',np.ones(len(mask),bool)),('startup_0p5s',startup),('after_startup',~startup)]:
                            chosen = valid&subset
                            if not chosen.any():
                                continue
                            row = dict(group=group, scenario=scenario, profile=profile, backend=name,
                                       method=method, filter=filt, event=event)
                            pooled[tuple(row.values())].append(error[chosen])
                            episodes.append(dict(uid=uid, block=block, **row, **metrics(error[chosen])))
                            if event=='all':
                                key=(scenario,profile,name,filt)
                                bootstrap_inputs[key][block].setdefault(method,[]).append(error[chosen])
                    audits.append(dict(uid=uid, profile=profile, backend=name, method=method, **result['stats']))
                    stored.append(dict(uid=uid, profile=profile, backend=name, method=method,
                                       prediction=result['prediction'], error=error, score=mask, ground=ground))
                if name=='fixed_cv':
                    # One transport audit is sufficient, shared by both backends.
                    with (args.out/'delivery_schedule.jsonl').open('a') as handle:
                        for event in corrected['schedule']:
                            handle.write(json.dumps(dict(uid=uid,profile=profile,**event))+'\n')
        print(f'Completed replay {stream_index+1}/{len(paths)} {group} {path.parents[1].name}/{path.parent.name}; elapsed={time.monotonic()-start:.1f}s', flush=True)
    summary = [dict(zip(['group','scenario','profile','backend','method','filter','event'],key),
                    **metrics(np.concatenate(parts))) for key,parts in pooled.items()]
    combined = defaultdict(list)
    for item in stored:
        meta=next(row for row in manifest if row['uid']==item['uid'])
        for filt, chosen in [('all_frames',item['score']),('ground',item['score']&item['ground'])]:
            if chosen.any():
                combined[(meta['scenario'],item['profile'],item['backend'],item['method'],filt)].append(item['error'][chosen])
    scenario_rows = [dict(zip(['scenario','profile','backend','method','filter'],key),
                          **metrics(np.concatenate(parts))) for key,parts in combined.items()]
    intervals = []; rng = np.random.default_rng(26100787)
    for key,blocks in bootstrap_inputs.items():
        block_errors=[]
        for block,methods in blocks.items():
            if 'sample_time_replay' in methods and 'arrival_time_control' in methods:
                block_errors.append((np.concatenate(methods['arrival_time_control']),np.concatenate(methods['sample_time_replay'])))
        if not block_errors:
            continue
        differences = {'mean':[], 'p95':[]}
        for _ in range(500):
            idx=rng.integers(len(block_errors),size=len(block_errors))
            a=np.concatenate([block_errors[i][0] for i in idx]);b=np.concatenate([block_errors[i][1] for i in idx])
            differences['mean'].append(b.mean()-a.mean());differences['p95'].append(np.quantile(b,.95)-np.quantile(a,.95))
        for metric,values in differences.items():
            intervals.append(dict(zip(['scenario','profile','backend','filter'],key),metric=metric,
                                  blocks=len(block_errors),delta_ci_low=float(np.quantile(values,.025)),
                                  delta_ci_high=float(np.quantile(values,.975)),bootstrap_unit='collection seed x environment'))
    write_csv(args.out/'grouped_metrics.csv',summary);write_csv(args.out/'scenario_metrics.csv',scenario_rows)
    write_csv(args.out/'episode_metrics.csv',episodes);write_csv(args.out/'paired_bootstrap.csv',intervals)
    write_csv(args.out/'timing_audit.csv',audits);write_csv(args.out/'frozen_equation_validation.csv',equations)
    write_csv(args.out/'actual_sampling_audit.csv',paired_capture_audit)
    (args.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.save(stored,args.out/'predictions.pt')
    (args.out/'COMPLETE').write_text('N0 frozen paired delivery evaluation complete.\n')
    print('N0 evaluation complete:',args.out,flush=True)


if __name__=='__main__':
    main()
