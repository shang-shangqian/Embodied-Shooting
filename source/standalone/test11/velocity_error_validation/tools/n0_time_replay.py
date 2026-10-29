"""Causal paired sensor-delivery replay with frozen CV / learned Q-R backends.

No labels or GT audit accepted by the estimator. Historical outputs are never
rewritten. Current output may reprocess only packets which have already arrived.
This is a reference replay, not a production latency benchmark.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speed_prediction.sensor_time_contract import (
    Delivery, DeliveryQueue, MatchedMeasurements, position_from_pair, yaw_rotation,
)
from optimization_components import fit_baseline, propagated_inputs, rotate, WIDTHS
from speed_prediction.filter_state_output import filter_state_output

PROFILES = ('sync', 'irregular_sampling', 'clock_noise', 'image_delay',
            'range_delay', 'proprio_delay', 'combined')


def delivery_schedule(packets, profile, seed):
    """Same acquisition trajectory, independently specified clock and transport.

Irregular sampling here thins acquired tracker-result/range packets. It does
not claim to rerun LK/OSTrack with a different image sampling cadence.
"""
    if profile not in PROFILES:
        raise ValueError(profile)
    rng = np.random.default_rng(seed)
    queue = DeliveryQueue()
    clock = []
    clock_now = packets[0]['capture_time_s']
    skip_until = -1
    rows = []
    for i, packet in enumerate(packets):
        now = packet['capture_time_s']
        dt = 0 if i == 0 else now-packets[i-1]['capture_time_s']
        noisy = profile in ('clock_noise', 'combined')
        jitter = rng.normal(0, .005) if noisy and i else 0.
        gap = int(rng.integers(1, 5)) if noisy and i and rng.random() < .01 else 0
        # Exactly the old style cumulative synthetic dt, isolated from sampling.
        if i:
            clock_now += max(.75*dt, (1+gap)*dt+jitter)
        clock.append(clock_now)
        drop = False
        if profile in ('irregular_sampling', 'combined'):
            if i >= skip_until and rng.random() < (.08 if profile == 'irregular_sampling' else .02):
                skip_until = i+int(rng.integers(2, 7))
            drop = i < skip_until
        delay = dict(direction=0., range=0., proprio=0.)
        if profile in ('image_delay', 'combined'):
            delay['direction'] = .04+float(rng.uniform(0, .025))
        if profile in ('range_delay', 'combined'):
            delay['range'] = .02+float(rng.uniform(0, .025))
        if profile in ('proprio_delay', 'combined'):
            delay['proprio'] = .04+float(rng.uniform(0, .015))
        for kind in ('direction', 'range', 'proprio'):
            if kind == 'direction' and not packet.get('image_sampled', True):
                continue
            if kind == 'range' and not packet.get('range_sampled', True):
                continue
            if drop and kind != 'proprio':
                continue
            event = Delivery(kind, packet['sample_id'], now, now+delay[kind], packet)
            queue.push(event)
            rows.append(dict(kind=kind, sample_id=packet['sample_id'],
                             sample_time_s=now, arrival_time_s=event.arrival_time_s))
    return queue, np.asarray(clock), rows


def feature_history(positions, linear, angular, sources, qualities, times):
    """Vectorized equivalent of the existing 24-feature capture contract."""
    n = len(times)
    latest = np.maximum.accumulate(np.where(np.isfinite(linear[:, 0]), np.arange(n), -1))
    lin = np.where((latest >= 0)[:, None], linear[latest.clip(0)], 0.)
    ang = np.where((latest >= 0)[:, None], angular[latest.clip(0)], 0.)
    lin = np.nan_to_num(lin); ang = np.nan_to_num(ang)
    valid = np.isfinite(positions).all(1)
    clean = np.nan_to_num(positions).astype(np.float32)
    dt = np.diff(times, prepend=times[0])
    fd = np.zeros_like(clean)
    if n > 1:
        pair = valid[1:] & valid[:-1] & (dt[1:] > 1e-6)
        fd[1:][pair] = (clean[1:][pair].astype(float)-clean[:-1][pair])/dt[1:][pair, None]
    direct = (sources <= 2) & valid
    last = np.maximum.accumulate(np.where(direct, np.arange(n), -1))
    fine = np.maximum.accumulate(np.where((sources == 2) & valid, np.arange(n), -1))
    age = np.where(last >= 0, times-times[last.clip(0)], 0).clip(0, 1)
    fine_age = np.where(fine >= 0, times-times[fine.clip(0)], 0).clip(0, 1)
    x = np.column_stack((clean, fd, lin, ang, dt, age, fine_age,
                         qualities, direct, valid)).astype(np.float32)
    x, resets = propagated_inputs(x, sources, times)
    extras = []
    for width in WIDTHS:
        base, extra = fit_baseline(x, times, resets, width, aligned=True, fd_weight=False)
        for i in range(1, n):
            if x[i, 23] < .5:
                base[i] = rotate(base[i-1]+x[i-1, 6:9], np.array(-x[i, 11]*dt[i]))-x[i, 6:9]
        extra[:, :3] = base/10
        extras.append(extra)
    return x, np.concatenate(extras, axis=1)


def initial_state():
    return (np.zeros(6), np.eye(6)*25, 0., np.zeros(3), np.zeros(3), 0., False)


def fit_tail(x, times, start, width, fd_weight=False):
    """Exact aligned linear-fit equations, evaluating only changed timestamps."""
    n = len(x); k=min(n,int(np.ceil(width/max(np.min(np.diff(times)) if n>1 else .02,.001)))+3)
    current=np.arange(start,n); indices=current[:,None]-np.arange(k)[None,:]
    index=indices.clip(0);tau=times[index]-times[current,None]
    causal=(indices>=0)&(tau>=-width-1e-9)&(tau<=0)
    weight=x[:,23]*(.05+.95*x[:,22])*(.1+.9*x[:,15])
    if fd_weight:weight*=np.minimum(1,30/np.maximum(np.linalg.norm(x[:,3:6],axis=1),1e-6))
    w=weight[index]*causal;dt=np.diff(times,prepend=times[0])
    yaw=np.cumsum(.5*(x[:,11]+np.r_[x[0,11],x[:-1,11]])*dt)
    world_v=rotate(x[:,6:9],yaw)
    base=np.cumsum(.5*(world_v+np.vstack((world_v[0],world_v[:-1])))*dt[:,None],axis=0)
    pos=rotate(x[:,:3].astype(float),yaw)+base;p=pos[index]
    count=(w>0).sum(1);sw=np.maximum(w.sum(1),1e-12)
    mt=(w*tau).sum(1)/sw;mp=(w[:,:,None]*p).sum(1)/sw[:,None]
    centered=tau-mt[:,None];variance=(w*centered**2).sum(1)
    slope=(w[:,:,None]*centered[:,:,None]*(p-mp[:,None,:])).sum(1)/np.maximum(variance[:,None],1e-6)
    slope[count<2]=0;fitted=mp[:,None,:]+centered[:,:,None]*slope[:,None,:]
    rmse=np.sqrt((w*np.linalg.norm(p-fitted,axis=-1)**2).sum(1)/sw)
    span=np.maximum(0,-np.min(np.where(w>0,tau,0),axis=1))
    velocity=rotate(slope,-yaw[current])-x[current,6:9];velocity[count<2]=0
    last=np.maximum.accumulate(np.where((x[:,22]>.5)&(x[:,23]>.5),np.arange(n),-1))
    age=times[current]-times[last[current].clip(0)]
    extra=np.column_stack((velocity/10,span/width,count/25,rmse,variance/np.maximum(sw*width**2,1e-9),age))
    return velocity.astype(np.float32),np.nan_to_num(extra).clip(-20,20).astype(np.float32)


class FeatureCache:
    """Cache past propagated-input states; redo only history touched by arrivals."""
    def __init__(self):
        self.soft=np.empty((0,24),np.float32);self.states=[];self.baselines=[]

    def build(self,positions,linear,angular,sources,qualities,times,start):
        n=len(times)
        latest=np.maximum.accumulate(np.where(np.isfinite(linear[:,0]),np.arange(n),-1))
        lin=np.nan_to_num(np.where((latest>=0)[:,None],linear[latest.clip(0)],0.))
        ang=np.nan_to_num(np.where((latest>=0)[:,None],angular[latest.clip(0)],0.))
        valid=np.isfinite(positions).all(1);clean=np.nan_to_num(positions).astype(np.float32)
        dt=np.diff(times,prepend=times[0]);fd=np.zeros_like(clean)
        if n>1:
            pair=valid[1:]&valid[:-1]&(dt[1:]>1e-6)
            fd[1:][pair]=(clean[1:][pair].astype(float)-clean[:-1][pair])/dt[1:][pair,None]
        direct=(sources<=2)&valid
        last=np.maximum.accumulate(np.where(direct,np.arange(n),-1))
        fine=np.maximum.accumulate(np.where((sources==2)&valid,np.arange(n),-1))
        age=np.where(last>=0,times-times[last.clip(0)],0).clip(0,1)
        fine_age=np.where(fine>=0,times-times[fine.clip(0)],0).clip(0,1)
        raw=np.column_stack((clean,fd,lin,ang,dt,age,fine_age,qualities,direct,valid)).astype(np.float32)
        prior,_=fit_tail(raw,times,start,.48,fd_weight=True)
        soft=raw.copy();soft[:start]=self.soft[:start]
        active,p,v,previous_robot=(False,np.zeros(3),np.zeros(3),np.zeros(3)) if start==0 else self.states[start-1]
        p=p.copy();v=v.copy();previous_robot=previous_robot.copy()
        del self.states[start:]
        for i in range(start,n):
            if raw[i,23]>.5 and raw[i,22]>.5:
                p=raw[i,:3].copy();v=prior[i-start].copy();active=True
            elif active:
                turn=raw[i,11]*dt[i]
                v=rotate(v+previous_robot,np.array(-turn))-raw[i,6:9]
                p=rotate(p,np.array(-turn))+v*dt[i]
                if raw[i,23]<.5:soft[i,:3]=p;soft[i,3:6]=0
            previous_robot=raw[i,6:9].copy()
            self.states.append((active,p.copy(),v.copy(),previous_robot.copy()))
        extra=[];new_baselines=[]
        for scale,width in enumerate(WIDTHS):
            tail,e=fit_tail(soft,times,start,width)
            previous=np.empty((0,3),np.float32) if start==0 else self.baselines[scale][:start]
            base=np.concatenate((previous,tail))
            for i in range(max(1,start),n):
                if soft[i,23]<.5:
                    base[i]=rotate(base[i-1]+soft[i-1,6:9],np.array(-soft[i,11]*dt[i]))-soft[i,6:9]
            e[:,:3]=base[start:]/10
            extra.append(e);new_baselines.append(base)
        self.soft=soft;self.baselines=new_baselines
        return soft[start:],np.concatenate(extra,axis=1)


def cv_step(state, x, noise):
    """Same CV/Joseph equations; physical dt is not silently truncated."""
    mean, cov, yaw, base, previous_robot, previous_omega, started = state
    mean = mean.copy(); cov = cov.copy(); base = base.copy()
    dt = float(x[12])
    if dt < 0 or not np.isfinite(dt):
        raise ValueError('Invalid physical elapsed time')
    omega = float(x[11]); yaw += .5*(omega+previous_omega)*dt
    rot = yaw_rotation(yaw)
    robot = rot@x[6:9]
    base += .5*(robot+previous_robot)*dt
    transition = np.eye(6); transition[:3, 3:] = np.eye(3)*dt
    q = np.diag(noise[:3]); process = np.block([[q*dt**3/3, q*dt**2/2],
                                                              [q*dt**2/2, q*dt]])
    mean = transition@mean; cov = transition@cov@transition.T+process
    if x[22] > .5 and x[23] > .5:
        position = rot@x[:3]+base
        if not started:
            mean[:3] = position
        innovation = position-mean[:3]
        measurement = np.diag(noise[3:])
        innovation_cov = cov[:3, :3]+measurement+np.eye(3)*1e-6
        gain = np.linalg.solve(innovation_cov, cov[:3, :]).T
        mean += gain@innovation
        joseph = np.eye(6); joseph[:, :3] -= gain
        cov = joseph@cov@joseph.T+gain@measurement@gain.T
        started = True
    velocity = rot.T@mean[3:]-x[6:9]
    return (mean, cov, yaw, base, robot, omega, started), velocity


class NoiseBackend:
    def __init__(self, model=None, device='cpu', q=.1, r=.1):
        self.model = model
        self.device = torch.device(device)
        self.q = q; self.r = r
        if model is not None:
            self.model.eval()

    @torch.inference_mode()
    def values(self, x, sources, extra):
        if self.model is None:
            return np.tile([self.q]*3+[self.r]*3, (len(x), 1))
        features = torch.as_tensor(x, device=self.device)
        src = torch.as_tensor(sources, dtype=torch.long, device=self.device)
        ex = torch.as_tensor(extra, device=self.device)
        normalized, _ = self.model.normalizer._prepare_input(features)
        noise = torch.nn.functional.softplus(self.model.noise(torch.cat(
            (normalized, self.model.embedding(src), ex), -1)))+1e-4
        return noise.cpu().numpy()


class TimeContractReplay:
    """Reference event consumer; measurements are revised within a fixed lag.

Arrays/history are retained for audit; production bounded-cache work is N6.
Only replay depth is bounded here. Expired observations never update old state.
"""
    def __init__(self, packets, backend, corrected=True, lag_s=.20):
        self.backend = backend; self.corrected = corrected; self.lag_s = lag_s
        self.times = np.asarray([p['capture_time_s'] for p in packets])
        self.ids = {p['sample_id']:i for i, p in enumerate(packets)}
        n = len(packets)
        self.positions = np.full((n, 3), np.nan)
        self.linear = np.full((n, 3), np.nan); self.angular = self.linear.copy()
        self.sources = np.full(n, 4, int); self.quality = np.zeros((n, 7))
        self.states = []; self.matcher = MatchedMeasurements(); self.latest = {}
        self.expired = 0; self.paired = 0; self.replayed_steps = 0
        self.max_replay_age_s = 0.; self.unique_updates = set()
        self.feature_cache = FeatureCache()

    def update(self, i, events, legacy_times):
        now = self.times[i]; dirty = i
        self.direct_update = False
        for event in events:
            if event.arrival_time_s > now+1e-9:
                raise ValueError('Future arrival entered current estimate')
            if event.sample_id not in self.ids:
                raise ValueError('Unregistered sampling tick; provide a clock/grid adapter')
            sample = self.ids[event.sample_id]
            if abs(self.times[sample]-event.sample_time_s)>1e-9 or event.value['sample_id']!=event.sample_id or abs(event.value['capture_time_s']-event.sample_time_s)>1e-9:
                raise ValueError('Capture identity or sampling clock mismatch')
            if sample > i:
                raise ValueError('Future sample entered current estimate')
            if self.corrected:
                if now-event.sample_time_s > self.lag_s+1e-9:
                    self.expired += 1
                    continue
                dirty = min(dirty, sample)
                self.max_replay_age_s = max(self.max_replay_age_s, now-event.sample_time_s)
                if event.kind == 'proprio':
                    self.linear[sample] = event.value['linear_velocity_base_mps']
                    self.angular[sample] = event.value['angular_velocity_base_rps']
                else:
                    pair = self.matcher.receive(event)
                    if pair is not None:
                        direction = pair['direction']
                        self.sources[sample] = direction['source_id']
                        self.quality[sample] = direction['quality']
                        if pair['position_base_m'] is not None:
                            self.positions[sample] = pair['position_base_m']
                            self.direct_update |= direction['source_id'] <= 2
                        self.paired += 1
                        self.unique_updates.add(sample)
            else:
                # Keep newest sample, not last arrival, so control also tolerates
                # arrival reordering. It still treats components as current.
                old = self.latest.get(event.kind)
                if old is None or old.sample_id < event.sample_id:
                    self.latest[event.kind] = event
        if self.corrected:
            self.expired += self.matcher.expire(now-self.lag_s)
            times = self.times[:i+1]
        else:
            direction = self.latest.get('direction'); distance = self.latest.get('range')
            proprio = self.latest.get('proprio')
            if direction is not None:
                self.sources[i] = direction.value['source_id']
                self.quality[i] = direction.value['quality']
            if direction is not None and distance is not None and distance.value['range_valid']:
                # Arrival-time fusion uses current camera calibration/orientation.
                current = self.current_packet
                wrong_direction = dict(direction.value, sample_id=distance.sample_id,
                                       capture_time_s=distance.sample_time_s,
                                       camera_rotation_base=current['camera_rotation_base'],
                                       camera_origin_base=current['camera_origin_base'])
                self.positions[i] = position_from_pair(wrong_direction, distance.value)
                self.sources[i] = direction.value['source_id']
                self.quality[i] = direction.value['quality']
            if proprio is not None:
                self.linear[i] = proprio.value['linear_velocity_base_mps']
                self.angular[i] = proprio.value['angular_velocity_base_rps']
            times = legacy_times[:i+1]
        x, extra = self.feature_cache.build(self.positions[:i+1], self.linear[:i+1], self.angular[:i+1],
                                          self.sources[:i+1], self.quality[:i+1], times, dirty)
        noise = self.backend.values(x, self.sources[dirty:i+1], extra)
        self.current_noise = noise[-1].copy()
        state = initial_state() if dirty == 0 else self.states[dirty-1]
        del self.states[dirty:]
        for j in range(dirty, i+1):
            state, velocity = cv_step(state, x[j-dirty], noise[j-dirty])
            self.states.append(state)
        self.replayed_steps += i-dirty
        return velocity, state


class CausalSensorTimeEstimator:
    """Live packet API: no future timeline, target labels, or audit argument.

Call once per acquisition/output tick with its identity and monotonic time.
Only deliveries already available may be passed. The current contract assumes
components identify the same capture grid; unmatched independent sensor clocks
need an explicit synchronization/measurement adapter.
"""
    def __init__(self, backend, lag_s=.20):
        self.engine=TimeContractReplay([],backend,corrected=True,lag_s=lag_s)
        self.reference_generation = 0

    def reset(self):
        self.engine=TimeContractReplay([],self.engine.backend,corrected=True,lag_s=self.engine.lag_s)
        self.reference_generation += 1

    def advance(self, sample_id, output_time_s, deliveries):
        engine=self.engine
        if sample_id in engine.ids:
            raise ValueError('Duplicate output tick identity')
        if len(engine.times) and output_time_s<=engine.times[-1]:
            raise ValueError('Output clock must increase; reset for a new identity')
        index=len(engine.times);engine.ids[sample_id]=index
        engine.times=np.r_[engine.times,float(output_time_s)]
        engine.positions=np.vstack((engine.positions,[np.nan]*3))
        engine.linear=np.vstack((engine.linear,[np.nan]*3))
        engine.angular=np.vstack((engine.angular,[np.nan]*3))
        engine.sources=np.r_[engine.sources,4];engine.quality=np.vstack((engine.quality,[0.]*7))
        velocity,state=engine.update(index,deliveries,engine.times)
        mean,cov,yaw,base,robot,_,started=state
        direct = np.flatnonzero(np.isfinite(engine.positions).all(1) & (engine.sources <= 2))
        last = int(direct[-1]) if len(direct) else None
        observation = None
        if last is not None:
            observation = yaw_rotation(engine.states[last][2])@engine.positions[last]+engine.states[last][3]
        output = filter_state_output(mean,cov,state_time_s=output_time_s,
                    heading_local_rad=yaw,base_position_local_m=base,
                    base_velocity_base_mps=yaw_rotation(yaw).T@robot,
                    process_intensity=engine.current_noise[:3],initialized=started,
                    reference_epoch_s=engine.times[0],reference_generation=self.reference_generation,
                    last_direct_measurement_time_s=None if last is None else engine.times[last],
                    last_observation_local_m=observation,direct_update=engine.direct_update)
        # Keep the original velocity bit-for-bit, including floating-point order.
        output['velocity_base_mps'] = velocity.copy()
        output['state_base'][3:] = velocity
        return output


def replay(packets, profile, seed, backend, corrected=True, lag_s=.20):
    queue, clock, schedule = delivery_schedule(packets, profile, seed)
    engine = TimeContractReplay(packets, backend, corrected=corrected, lag_s=lag_s)
    prediction = []; covariances = []; positions = []
    for i, packet in enumerate(packets):
        engine.current_packet = packet
        velocity, state = engine.update(i, queue.arrived(packet['capture_time_s']), clock)
        prediction.append(velocity.copy()); covariances.append(state[1].copy()); positions.append(state[0][:3].copy())
    return dict(prediction=np.asarray(prediction), covariance=np.asarray(covariances),
                position_local=np.asarray(positions), schedule=schedule,
                stats=dict(expired_packets=engine.expired, paired_measurements=engine.paired,
                           unique_pairs=len(engine.unique_updates), replayed_steps=engine.replayed_steps,
                           max_replay_age_s=engine.max_replay_age_s,
                           sensor_clock_elapsed_error_s=float(clock[-1]-clock[0]-engine.times[-1]+engine.times[0])))
