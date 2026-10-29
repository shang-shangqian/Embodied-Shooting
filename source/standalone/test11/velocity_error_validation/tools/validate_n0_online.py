"""Check live, future-timeline-free packet API against three real replay streams."""
import csv
from pathlib import Path
import numpy as np
import torch
from n0_time_replay import CausalSensorTimeEstimator, NoiseBackend, delivery_schedule
from optimization_components import LearnedCVFilter
from speed_prediction.time_aware_velocity_model import TimeAwareQualityGatedGRU

BASE=Path('source/standalone/test11/velocity_error_validation')
DATA=BASE/'outputs/n0_capture_20261007'
RUN=BASE/'reports/n0_time_contract_20261007'


def main():
    torch.set_num_threads(1)
    if not (RUN/'COMPLETE').exists():raise ValueError('Reference evaluation incomplete')
    cp=torch.load(BASE/'reports/optimization_flow_runs_20261007/learned_filter_seed41/best_model.pt',map_location='cpu',weights_only=False)
    model=LearnedCVFilter(TimeAwareQualityGatedGRU(**cp['original_config'])).eval()
    model.load_state_dict(cp['state_dict'])
    saved=torch.load(RUN/'predictions.pt',map_location='cpu',weights_only=False)
    lookup={r['uid']:r for r in saved if r['backend']=='frozen_learned_filter' and r['profile']=='combined' and r['method']=='sample_time_replay'}
    chosen=[]
    for group in ('ss_standing','sd_standing','sd_moving'):
        folder=sorted(p for p in DATA.glob(group+'_*') if p.is_dir())[0]
        path=sorted(folder.rglob('timed_sensor_packets.pt'))[0]
        payload=torch.load(path,map_location='cpu',weights_only=False)
        packets=payload['packets'];uid=payload['episode_uid']
        seed=int(folder.name.rsplit('_',1)[1])+10000+100*int(path.parents[1].name.split('_')[1])+int(path.parent.name.split('_')[1])
        queue,_,_=delivery_schedule(packets,'combined',seed)
        online=CausalSensorTimeEstimator(NoiseBackend(model,'cpu'))
        prediction=[]
        for packet in packets:
            result=online.advance(packet['sample_id'],packet['capture_time_s'],queue.arrived(packet['capture_time_s']))
            prediction.append(result['velocity_base_mps'])
        difference=np.linalg.norm(np.array(prediction)-lookup[uid]['prediction'],axis=1)
        if difference.max()>.005:raise AssertionError(f'Live/reference mismatch {uid}: {difference.max()}')
        chosen.append(dict(group=group,uid=uid,frames=len(packets),
                           max_vector_difference_mps=float(difference.max()),
                           mean_vector_difference_mps=float(difference.mean()),
                           has_future_timeline_argument=False,has_gt_argument=False,
                           devices='CPU live API versus CUDA noise-network reference',tolerance_mps=.005))
    with (RUN/'live_api_validation.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(chosen[0]));writer.writeheader();writer.writerows(chosen)
    print(chosen)


if __name__=='__main__':main()
