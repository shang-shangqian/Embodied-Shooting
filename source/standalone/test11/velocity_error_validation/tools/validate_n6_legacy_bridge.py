"""Verify the CV feature API still matches saved pre-interface CUDA predictions."""
import csv
from pathlib import Path
import numpy as np
import torch
from online_experimental_velocity import CausalVelocityCandidate

BASE=Path('source/standalone/test11/velocity_error_validation')
RUN=BASE/'reports/n6_interface_20261007'
FROZEN=BASE/'reports/optimization_flow_runs_20261007'


def main():
    torch.set_num_threads(1)
    data=torch.load(FROZEN/'prepared_flows.pt',map_location='cpu',weights_only=False)['test']
    reference={r['uid']:r for r in torch.load(FROZEN/'learned_filter_seed41/test_predictions.pt',map_location='cpu',weights_only=False)}
    rows=[]
    for scenario in ('ss','sd'):
        for profile in ('clean','mixed'):
            # Use a complete long stream for each group, not an easy short fragment.
            record=max((r for r in data if r['scenario']==scenario and r['profile']==profile),key=lambda r:len(r['x']))
            api=CausalVelocityCandidate(FROZEN/'learned_filter_seed41/best_model.pt',device='cpu')
            prediction=[];cov_delta=0.;state_delta=0.;min_eig=float('inf')
            for x,source,timestamp in zip(record['x'],record['source'],record['time']):
                result=api.update(x,int(source),float(timestamp))
                prediction.append(result['velocity_base_mps'])
                cov_delta=max(cov_delta,float(np.max(np.abs(result['covariance_local']-api.state[1][0].numpy()))))
                state_delta=max(state_delta,float(np.max(np.abs(result['state_local']-api.state[0][0].numpy()))))
                min_eig=min(min_eig,float(np.linalg.eigvalsh(result['covariance_local']).min()))
                if result['state_time_s']!=float(timestamp):raise AssertionError('Feature API misdated state')
            error=np.linalg.norm(np.array(prediction)-reference[record['uid']]['prediction'],axis=1)
            if error.max()>.001 or state_delta!=0 or cov_delta>1e-6 or min_eig<=0:raise AssertionError('Legacy bridge regression')
            rows.append(dict(uid=record['uid'],scenario=scenario,profile=profile,frames=len(prediction),
                velocity_max_difference_mps=float(error.max()),state_max_difference=state_delta,
                covariance_max_difference=cov_delta,covariance_min_eigenvalue=min_eig,
                reference='CPU interface versus saved frozen CUDA stream; unchanged weights'))
    with (RUN/'legacy_bridge_validation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(rows)


if __name__=='__main__':main()
