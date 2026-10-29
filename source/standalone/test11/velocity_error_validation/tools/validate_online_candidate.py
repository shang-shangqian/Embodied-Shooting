"""Verify selected GT-free online API against frozen test replay (no tuning)."""
import json,time
from pathlib import Path
import numpy as np
import torch
from online_experimental_velocity import CausalVelocityCandidate
from speed_prediction.time_aware_velocity_data import load_torch_payload
from train_flow_ablations import write_csv

root=Path('source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007')
selection=json.loads((root/'evaluation/frozen_selection.json').read_text());data=load_torch_payload(root/'prepared_flows.pt')['test'];saved=load_torch_payload(root/selection['run']/'test_predictions.pt');truth={r['uid']:r for r in saved};torch.set_num_threads(4)
records=[]
for scenario,profile in [('ss','clean'),('ss','mixed'),('sd','clean'),('sd','mixed')]:
    record=next(r for r in data if r['scenario']==scenario and r['profile']==profile and len(r['x'])>200)
    runner=CausalVelocityCandidate.from_selection(root,device='cuda:0');pred=[];times=[]
    for i in range(len(record['x'])):
        begin=time.perf_counter()
        # This is the full API input: no audit/GT position/label argument.
        out=runner.update(record['x'][i],int(record['source'][i]),float(record['time'][i]),tracker_reset=bool(record['reset'][i]))
        pred.append(out['velocity_base_mps']);times.append((time.perf_counter()-begin)*1000)
    difference=np.linalg.norm(np.array(pred)-truth[record['uid']]['prediction'],axis=1)
    records.append(dict(uid=record['uid'],scenario=scenario,profile=profile,frames=len(pred),mean_vector_difference_mps=float(difference.mean()),max_vector_difference_mps=float(difference.max()),api_p50_ms=float(np.quantile(times,.5)),api_p95_ms=float(np.quantile(times,.95))))
    if difference.max()>.001:raise RuntimeError(f'Online replay mismatch {record["uid"]}: {difference.max()}')
    print(records[-1],flush=True)
write_csv(root/'evaluation/online_reference_validation.csv',records)
(root/'evaluation/online_reference_limitations.json').write_text(json.dumps({'protocol':'frozen candidate; no further hyperparameter changes','scope':'4 independent test episodes, source and timestamps only','cache':'full causal prefix for reference consistency; elapsed API cost grows with episode age','comparison':'single-target API versus cached batched CUDA stream','tolerance_mps':.001},indent=2))
