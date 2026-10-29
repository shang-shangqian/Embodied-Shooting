"""Conservative environment-block CI; frozen model, no test tuning."""
from pathlib import Path
import numpy as np
import torch
from train_flow_ablations import write_csv,evaluate
from speed_prediction.time_aware_velocity_data import load_torch_payload
from analyze_error_sources import _load_model
from optimization_components import CHECKPOINT

root=Path('source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007')
data=load_torch_payload(root/'prepared_flows.pt')['test'];selected=load_torch_payload(root/'learned_filter_seed41/test_predictions.pt')
# Recompute only original NN residual on CPU; record.old retains the frozen
# CUDA baseline (including its exact numerical window membership).
model,_=_load_model(CHECKPOINT,torch.device('cpu'));torch.set_num_threads(4)
_,_,_,reference=evaluate(model,data,{},torch.device('cpu'),return_rows=True)
rng=np.random.default_rng(261007);rows=[]
for scene in ('ss','sd'):
    blocks={}
    for r,a,b in zip(data,reference,selected):
        if r['scenario']!=scene:continue
        mask=r['score'].numpy()&r['ground'].numpy()
        if not mask.any():continue
        key=(r['seed'],Path(r['path']).parts[-3]);blocks.setdefault(key,[]).append((a['error'][mask],b['error'][mask]))
    blocks=[(np.concatenate([x[0] for x in block]),np.concatenate([x[1] for x in block])) for block in blocks.values()]
    means=[];quantiles=[]
    for _ in range(500):
        indices=rng.integers(len(blocks),size=len(blocks));a=np.concatenate([blocks[i][0] for i in indices]);b=np.concatenate([blocks[i][1] for i in indices]);means.append(b.mean()-a.mean());quantiles.append(np.quantile(b,.95)-np.quantile(a,.95))
    for metric,value in [('mean',means),('p95',quantiles)]:rows.append(dict(scenario=scene,metric=metric,blocks=len(blocks),unit='collection seed x environment; episodes grouped',ci_low=float(np.quantile(value,.025)),ci_high=float(np.quantile(value,.975))))
write_csv(root/'evaluation/paired_block_bootstrap.csv',rows);print(rows)
