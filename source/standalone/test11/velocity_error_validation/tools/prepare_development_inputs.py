"""Prepare existing paired D inputs for later development regression checks.
They are never merged with the new train/validation/test campaign.
"""
from pathlib import Path
import torch,json
from analyze_error_sources import _load_model
from analyze_inference_context import prepare_pairs
from optimization_components import CHECKPOINT
from train_flow_ablations import prepare_record

torch.set_num_threads(4)
base=Path('source/standalone/test11/velocity_error_validation');out=base/'reports/optimization_flow_runs_20261007';out.mkdir(exist_ok=True)
template,_=_load_model(CHECKPOINT,torch.device('cuda:0'));result={};manifest=[]
for dataset in ('pilot_fixed_camera_v2_warmup_flowgate','formal_fixed_camera_3090_20260929'):
    streams,_=prepare_pairs(sorted((base/'outputs'/dataset).rglob('*.taqstream')),0);records=[]
    for s in streams:
        if s.condition!='D':continue
        record,row=prepare_record(s.path,template,dataset);record['score']=torch.tensor(s.score);records.append(record);manifest.append(row)
    result[dataset]=records;print('Prepared development',dataset,len(records),flush=True)
torch.save(result,out/'development_inputs.pt');(out/'development_manifest.json').write_text(json.dumps(manifest,indent=2))
