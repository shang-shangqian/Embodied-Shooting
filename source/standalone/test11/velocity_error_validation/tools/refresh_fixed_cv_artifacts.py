"""Repair fixed-CV CSVs using frozen validation parameters, without reselection."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from analyze_error_sources import _load_model
from optimization_components import CHECKPOINT, LearnedCVFilter
from speed_prediction.time_aware_velocity_data import load_torch_payload
from train_flow_ablations import evaluate, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = args.runs / 'evaluation'
    selection = json.loads((report / 'frozen_selection.json').read_text())
    choice = selection['fixed_cv_validation_choice']
    data = load_torch_payload(args.runs / 'prepared_flows.pt')
    device = torch.device(args.device)
    normalizer, _ = _load_model(CHECKPOINT, device)
    model = LearnedCVFilter(normalizer, False, choice['q'], choice['r']).to(device)
    _, summary, episodes, _ = evaluate(
        model, data['test'], dict(missing=True, aligned=True, kalman=True),
        device, return_rows=True,
    )
    archive = report / 'pre_fixed_cv_csv_repair'
    archive.mkdir(exist_ok=True)
    for filename, rows in [('fixed_cv_test_summary.csv', summary),
                           ('fixed_cv_test_episodes.csv', episodes)]:
        original = report / filename
        if original.exists() and not (archive / filename).exists():
            shutil.copy2(original, archive / filename)
        write_csv(original, rows)
    (report / 'fixed_cv_csv_repair.json').write_text(json.dumps({
        'reason': 'Fixed CV summary variables were overwritten by the subsequent baseline loop.',
        'validation_parameters_unchanged': choice,
        'selection_unchanged': True,
        'other_predictions_unchanged': True,
    }, indent=2))
    for row in summary:
        if row['group'] == 'all':
            print(row, flush=True)


if __name__ == '__main__':
    main()
