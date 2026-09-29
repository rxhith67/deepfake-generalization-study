"""Validation-only threshold helpers and saved probability calibration reports."""
from pathlib import Path
import pandas as pd

from src.eval.metrics import find_optimal_threshold
from src.experiments.common import configuration, save_csv, atomic_text, update_master
from src.experiments.evaluation import measurement, probability_figures


def select_threshold(validation):
    if set(validation['split']) != {'val'} or not validation.dataset.str.startswith('ffpp').all():
        raise ValueError('Threshold selection requires FF++ validation only')
    if validation.sample_id.duplicated().any():
        raise ValueError('Duplicate validation IDs')
    return find_optimal_threshold(validation.label,validation.prob_fake)


def run(config):
    root = Path(config['output_dir'])
    destination = root/'03_calibration'
    results, distributions = [], []
    sources = [(root/'00_baseline_reproduction',('test','external')),
               (root/'02_external_bias_audit',('normalized',))]
    by_model = {}
    for directory,suffixes in sources:
        for suffix in suffixes:
            for model in ('hybrid','xception','ensemble'):
                path = directory/f'{model}_{suffix}_predictions.csv'
                if not path.exists():
                    raise FileNotFoundError(path)
                rows = pd.read_csv(path)
                results.append(measurement(rows,'calibration',config))
                by_model.setdefault(model,{})[str(rows.dataset.iloc[0])] = rows
                distributions.append(rows)
    save_csv(destination/'calibration_metrics.csv',pd.DataFrame(results))
    update_master(config['output_dir'],results)
    save_csv(destination/'threshold_transfer.csv',pd.DataFrame(results)[['model','test_dataset','threshold','threshold_source','auc','balanced_accuracy','ece','nll','brier']])
    save_csv(destination/'confidence_distributions'/'probabilities.csv',pd.concat(distributions,ignore_index=True))
    for model,frames in by_model.items():
        if len({float(f.threshold.iloc[0]) for f in frames.values()}) != 1:
            raise ValueError('Threshold changed across domains')
        probability_figures(frames,destination/'reliability_diagrams'/model)
    atomic_text(destination/'calibration.md','# Calibration and threshold transfer\n\n'
                '15 equal-width probability bins; probability-ECE, NLL clipped at 1e-7, and Brier score. '
                'Thresholds are frozen from FF++ validation, not fitted externally. '
                'Probability histograms show domain/class shifts; calibration scores are not threshold-selection criteria.\n')


if __name__ == '__main__':
    run(configuration())
