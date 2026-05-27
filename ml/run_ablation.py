from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project.settings')

import django

django.setup()

from monitoring.services.risk import compute_risk_score


def run_ablation(args):
    predictions_path = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with predictions_path.open(encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            confidence = float(row['confidence'])
            label = row['predicted_label']
            rows.append(
                {
                    'label': label,
                    'confidence': confidence,
                    'classifier_only_confidence': confidence,
                    'classifier_weather_risk': compute_risk_score(label, confidence, 0.0, args.temperature, args.humidity).risk_score,
                    'classifier_segmentation_risk': compute_risk_score(label, confidence, args.affected_area, 27.0, 55.0).risk_score,
                    'full_pipeline_risk': compute_risk_score(label, confidence, args.affected_area, args.temperature, args.humidity).risk_score,
                }
            )

    averages = {
        key: round(sum(row[key] for row in rows) / max(len(rows), 1), 4)
        for key in [
            'classifier_only_confidence',
            'classifier_weather_risk',
            'classifier_segmentation_risk',
            'full_pipeline_risk',
        ]
    }
    summary = {
        'samples': len(rows),
        'assumed_temperature_c': args.temperature,
        'assumed_humidity_pct': args.humidity,
        'assumed_affected_area_pct': args.affected_area,
        'averages': averages,
    }

    with (output_dir / 'ablation_rows.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / 'ablation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a lightweight ablation over classifier, segmentation, and weather risk components.')
    parser.add_argument('--predictions-csv', required=True, help='Predictions CSV from evaluate_classifier.py.')
    parser.add_argument('--output-dir', default='ml/reports/ablation')
    parser.add_argument('--temperature', type=float, default=31.0)
    parser.add_argument('--humidity', type=float, default=82.0)
    parser.add_argument('--affected-area', type=float, default=18.0)
    run_ablation(parser.parse_args())
