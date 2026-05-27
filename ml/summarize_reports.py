from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import build_model

MODEL_NAMES = {
    'mobilenet_v3_small': 'MobileNetV3 Small',
    'resnet18': 'ResNet18',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit_b_16': 'ViT-B/16',
}


def summarize(args):
    weights_dir = Path(args.weights_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arch in args.archs:
        classes_path = weights_dir / f'{arch}_classes.json'
        metadata_path = weights_dir / f'{arch}_metadata.json'
        history_path = weights_dir / f'{arch}_history.json'
        summary_path = reports_dir / f'{arch}_{args.split}_summary.json'
        report_path = reports_dir / f'{arch}_{args.split}_classification_report.json'
        if not all(path.exists() for path in [classes_path, metadata_path, history_path, summary_path, report_path]):
            print({'arch': arch, 'skipped': True, 'reason': 'missing artifacts'}, flush=True)
            continue

        classes = json.loads(classes_path.read_text(encoding='utf-8'))
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        history = json.loads(history_path.read_text(encoding='utf-8'))
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        report = json.loads(report_path.read_text(encoding='utf-8'))
        model = build_model(arch, len(classes), pretrained=False)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        best_record = metadata.get('best_record') or (history[-1] if history else {})

        rows.append(
            {
                'model': MODEL_NAMES.get(arch, arch),
                'arch': arch,
                'total_parameters': total_parameters,
                'image_size': metadata.get('image_size'),
                'head_epochs': metadata.get('head_epochs', 0),
                'finetune_epochs': metadata.get('finetune_epochs', metadata.get('epochs')),
                'train_samples': metadata.get('train_samples'),
                'validation_samples': summary.get('samples', metadata.get('val_samples')),
                'best_stage': best_record.get('stage'),
                'best_epoch': best_record.get('epoch'),
                'train_loss': best_record.get('train_loss'),
                'val_loss': best_record.get('val_loss'),
                'accuracy': summary.get('accuracy'),
                'macro_precision': round(report['macro avg']['precision'], 5),
                'macro_recall': round(report['macro avg']['recall'], 5),
                'macro_f1': summary.get('macro_f1'),
            }
        )

    output_csv = reports_dir / args.output_csv
    output_json = reports_dir / args.output_json
    output_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    if rows:
        with output_csv.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a paper-friendly model comparison table from evaluation reports.')
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--reports-dir', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--archs', nargs='+', required=True)
    parser.add_argument('--output-csv', default='model_comparison.csv')
    parser.add_argument('--output-json', default='model_comparison.json')
    summarize(parser.parse_args())
