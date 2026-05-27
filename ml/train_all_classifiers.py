from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import SUPPORTED_ARCHS


def train_all(args):
    archs = args.archs or list(SUPPORTED_ARCHS)
    for arch in archs:
        command = [
            sys.executable,
            '-u',
            'ml/train_classifier.py',
            '--dataset-dir',
            args.dataset_dir,
            '--output-dir',
            args.output_dir,
            '--arch',
            arch,
            '--epochs',
            str(args.epochs),
            '--freeze-epochs',
            str(args.freeze_epochs),
            '--batch-size',
            str(args.batch_size),
            '--image-size',
            str(args.image_size),
            '--learning-rate',
            str(args.learning_rate),
            '--head-learning-rate',
            str(args.head_learning_rate),
            '--min-learning-rate',
            str(args.min_learning_rate),
            '--weight-decay',
            str(args.weight_decay),
            '--label-smoothing',
            str(args.label_smoothing),
            '--optimizer',
            args.optimizer,
            '--scheduler',
            args.scheduler,
            '--augmentation',
            args.augmentation,
            '--save-metric',
            args.save_metric,
            '--seed',
            str(args.seed),
            '--log-every',
            str(args.log_every),
        ]
        if args.pretrained:
            command.append('--pretrained')
        if args.freeze_backbone:
            command.append('--freeze-backbone')
        if args.class_weights:
            command.append('--class-weights')
        if args.max_train_samples:
            command.extend(['--max-train-samples', str(args.max_train_samples)])
        if args.max_val_samples:
            command.extend(['--max-val-samples', str(args.max_val_samples)])
        if args.num_workers:
            command.extend(['--num-workers', str(args.num_workers)])
        if args.patience:
            command.extend(['--patience', str(args.patience)])

        print(f'Running: {" ".join(command)}', flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV3, ResNet18, EfficientNet-B0, and ViT-B/16 sequentially.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', default='ml/weights')
    parser.add_argument('--archs', nargs='*', choices=SUPPORTED_ARCHS)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--freeze-epochs', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--head-learning-rate', type=float, default=0.001)
    parser.add_argument('--min-learning-rate', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--optimizer', choices=('adam', 'adamw'), default='adamw')
    parser.add_argument('--scheduler', choices=('none', 'cosine', 'step'), default='cosine')
    parser.add_argument('--augmentation', choices=('light', 'standard', 'strong'), default='standard')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--save-metric', choices=('accuracy', 'macro_f1'), default='macro_f1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true', help='Train only each architecture-specific classifier head.')
    train_all(parser.parse_args())
