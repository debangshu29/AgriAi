from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import SUPPORTED_ARCHS, build_model

SEED = 42


def load_metadata(weights_dir: Path, arch: str, image_size: int | None):
    metadata_path = weights_dir / f'{arch}_metadata.json'
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if image_size is not None:
        metadata['image_size'] = image_size
    metadata.setdefault('image_size', 224)
    return metadata


def maybe_subset(dataset, max_samples: int | None):
    if not max_samples or max_samples >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def plot_confusion_matrix(matrix, labels, output_path: Path):
    figure_size = max(10, len(labels) * 0.32)
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(matrix, interpolation='nearest', cmap='Greens')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
    plt.yticks(tick_marks, labels, fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def evaluate(args):
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes_path = weights_dir / f'{args.arch}_classes.json'
    weights_path = weights_dir / f'{args.arch}_classifier.pt'
    if not classes_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f'Missing weights/classes for {args.arch} in {weights_dir}')

    class_names = json.loads(classes_path.read_text(encoding='utf-8'))
    metadata = load_metadata(weights_dir, args.arch, args.image_size)
    image_size = int(metadata['image_size'])

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_path = Path(args.dataset_dir) / args.split
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataset = maybe_subset(dataset, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.arch, len(class_names), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    y_true = []
    y_pred = []
    rows = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            for label, pred, confidence in zip(labels.tolist(), predictions.tolist(), confidences.tolist()):
                rows.append(
                    {
                        'true_label': dataset.dataset.classes[label] if isinstance(dataset, Subset) else dataset.classes[label],
                        'predicted_label': class_names[pred],
                        'confidence': round(float(confidence), 5),
                    }
                )

    true_names = [dataset.dataset.classes[index] if isinstance(dataset, Subset) else dataset.classes[index] for index in y_true]
    pred_names = [class_names[index] for index in y_pred]
    labels = sorted(set(class_names) | set(true_names))
    report = classification_report(true_names, pred_names, labels=labels, zero_division=0, output_dict=True)
    matrix = confusion_matrix(true_names, pred_names, labels=labels)

    summary = {
        'arch': args.arch,
        'dataset_dir': str(dataset_path),
        'image_size': image_size,
        'samples': len(y_true),
        'accuracy': round(accuracy_score(true_names, pred_names), 5),
        'macro_f1': round(f1_score(true_names, pred_names, average='macro', zero_division=0), 5),
    }
    (output_dir / f'{args.arch}_{args.split}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output_dir / f'{args.arch}_{args.split}_classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    with (output_dir / f'{args.arch}_{args.split}_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['true_label', 'predicted_label', 'confidence'])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / f'{args.arch}_{args.split}_confusion_matrix.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])

    plot_confusion_matrix(matrix, labels, output_dir / f'{args.arch}_{args.split}_confusion_matrix.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained AgriVision classifier.')
    parser.add_argument('--dataset-dir', required=True, help='Dataset root containing split folders like train/ and val/.')
    parser.add_argument('--split', default='val')
    parser.add_argument('--weights-dir', default='ml/weights')
    parser.add_argument('--output-dir', default='ml/reports')
    parser.add_argument('--arch', choices=SUPPORTED_ARCHS, default='mobilenet_v3_small')
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    evaluate(parser.parse_args())
