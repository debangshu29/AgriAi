from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DatasetSpec:
    name: str
    split: str
    source: str


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace('&', 'and')
    text = re.sub(r'[()\/,]+', ' ', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def normalize_plantvillage_label(raw_label: str) -> str:
    parts = re.split(r'_{2,}', raw_label, maxsplit=1)
    if len(parts) == 2:
        crop, disease = parts
    else:
        crop, disease = 'crop', raw_label
    crop_slug = slugify(crop)
    disease_slug = slugify(disease)
    return f'{crop_slug}_{disease_slug}'


def normalize_pathology_label(label_names: Iterable[str]) -> str:
    names = [slugify(name) for name in label_names]
    if not names:
        return 'apple_unknown'
    if len(names) == 1:
        return f'apple_{names[0]}'
    if 'healthy' in names and len(names) == 1:
        return 'apple_healthy'
    if 'complex' in names:
        return 'apple_complex'
    return 'apple_multi_disease'


class MultiSourcePlantDataset(Dataset):
    def __init__(self, image_size: int, pathology_train_limit: int | None = None, pathology_val_limit: int | None = None):
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cache_dir = 'data/hf_cache'
        self.pathology_train_limit = pathology_train_limit
        self.pathology_val_limit = pathology_val_limit
        self.train_splits = []
        self.val_splits = []
        self.class_names = []
        self.class_to_idx = {}
        self.train_records = []
        self.val_records = []
        self.train_class_counts = Counter()
        self.val_class_counts = Counter()
        self.source_counts = Counter()

        self._build_records()

    def _build_records(self):
        specs = [
            DatasetSpec('GVJahnavi/PlantVillage_dataset', 'train', 'plantvillage'),
            DatasetSpec('GVJahnavi/PlantVillage_dataset', 'test', 'plantvillage_val'),
            DatasetSpec('timm/plant-pathology-2021', f'train[:{self.pathology_train_limit}]' if self.pathology_train_limit else 'train', 'plant_pathology'),
            DatasetSpec('timm/plant-pathology-2021', f'validation[:{self.pathology_val_limit}]' if self.pathology_val_limit else 'validation', 'plant_pathology_val'),
        ]

        pending_train = []
        pending_val = []

        for spec in specs:
            print(f'Loading {spec.name} [{spec.split}]')
            dataset = load_dataset(spec.name, split=spec.split, cache_dir=self.cache_dir)
            print(f'Loaded {spec.name} [{spec.split}] -> {len(dataset)} records')
            is_train = spec.source in {'plantvillage', 'plant_pathology'}
            target_records = pending_train if is_train else pending_val
            split_store = self.train_splits if is_train else self.val_splits
            split_index = len(split_store)
            split_store.append(dataset)

            if spec.source.startswith('plantvillage'):
                labels = dataset['label']
                for row_index, label_index in enumerate(labels):
                    label_name = dataset.features['label'].int2str(label_index)
                    canonical_label = normalize_plantvillage_label(label_name)
                    target_records.append((split_index, row_index, canonical_label, spec.source))
                    self.source_counts[spec.source] += 1
            else:
                label_names_column = dataset['label_names']
                for row_index, label_names in enumerate(label_names_column):
                    canonical_label = normalize_pathology_label(label_names)
                    target_records.append((split_index, row_index, canonical_label, spec.source))
                    self.source_counts[spec.source] += 1

        self.class_names = sorted({record[2] for record in pending_train + pending_val})
        print(f'Collected {len(pending_train)} training records and {len(pending_val)} validation records across {len(self.class_names)} classes')
        self.class_to_idx = {label: index for index, label in enumerate(self.class_names)}

        for split_index, row_index, canonical_label, source in pending_train:
            class_index = self.class_to_idx[canonical_label]
            self.train_records.append((split_index, row_index, class_index))
            self.train_class_counts[canonical_label] += 1

        for split_index, row_index, canonical_label, source in pending_val:
            class_index = self.class_to_idx[canonical_label]
            self.val_records.append((split_index, row_index, class_index))
            self.val_class_counts[canonical_label] += 1

    def dataset_for_training(self):
        return IndexedPlantDataset(self.train_splits, self.train_records, self.train_transform)

    def dataset_for_validation(self):
        return IndexedPlantDataset(self.val_splits, self.val_records, self.eval_transform)


class IndexedPlantDataset(Dataset):
    def __init__(self, splits, records, transform):
        self.splits = splits
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        split_index, row_index, class_index = self.records[index]
        example = self.splits[split_index][row_index]
        image = example['image'].convert('RGB')
        return self.transform(image), class_index


def build_model(num_classes: int, pretrained: bool = True):
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def freeze_backbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        'loss': running_loss / len(loader.dataset),
        'accuracy': accuracy_score(targets, predictions),
        'macro_f1': f1_score(targets, predictions, average='macro', zero_division=0),
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train a multi-source plant disease classifier with PlantVillage and Plant Pathology 2021.')
    parser.add_argument('--output-dir', default='ml/weights')
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--head-epochs', type=int, default=1)
    parser.add_argument('--finetune-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--finetune-learning-rate', type=float, default=0.00015)
    parser.add_argument('--pathology-train-limit', type=int, default=4000)
    parser.add_argument('--pathology-val-limit', type=int, default=1200)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    set_seed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = MultiSourcePlantDataset(
        image_size=args.image_size,
        pathology_train_limit=args.pathology_train_limit,
        pathology_val_limit=args.pathology_val_limit,
    )
    train_dataset = bundle.dataset_for_training()
    val_dataset = bundle.dataset_for_validation()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(bundle.class_names), pretrained=args.pretrained).to(device)

    class_counts = Counter(label for _, _, label in bundle.train_records)
    class_weights = []
    for class_index in range(len(bundle.class_names)):
        class_weights.append(1.0 / max(class_counts[class_index], 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    history = []
    best_f1 = 0.0

    if args.head_epochs > 0:
        freeze_backbone(model)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        for epoch in range(1, args.head_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, device)
            record = {
                'stage': 'head',
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'val_loss': round(metrics['loss'], 4),
                'accuracy': round(metrics['accuracy'], 4),
                'macro_f1': round(metrics['macro_f1'], 4),
            }
            history.append(record)
            print(record)
            if metrics['macro_f1'] >= best_f1:
                best_f1 = metrics['macro_f1']
                torch.save(model.state_dict(), output_dir / 'mobilenet_v3_small_classifier.pt')

    if args.finetune_epochs > 0:
        unfreeze_all(model)
        optimizer = AdamW(model.parameters(), lr=args.finetune_learning_rate)
        for epoch in range(1, args.finetune_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, device)
            record = {
                'stage': 'finetune',
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'val_loss': round(metrics['loss'], 4),
                'accuracy': round(metrics['accuracy'], 4),
                'macro_f1': round(metrics['macro_f1'], 4),
            }
            history.append(record)
            print(record)
            if metrics['macro_f1'] >= best_f1:
                best_f1 = metrics['macro_f1']
                torch.save(model.state_dict(), output_dir / 'mobilenet_v3_small_classifier.pt')

    (output_dir / 'mobilenet_v3_small_classes.json').write_text(json.dumps(bundle.class_names, indent=2), encoding='utf-8')
    (output_dir / 'mobilenet_v3_small_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (output_dir / 'mobilenet_v3_small_dataset_report.json').write_text(
        json.dumps(
            {
                'sources': bundle.source_counts,
                'train_class_counts': bundle.train_class_counts,
                'val_class_counts': bundle.val_class_counts,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'Best macro F1: {best_f1:.4f}')
    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Classes: {len(bundle.class_names)}')


if __name__ == '__main__':
    main()



