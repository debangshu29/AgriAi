from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import SUPPORTED_ARCHS, build_model

DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples: int | None, seed: int):
    if not max_samples or max_samples >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def build_train_transform(image_size: int, augmentation: str):
    if augmentation == 'light':
        steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
        ]
    elif augmentation == 'strong':
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14),
        ]
        if hasattr(transforms, 'RandAugment'):
            steps.append(transforms.RandAugment(num_ops=2, magnitude=6))
    else:
        steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
        ]

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(
    dataset_dir: Path,
    batch_size: int,
    image_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    num_workers: int,
    augmentation: str,
    seed: int,
    device: torch.device,
):
    train_dataset = datasets.ImageFolder(dataset_dir / 'train', transform=build_train_transform(image_size, augmentation))
    val_dataset = datasets.ImageFolder(dataset_dir / 'val', transform=build_eval_transform(image_size))

    train_dataset = maybe_subset(train_dataset, max_train_samples, seed)
    val_dataset = maybe_subset(val_dataset, max_val_samples, seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def dataset_targets(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.targets[index] for index in dataset.indices]
    return list(dataset.targets)


def build_criterion(args, train_dataset, class_count: int, device: torch.device):
    weights = None
    if args.class_weights:
        counts = Counter(dataset_targets(train_dataset))
        total = sum(counts.values())
        weights = []
        for class_index in range(class_count):
            class_frequency = counts.get(class_index, 0)
            weights.append(total / max(class_count * class_frequency, 1))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)


def evaluate(model, loader, criterion, device):
    model.eval()
    targets = []
    predictions = []
    running_loss = 0.0

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            targets.extend(labels.cpu().tolist())

    return {
        'val_loss': running_loss / len(loader.dataset),
        'accuracy': accuracy_score(targets, predictions),
        'macro_f1': f1_score(targets, predictions, average='macro', zero_division=0),
    }


def freeze_backbone(model, arch: str):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if arch == 'mobilenet_v3_small':
        trainable_module = model.classifier
    elif arch == 'resnet18':
        trainable_module = model.fc
    elif arch == 'efficientnet_b0':
        trainable_module = model.classifier
    elif arch == 'vit_b_16':
        trainable_module = model.heads
    else:
        raise ValueError(f'Unsupported architecture for freezing: {arch}')

    for parameter in trainable_module.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_optimizer(args, model, learning_rate: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == 'adam':
        return Adam(parameters, lr=learning_rate, weight_decay=args.weight_decay)
    return AdamW(parameters, lr=learning_rate, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer, epochs: int):
    if args.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=args.min_learning_rate)
    if args.scheduler == 'step':
        return StepLR(optimizer, step_size=max(args.step_size, 1), gamma=args.gamma)
    return None


def save_checkpoint(args, output_dir: Path, class_names, model, metadata, history, best_record):
    torch.save(model.state_dict(), output_dir / f'{args.arch}_classifier.pt')
    (output_dir / f'{args.arch}_classes.json').write_text(json.dumps(class_names, indent=2), encoding='utf-8')
    (output_dir / f'{args.arch}_metadata.json').write_text(
        json.dumps({**metadata, 'best_record': best_record}, indent=2),
        encoding='utf-8',
    )
    (output_dir / f'{args.arch}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def train_one_stage(
    args,
    stage_name: str,
    stage_epochs: int,
    learning_rate: float,
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    output_dir: Path,
    class_names,
    metadata,
    history,
    best_state,
):
    if stage_epochs <= 0:
        return

    if stage_name == 'head':
        freeze_backbone(model, args.arch)
    else:
        unfreeze_all(model)

    total_parameters, trainable_parameters = count_parameters(model)
    optimizer = build_optimizer(args, model, learning_rate)
    scheduler = build_scheduler(args, optimizer, stage_epochs)
    stage_no_improvement = 0

    print(
        {
            'stage': stage_name,
            'stage_epochs': stage_epochs,
            'learning_rate': learning_rate,
            'total_parameters': total_parameters,
            'trainable_parameters': trainable_parameters,
        },
        flush=True,
    )

    for stage_epoch in range(1, stage_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_index % args.log_every == 0:
                print(
                    {
                        'stage': stage_name,
                        'stage_epoch': stage_epoch,
                        'batch': batch_index,
                        'train_batches': len(train_loader),
                        'loss': round(loss.item(), 4),
                    },
                    flush=True,
                )

        metrics = evaluate(model, val_loader, criterion, device)
        epoch_loss = running_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_record = {
            'stage': stage_name,
            'epoch': len(history) + 1,
            'stage_epoch': stage_epoch,
            'train_loss': round(epoch_loss, 4),
            'val_loss': round(metrics['val_loss'], 4),
            'accuracy': round(metrics['accuracy'], 4),
            'macro_f1': round(metrics['macro_f1'], 4),
            'learning_rate': current_lr,
            'trainable_parameters': trainable_parameters,
        }
        history.append(epoch_record)
        print(epoch_record, flush=True)

        metric_value = metrics[args.save_metric]
        improved = best_state['value'] is None or metric_value >= best_state['value'] + args.min_delta
        if improved:
            best_state['value'] = metric_value
            best_state['record'] = epoch_record
            stage_no_improvement = 0
            save_checkpoint(args, output_dir, class_names, model, metadata, history, epoch_record)
        else:
            stage_no_improvement += 1

        if args.patience and stage_no_improvement >= args.patience:
            print({'stage': stage_name, 'early_stop': True, 'patience': args.patience}, flush=True)
            break

        if scheduler is not None:
            scheduler.step()


def train(args):
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_train_dataset = datasets.ImageFolder(dataset_dir / 'train')
    class_names = base_train_dataset.classes

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        dataset_dir,
        args.batch_size,
        args.image_size,
        args.max_train_samples,
        args.max_val_samples,
        args.num_workers,
        args.augmentation,
        args.seed,
        device,
    )
    model = build_model(args.arch, len(class_names), args.pretrained).to(device)
    criterion = build_criterion(args, train_dataset, len(class_names), device)
    total_parameters, trainable_parameters = count_parameters(model)

    metadata = {
        'arch': args.arch,
        'image_size': args.image_size,
        'dataset_dir': str(dataset_dir),
        'classes': len(class_names),
        'head_epochs': args.freeze_epochs if not args.freeze_backbone else args.epochs,
        'finetune_epochs': 0 if args.freeze_backbone else args.epochs,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'augmentation': args.augmentation,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'learning_rate': args.learning_rate,
        'head_learning_rate': args.head_learning_rate,
        'min_learning_rate': args.min_learning_rate,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'class_weights': args.class_weights,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone,
        'seed': args.seed,
        'device': str(device),
        'total_parameters': total_parameters,
        'initial_trainable_parameters': trainable_parameters,
        'save_metric': args.save_metric,
    }
    history = []
    best_state = {'value': None, 'record': None}

    print(
        {
            'classes': len(class_names),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'device': str(device),
            'arch': args.arch,
            'pretrained': args.pretrained,
            'image_size': args.image_size,
            'total_parameters': total_parameters,
        },
        flush=True,
    )

    if args.freeze_backbone:
        train_one_stage(
            args,
            'head',
            args.epochs,
            args.head_learning_rate,
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            output_dir,
            class_names,
            metadata,
            history,
            best_state,
        )
    else:
        train_one_stage(
            args,
            'head',
            args.freeze_epochs,
            args.head_learning_rate,
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            output_dir,
            class_names,
            metadata,
            history,
            best_state,
        )
        train_one_stage(
            args,
            'finetune',
            args.epochs,
            args.learning_rate,
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            output_dir,
            class_names,
            metadata,
            history,
            best_state,
        )

    if history:
        (output_dir / f'{args.arch}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Best {args.save_metric}: {(best_state["value"] or 0.0):.4f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a crop disease classifier with transfer learning.')
    parser.add_argument('--dataset-dir', required=True, help='Path with train/ and val/ subfolders.')
    parser.add_argument('--output-dir', default='ml/weights', help='Directory for weights and metadata.')
    parser.add_argument('--arch', choices=SUPPORTED_ARCHS, default='mobilenet_v3_small')
    parser.add_argument('--epochs', type=int, default=8, help='Fine-tuning epochs, or head-only epochs with --freeze-backbone.')
    parser.add_argument('--freeze-epochs', type=int, default=0, help='Warm-up epochs that train only the classifier head before fine-tuning.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Fine-tuning learning rate.')
    parser.add_argument('--head-learning-rate', type=float, default=0.001, help='Classifier-head learning rate.')
    parser.add_argument('--min-learning-rate', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--optimizer', choices=('adam', 'adamw'), default='adamw')
    parser.add_argument('--scheduler', choices=('none', 'cosine', 'step'), default='cosine')
    parser.add_argument('--step-size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--augmentation', choices=('light', 'standard', 'strong'), default='standard')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--patience', type=int, default=0, help='Early-stop patience per stage. 0 disables early stopping.')
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--save-metric', choices=('accuracy', 'macro_f1'), default='macro_f1')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--pretrained', action='store_true', help='Use torchvision pretrained weights when available.')
    parser.add_argument('--freeze-backbone', action='store_true', help='Train only the architecture-specific classifier head.')
    train(parser.parse_args())
