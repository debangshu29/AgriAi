from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int):
        self.samples = sorted(image_dir.glob('*'))
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert('L').resize((self.image_size, self.image_size))

        image_array = np.asarray(image).astype('float32') / 255.0
        mask_array = (np.asarray(mask).astype('float32') / 255.0 > 0.5).astype('float32')

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(64, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bridge = self.bridge(self.pool2(d2))
        u1 = self.up1(bridge)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.dec1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.dec2(u2)
        return self.out(u2)


def dice_score(logits, masks):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean().item()


def iou_score(logits, masks):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def evaluate(model, loader, device):
    model.eval()
    dice_values = []
    iou_values = []
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            dice_values.append(dice_score(logits, masks))
            iou_values.append(iou_score(logits, masks))
    return {
        'dice': sum(dice_values) / max(len(dice_values), 1),
        'iou': sum(iou_values) / max(len(iou_values), 1),
    }


def train(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SegmentationDataset(dataset_dir / 'train' / 'images', dataset_dir / 'train' / 'masks', args.image_size)
    val_dataset = SegmentationDataset(dataset_dir / 'val' / 'images', dataset_dir / 'val' / 'masks', args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetSmall().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        metrics = evaluate(model, val_loader, device)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_record = {
            'epoch': epoch,
            'loss': round(epoch_loss, 4),
            'dice': round(metrics['dice'], 4),
            'iou': round(metrics['iou'], 4),
        }
        history.append(epoch_record)
        print(epoch_record)

        if metrics['dice'] >= best_dice:
            best_dice = metrics['dice']
            torch.save(model.state_dict(), output_dir / 'segmenter_unet.pt')

    (output_dir / 'segmenter_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Best dice: {best_dice:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a small U-Net style crop stress segmenter.')
    parser.add_argument('--dataset-dir', required=True, help='Path with train/ and val/ folders containing images/ and masks/.')
    parser.add_argument('--output-dir', default='ml/weights', help='Directory for weights and training logs.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    train(parser.parse_args())
