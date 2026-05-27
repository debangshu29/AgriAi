from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from django.conf import settings
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


@dataclass
class SegmentationResult:
    affected_area_pct: float
    vegetation_cover_pct: float
    stress_breakdown: dict[str, float]
    overlay_relative_path: str
    source: str


class DoubleConv(nn.Module if nn is not None else object):
    def __init__(self, in_channels, out_channels):
        if nn is None:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module if nn is not None else object):
    def __init__(self):
        if nn is None:
            return
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


def _save_overlay(base_image: np.ndarray, stress_mask: np.ndarray) -> str:
    overlay = base_image.copy()
    overlay[stress_mask] = [255, 80, 70]
    blended = (0.65 * base_image + 0.35 * overlay).astype(np.uint8)
    filename = f'{uuid4().hex}_overlay.png'
    relative_path = Path('processed') / filename
    output_path = Path(settings.MEDIA_ROOT) / relative_path
    Image.fromarray(blended).save(output_path)
    return relative_path.as_posix()


def _load_unet_segmenter():
    if torch is None or nn is None:
        return None
    weights_path = Path(settings.MODEL_DIR) / 'segmenter_unet.pt'
    if not weights_path.exists():
        return None
    model = UNetSmall()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def _segment_with_unet(image_path: str) -> SegmentationResult | None:
    model = _load_unet_segmenter()
    if model is None:
        return None

    image = Image.open(image_path).convert('RGB')
    rgb = np.asarray(image)
    resized = image.resize((256, 256))
    tensor = torch.from_numpy(np.asarray(resized).astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    probability_image = Image.fromarray(np.uint8(probability * 255)).resize(image.size, Image.Resampling.BILINEAR)
    probability_full = np.asarray(probability_image).astype('float32') / 255.0
    stress_mask = probability_full >= 0.5

    stress_pixels = int(stress_mask.sum())
    total_pixels = int(rgb.shape[0] * rgb.shape[1]) or 1
    affected_area_pct = round(stress_pixels / total_pixels * 100, 2)
    vegetation_cover_pct = affected_area_pct
    stress_breakdown = {
        'stress_pixels': stress_pixels,
        'total_pixels': total_pixels,
        'mean_stress_probability': round(float(probability_full.mean()), 4),
    }
    return SegmentationResult(
        affected_area_pct=affected_area_pct,
        vegetation_cover_pct=vegetation_cover_pct,
        stress_breakdown=stress_breakdown,
        overlay_relative_path=_save_overlay(rgb, stress_mask),
        source='unet-trained',
    )


def segment_stress_regions(image_path: str) -> SegmentationResult:
    trained_result = _segment_with_unet(image_path)
    if trained_result is not None:
        return trained_result

    image = Image.open(image_path).convert('RGB')
    rgb = np.asarray(image)

    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        vegetation_mask = cv2.inRange(hsv, (25, 35, 25), (95, 255, 255)) > 0
        yellow_mask = cv2.inRange(hsv, (10, 40, 40), (40, 255, 255)) > 0
        brown_mask = cv2.inRange(hsv, (0, 50, 20), (18, 255, 180)) > 0
        stress_mask = np.logical_and(vegetation_mask, np.logical_or(yellow_mask, brown_mask))
        stress_mask = cv2.medianBlur(stress_mask.astype(np.uint8) * 255, 5) > 0
        source = 'opencv-thresholding'
    else:
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        vegetation_mask = (green > red * 0.85) & (green > blue * 1.05)
        stress_mask = vegetation_mask & ((red > green * 0.9) | (blue < 90))
        source = 'numpy-thresholding'

    vegetation_pixels = int(vegetation_mask.sum())
    stress_pixels = int(stress_mask.sum())
    total_pixels = int(rgb.shape[0] * rgb.shape[1]) or 1
    vegetation_cover_pct = round(vegetation_pixels / total_pixels * 100, 2)
    affected_area_pct = round(stress_pixels / total_pixels * 100, 2)

    stress_breakdown = {
        'vegetation_pixels': vegetation_pixels,
        'stress_pixels': stress_pixels,
        'total_pixels': total_pixels,
    }
    overlay_relative_path = _save_overlay(rgb, stress_mask)
    return SegmentationResult(
        affected_area_pct=affected_area_pct,
        vegetation_cover_pct=vegetation_cover_pct,
        stress_breakdown=stress_breakdown,
        overlay_relative_path=overlay_relative_path,
        source=source,
    )
