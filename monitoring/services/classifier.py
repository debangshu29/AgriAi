from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.conf import settings
from PIL import Image

try:
    import torch
    from torchvision import transforms
    from ml.model_registry import build_model
except Exception:
    torch = None
    transforms = None
    build_model = None


DEFAULT_CLASSES = [
    'healthy',
    'bacterial_spot',
    'early_blight',
    'late_blight',
    'leaf_mold',
    'mosaic_virus',
    'nutrient_stress',
    'pest_pressure',
]


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    backbone: str
    source: str
    notes: str


def _load_trained_model(backbone: str):
    if torch is None or transforms is None:
        return None, DEFAULT_CLASSES, {'image_size': 224}

    weights_path = Path(settings.MODEL_DIR) / f'{backbone}_classifier.pt'
    classes_path = Path(settings.MODEL_DIR) / f'{backbone}_classes.json'
    metadata_path = Path(settings.MODEL_DIR) / f'{backbone}_metadata.json'
    if not weights_path.exists():
        return None, DEFAULT_CLASSES, {'image_size': 224}

    classes = DEFAULT_CLASSES
    if classes_path.exists():
        classes = json.loads(classes_path.read_text(encoding='utf-8'))

    metadata = {'image_size': 224}
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding='utf-8')))

    if build_model is None:
        return None, classes, metadata

    model = build_model(backbone, len(classes), pretrained=False)
    if model is None:
        return None, classes, metadata

    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, classes, metadata


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values()) or 1
    return {key: round(value / total, 4) for key, value in probabilities.items()}


def _heuristic_leaf_classifier(image_array: np.ndarray) -> ClassificationResult:
    red_mean = float(image_array[:, :, 0].mean())
    green_mean = float(image_array[:, :, 1].mean())
    blue_mean = float(image_array[:, :, 2].mean())
    brightness = float(image_array.mean())
    spread = float(image_array.std())

    probabilities = {
        'healthy': 0.20,
        'bacterial_spot': 0.10,
        'early_blight': 0.10,
        'late_blight': 0.10,
        'leaf_mold': 0.10,
        'mosaic_virus': 0.10,
        'nutrient_stress': 0.15,
        'pest_pressure': 0.15,
    }

    if green_mean > red_mean + 12 and green_mean > blue_mean + 8 and spread < 58:
        label = 'healthy'
        confidence = 0.78
        probabilities['healthy'] = 0.78
    elif red_mean > green_mean + 10 and brightness < 135:
        label = 'early_blight'
        confidence = 0.66
        probabilities['early_blight'] = 0.66
    elif green_mean < 110 and red_mean > 120 and blue_mean < 110:
        label = 'nutrient_stress'
        confidence = 0.63
        probabilities['nutrient_stress'] = 0.63
    elif spread > 70:
        label = 'pest_pressure'
        confidence = 0.61
        probabilities['pest_pressure'] = 0.61
    else:
        label = 'leaf_mold'
        confidence = 0.57
        probabilities['leaf_mold'] = 0.57

    normalized = _normalize_probabilities(probabilities)
    return ClassificationResult(
        label=label,
        confidence=confidence,
        probabilities=normalized,
        backbone='mobilenet_v3_small',
        source='heuristic',
        notes='Fallback color-statistics model used because trained weights were not available.',
    )


def classify_crop_health(image_path: str, backbone: str = 'mobilenet_v3_small') -> ClassificationResult:
    image = Image.open(image_path).convert('RGB')
    image_array = np.asarray(image)

    model, classes, metadata = _load_trained_model(backbone)
    if model is None or transforms is None or torch is None:
        return _heuristic_leaf_classifier(image_array)

    image_size = int(metadata.get('image_size', 224))
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

    probability_map = {label: round(float(score), 4) for label, score in zip(classes, probabilities)}
    top_label = max(probability_map, key=probability_map.get)
    return ClassificationResult(
        label=top_label,
        confidence=probability_map[top_label],
        probabilities=probability_map,
        backbone=backbone,
        source='trained_model',
        notes=f'Predicted with a fine-tuned transfer learning backbone at {image_size}px input size.',
    )
