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
    from torchvision import transforms
    from ml.model_registry import get_gradcam_target_layer
except Exception:
    torch = None
    transforms = None
    get_gradcam_target_layer = None

from .classifier import _load_trained_model


@dataclass
class ExplanationResult:
    overlay_relative_path: str
    source: str


def _save_overlay(image: np.ndarray, heatmap: np.ndarray) -> str:
    if cv2 is not None:
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        colored = np.zeros_like(image)
        colored[:, :, 0] = np.uint8(255 * heatmap)
        colored[:, :, 1] = np.uint8(120 * (1 - heatmap))

    blended = np.uint8(0.62 * image + 0.38 * colored)
    filename = f'{uuid4().hex}_gradcam.png'
    relative_path = Path('explanations') / filename
    output_path = Path(settings.MEDIA_ROOT) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)
    return relative_path.as_posix()


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    max_value = float(heatmap.max())
    if max_value <= 0:
        return np.zeros_like(heatmap)
    return heatmap / max_value


def generate_visual_explanation(image_path: str, backbone: str = 'mobilenet_v3_small') -> ExplanationResult:
    if torch is None or transforms is None or get_gradcam_target_layer is None:
        return ExplanationResult('', 'explainability-unavailable')

    model, classes, metadata = _load_trained_model(backbone)
    if model is None:
        return ExplanationResult('', 'no-trained-model')

    target_layer = get_gradcam_target_layer(model, backbone)
    if target_layer is None:
        return _generate_input_saliency(image_path, model, metadata)

    image_size = int(metadata.get('image_size', 224))
    image = Image.open(image_path).convert('RGB')
    original = np.asarray(image)
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image).unsqueeze(0)

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if not activations or not gradients:
        return ExplanationResult('', 'gradcam-empty')

    activation = activations[0][0]
    gradient = gradients[0][0]
    weights = gradient.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * activation, dim=0).cpu().numpy()
    cam = _normalize_heatmap(cam)
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BILINEAR)
    heatmap = np.asarray(cam_image).astype(np.float32) / 255.0
    return ExplanationResult(_save_overlay(original, heatmap), 'gradcam')


def _generate_input_saliency(image_path: str, model, metadata: dict) -> ExplanationResult:
    image_size = int(metadata.get('image_size', 224))
    image = Image.open(image_path).convert('RGB')
    original = np.asarray(image)
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image).unsqueeze(0)
    tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    class_index = int(logits.argmax(dim=1).item())
    logits[:, class_index].backward()

    saliency = tensor.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
    saliency = _normalize_heatmap(saliency)
    saliency_image = Image.fromarray(np.uint8(saliency * 255)).resize(image.size, Image.Resampling.BILINEAR)
    heatmap = np.asarray(saliency_image).astype(np.float32) / 255.0
    return ExplanationResult(_save_overlay(original, heatmap), 'input-gradient-saliency')
