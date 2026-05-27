from __future__ import annotations

from torch import nn
from torchvision import models


SUPPORTED_ARCHS = (
    'mobilenet_v3_small',
    'resnet18',
    'efficientnet_b0',
    'vit_b_16',
)


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    if arch == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if arch == 'vit_b_16':
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if arch == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError(f'Unsupported architecture: {arch}. Choose one of {SUPPORTED_ARCHS}.')


def get_gradcam_target_layer(model, arch: str):
    if arch == 'mobilenet_v3_small':
        return model.features[-1]
    if arch == 'resnet18':
        return model.layer4[-1]
    if arch == 'efficientnet_b0':
        return model.features[-1]
    return None
