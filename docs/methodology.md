# Methodology

## Study Objective

This project proposes a lightweight multimodal crop health monitoring framework that combines image-based disease classification, affected-region estimation, weather-aware risk scoring, and dashboard-based decision support.

## System Overview

The system accepts a crop image through a Django web interface. The backend performs four sequential operations:

1. Disease classification using a transfer-learning image classifier.
2. Region-level stress localization using a segmentation module.
3. Weather-aware risk estimation using humidity and temperature.
4. Dashboard visualization with prediction confidence, affected-area percentage, risk score, recommendations, and model explanation overlays.

## Classification Models

The research pipeline supports four architectures trained on the same dataset:

- MobileNetV3 Small
- ResNet18
- EfficientNet-B0
- ViT-B/16

Each model replaces its original ImageNet classifier head with a task-specific 38-class crop disease head. Images are resized, normalized with ImageNet statistics, and trained using cross-entropy loss. Evaluation uses a fixed seeded validation subset and reports accuracy, macro F1-score, per-class precision, per-class recall, classification reports, and confusion matrices.

For the CPU-only controlled baseline, MobileNetV3 Small, ResNet18, and EfficientNet-B0 are full fine-tuned. ViT-B/16 is included as a frozen-backbone/head-only transfer-learning baseline because full ViT fine-tuning on CPU was too slow for an interactive experiment.

## Segmentation Model

The application now supports a trained U-Net style segmenter. If `ml/weights/segmenter_unet.pt` is available, the web pipeline uses the trained segmenter to predict stressed regions. If the trained segmenter is absent, the pipeline falls back to an OpenCV HSV thresholding baseline. This creates a clear baseline-vs-learned segmentation comparison for experiments.

Segmentation should be evaluated with:

- Intersection over Union
- Dice coefficient
- Pixel precision
- Pixel recall
- Affected-area error

## Explainability

The framework generates visual explanations using Grad-CAM for CNN backbones. For transformer-style models without convolutional feature maps, the system falls back to input-gradient saliency. Explanation overlays are saved and displayed in the scan report to support interpretability.

## Risk Score

The risk module combines four signals:

- Disease severity prior
- Classifier confidence
- Affected-area percentage
- Weather stress from humidity and temperature

The risk score is a weighted score from 0 to 100 and maps to one of four states: healthy, watchlist, alert, and critical.

## Experimental Design

Recommended experiments:

1. Train MobileNetV3 Small, ResNet18, EfficientNet-B0, and ViT-B/16 under identical splits and preprocessing.
2. Evaluate each model on the internal validation set.
3. Evaluate the best model on at least one external dataset not used during training.
4. Compare OpenCV segmentation against trained U-Net segmentation.
5. Generate Grad-CAM figures for correct and incorrect predictions.
6. Run ablation over classifier-only, classifier plus weather, classifier plus segmentation, and full pipeline.

## Reproducibility

The training scripts use a fixed random seed. Model weights, classes, metadata, histories, reports, confusion matrices, and prediction CSVs are written to `ml/weights`, `ml/weights_research`, `ml/reports`, and `ml/reports_research`.
