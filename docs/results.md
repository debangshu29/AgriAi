# Results Draft

## Current Implemented Result

The current deployed classifier is MobileNetV3 Small trained with transfer learning.

| Model | Train Samples | Validation Samples | Epochs | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Small | 20,000 | 4,000 | 1 | 0.9460 | 0.9465 |

This result is suitable as a prototype baseline, not as the final paper result. The paper should include longer training runs and direct comparison against ResNet18, EfficientNet-B0, and ViT-B/16.

## Controlled Four-Model Baseline

The following CPU baseline used the same Plant-Diseases-Dataset split, the same seeded random subset, ImageNet preprocessing, input size 224, one epoch, 2,000 training images, and 800 validation images. MobileNetV3 Small, ResNet18, and EfficientNet-B0 were full fine-tuned. ViT-B/16 was trained as a frozen-backbone/head-only transfer-learning baseline because full ViT fine-tuning was not practical on the available CPU-only machine.

| Model | Training Mode | Total Params | Trainable Params | Train Loss | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Small | Full fine-tune | 1,556,806 | 1,556,806 | 1.8277 | 0.72625 | 0.82374 | 0.74032 | 0.72543 |
| ResNet18 | Full fine-tune | 11,196,006 | 11,196,006 | 1.1689 | 0.88000 | 0.89720 | 0.88262 | 0.87358 |
| EfficientNet-B0 | Full fine-tune | 4,056,226 | 4,056,226 | 1.9204 | 0.88625 | 0.90927 | 0.88863 | 0.88112 |
| ViT-B/16 | Frozen backbone/head only | 85,827,878 | 29,222 | 2.8383 | 0.69000 | 0.73300 | 0.68183 | 0.66584 |

Generated artifacts:

- `ml/reports_research/model_comparison.csv`
- `ml/reports_research/model_comparison.json`
- Per-model classification reports, prediction CSVs, confusion matrices, and confusion-matrix PNGs in `ml/reports_research/`

Interpretation: EfficientNet-B0 achieved the best macro F1 in this fast controlled run, followed closely by ResNet18. ViT-B/16 underperformed in the head-only CPU baseline, which is expected because transformer fine-tuning generally needs more compute, longer training, and often larger datasets.

## Tables To Add After Full Experiments

### External Dataset Generalization

| Model | External Dataset | Accuracy | Macro F1 | Main Failure Classes |
| --- | --- | ---: | ---: | --- |
| Best model | TBD | TBD | TBD | TBD |

### Segmentation Comparison

| Method | IoU | Dice | Pixel Precision | Pixel Recall | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| OpenCV HSV baseline | TBD | TBD | TBD | TBD | Rule-based |
| U-Net | TBD | TBD | TBD | TBD | Learned |

### Ablation Study

| Configuration | Output Type | Metric |
| --- | --- | ---: |
| Classifier only | Disease confidence | TBD |
| Classifier + weather | Risk score | TBD |
| Classifier + segmentation | Risk score | TBD |
| Classifier + segmentation + weather | Risk score | TBD |

## Discussion Points

- MobileNetV3 is expected to be fast and lightweight, suitable for edge deployment.
- EfficientNet-B0 may improve accuracy with moderate compute cost.
- ResNet18 provides a strong classical CNN baseline.
- ViT-B/16 may require more data and compute but can provide a transformer baseline.
- Grad-CAM improves interpretability by showing whether the model focuses on lesion regions.
- U-Net segmentation makes affected-area estimation more scientifically defensible than HSV thresholding.
