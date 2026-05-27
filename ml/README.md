# ML folder

## Recommended dataset layouts

### Classifier
- `data/classifier_raw/<class_name>/*.jpg`
- Run `python ml/prepare_datasets.py --task classifier --source-dir data/classifier_raw --output-dir data/classifier_ready`
- Train with `python ml/train_classifier.py --dataset-dir data/classifier_ready --arch mobilenet_v3_small --pretrained`

### Segmenter
- `data/segmenter_raw/images/*.jpg`
- `data/segmenter_raw/masks/*.png`
- Run `python ml/prepare_datasets.py --task segmenter --source-dir data/segmenter_raw --output-dir data/segmenter_ready`
- The prep script now creates:
  - `data/segmenter_ready/train/images`
  - `data/segmenter_ready/train/masks`
  - `data/segmenter_ready/val/images`
  - `data/segmenter_ready/val/masks`
- Train with `python ml/train_segmenter.py --dataset-dir data/segmenter_ready`

## Notes
- The web dashboard can run before training, but it will use heuristic classification and threshold-based segmentation until weights are placed in `ml/weights/`.
- The expected classifier weight files are `mobilenet_v3_small_classifier.pt` or `resnet18_classifier.pt` plus the matching `*_classes.json` metadata file.
