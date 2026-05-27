from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def classifier_split(source_dir: Path, output_dir: Path, val_ratio: float):
    random.seed(42)
    classes = [item for item in source_dir.iterdir() if item.is_dir()]
    manifest = {}

    for class_dir in classes:
        files = [item for item in class_dir.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES]
        random.shuffle(files)
        split_index = max(1, int(len(files) * (1 - val_ratio)))
        train_files = files[:split_index]
        val_files = files[split_index:]

        for split_name, split_files in [('train', train_files), ('val', val_files)]:
            destination = output_dir / split_name / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for file_path in split_files:
                shutil.copy2(file_path, destination / file_path.name)

        manifest[class_dir.name] = {
            'train_count': len(train_files),
            'val_count': len(val_files),
        }

    return manifest


def _find_matching_mask(mask_dir: Path, image_path: Path):
    exact = mask_dir / image_path.name
    if exact.exists():
        return exact

    for suffix in IMAGE_SUFFIXES:
        candidate = mask_dir / f'{image_path.stem}{suffix}'
        if candidate.exists():
            return candidate

    return None


def segmentation_manifest(source_dir: Path, output_dir: Path, val_ratio: float):
    random.seed(42)
    image_dir = source_dir / 'images'
    mask_dir = source_dir / 'masks'
    images = [item for item in image_dir.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    random.shuffle(images)

    split_index = max(1, int(len(images) * (1 - val_ratio)))
    split_map = {
        'train': images[:split_index],
        'val': images[split_index:],
    }

    manifest = {}
    for split_name, files in split_map.items():
        manifest[split_name] = []
        image_destination = output_dir / split_name / 'images'
        mask_destination = output_dir / split_name / 'masks'
        image_destination.mkdir(parents=True, exist_ok=True)
        mask_destination.mkdir(parents=True, exist_ok=True)

        for image_path in files:
            mask_path = _find_matching_mask(mask_dir, image_path)
            if mask_path is None:
                continue

            shutil.copy2(image_path, image_destination / image_path.name)
            shutil.copy2(mask_path, mask_destination / mask_path.name)
            manifest[split_name].append(
                {
                    'image': str((image_destination / image_path.name).resolve()),
                    'mask': str((mask_destination / mask_path.name).resolve()),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'segmenter_split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare classifier or segmenter datasets for AgriVision AI.')
    parser.add_argument('--task', choices=['classifier', 'segmenter'], required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if args.task == 'classifier':
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = classifier_split(source_dir, output_dir, args.val_ratio)
        (output_dir / 'classifier_split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    else:
        manifest = segmentation_manifest(source_dir, output_dir, args.val_ratio)

    print(json.dumps(manifest, indent=2))
