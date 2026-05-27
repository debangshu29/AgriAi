from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset


def safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def export(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)

    if args.label_column not in dataset.features:
        raise ValueError(f'Label column {args.label_column} not found in dataset features: {dataset.features}')

    label_feature = dataset.features[args.label_column]
    for index, row in enumerate(dataset):
        if args.max_samples and index >= args.max_samples:
            break

        label_value = row[args.label_column]
        if hasattr(label_feature, 'int2str'):
            label_name = label_feature.int2str(label_value)
        else:
            label_name = str(label_value)

        image = row[args.image_column].convert('RGB')
        class_dir = output_dir / args.output_split / safe_name(label_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f'{index:06d}.jpg', quality=92)

    print({'dataset': args.dataset_id, 'split': args.split, 'output': str(output_dir)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a Hugging Face image dataset into torchvision ImageFolder layout.')
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-split', default='val')
    parser.add_argument('--image-column', default='image')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--cache-dir', default='data/hf_cache')
    export(parser.parse_args())
