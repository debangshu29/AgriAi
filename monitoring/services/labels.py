from __future__ import annotations

import re


def slugify_label(label: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def humanize_label(label: str) -> str:
    if not label:
        return 'Pending analysis'
    return re.sub(r'[_\s]+', ' ', label).strip().title()


def infer_disease_key(label: str) -> str:
    slug = slugify_label(label)
    if 'healthy' in slug and all(keyword not in slug for keyword in ['blight', 'spot', 'rot', 'mildew', 'rust', 'virus', 'scab', 'mold']):
        return 'healthy'

    keyword_map = [
        'late_blight',
        'early_blight',
        'bacterial_spot',
        'yellow_leaf_curl_virus',
        'mosaic_virus',
        'powdery_mildew',
        'leaf_mold',
        'black_rot',
        'cedar_apple_rust',
        'frog_eye_leaf_spot',
        'leaf_scorch',
        'leaf_spot',
        'target_spot',
        'spider_mite',
        'spider_mites',
        'scab',
        'rust',
        'greening',
        'esca',
        'complex',
        'multi_disease',
        'mildew',
        'virus',
        'rot',
        'spot',
    ]
    for keyword in keyword_map:
        if keyword in slug:
            return keyword
    if 'healthy' in slug:
        return 'healthy'
    return slug
