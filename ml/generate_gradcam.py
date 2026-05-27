from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project.settings')
    import django

    django.setup()

    from monitoring.services.explainability import generate_visual_explanation

    result = generate_visual_explanation(args.image, backbone=args.arch)
    if not result.overlay_relative_path:
        raise RuntimeError(f'Could not generate explanation. Source: {result.source}')

    output_path = Path('media') / result.overlay_relative_path
    print({'source': result.source, 'output': str(output_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grad-CAM or saliency explanation for a single image.')
    parser.add_argument('--image', required=True)
    parser.add_argument('--arch', default='mobilenet_v3_small')
    main(parser.parse_args())
