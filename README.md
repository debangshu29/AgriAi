# AgriVision AI

AgriVision AI is a Django-based crop health monitoring prototype. It combines leaf disease classification, affected-region highlighting, explainable-AI overlays (Grad-CAM / saliency), weather-aware risk scoring, and a dashboard that stores and visualizes scan results over time.

The project is designed to work without a physical drone during development. You can use public leaf-image datasets, phone images, or later connect drone/aerial imagery to the same upload and analysis workflow.

## Features

- End-to-end Django web app: upload crop/leaf images, run analysis, and persist results in SQLite.
- Scan dashboard: recent analyses list with health status, risk %, affected area %, and drill-down detail pages.
- Disease classification via PyTorch transfer learning.
- Supported classifier backbones: MobileNetV3 Small, ResNet18, EfficientNet-B0, and ViT-B/16.
- Stress-region segmentation:
  - baseline OpenCV masking/thresholding to estimate affected footprint
  - optional U-Net segmenter when trained weights are available
- Explainability overlays:
  - Grad-CAM for CNN backbones
  - input-gradient saliency fallback for ViT
- Weather-aware risk engine (humidity/temperature + disease severity + affected area).
- Auto-generated recommendations and a probability breakdown (class → score) stored per scan.
- Research tooling: training/evaluation scripts, confusion matrices, classification reports, ablation, and external dataset evaluation.

## Project Structure

- `django_project/` - Django settings and root URLs.
- `monitoring/` - Main app with models, forms, views, and analysis services.
- `templates/` - Dashboard templates.
- `static/` - CSS and JavaScript assets.
- `ml/` - Training, evaluation, Grad-CAM, and research helper scripts.
- `docs/` - Architecture, methodology, commands, and results drafts.
- `data/` - Local dataset staging folder. Dataset contents are intentionally ignored by git.
- `media/` - Local uploads and generated overlays. Runtime contents are ignored by git.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and upload a crop image.

## Environment Variables

For local development, the app runs with safe defaults. For deployment, set:

```powershell
$env:DJANGO_SECRET_KEY="replace-with-a-secret-key"
$env:DJANGO_DEBUG="False"
$env:DJANGO_ALLOWED_HOSTS="your-domain.com,127.0.0.1"
```

## Dataset Layout

Classifier datasets should use ImageFolder format:

```text
data/sources/Plant-Diseases-Dataset/
  train/
    Apple___Apple_scab/
    Apple___healthy/
  val/
    Apple___Apple_scab/
    Apple___healthy/
```

Segmentation datasets should use paired images and masks:

```text
data/segmenter_ready/
  train/images/
  train/masks/
  val/images/
  val/masks/
```

Large datasets are not committed to the repository. Download them locally before training.

## Training

Train one classifier:

```powershell
python -u ml\train_classifier.py --dataset-dir data\sources\Plant-Diseases-Dataset --output-dir ml\weights --arch mobilenet_v3_small --freeze-epochs 1 --epochs 6 --batch-size 32 --image-size 160 --pretrained
```

Train the main CNN comparison run:

```powershell
powershell -ExecutionPolicy Bypass -File ml\run_high_accuracy_experiment.ps1
```

Evaluate one classifier:

```powershell
python -u ml\evaluate_classifier.py --dataset-dir data\sources\Plant-Diseases-Dataset --split val --weights-dir ml\weights --output-dir ml\reports --arch mobilenet_v3_small
```

Train the U-Net segmenter:

```powershell
python -u ml\train_segmenter.py --dataset-dir data\segmenter_ready --epochs 20 --batch-size 8 --image-size 256
```

## Research Status

The current repository is ready for GitHub as a project prototype. For publication, the remaining work is mainly experimental: final full-dataset training, external dataset testing, real segmentation masks and Dice/IoU scores, and a stronger ablation study. See `docs/results.md` and `docs/research_commands.md`.

## Roadmap (Planned Enhancements)

- Geospatial heatmaps and field trend analysis (map view) for drone/satellite workflows.
- Async processing (Celery / Django Q) to queue heavier model inference.
- REST API endpoints for mobile apps and drone ingestion.
- Integrations for advisory content (for example, government schemes/subsidy lookups) once requirements and data sources are finalized.

## GitHub Notes

The repository intentionally ignores:

- downloaded datasets
- Hugging Face caches
- model checkpoints
- generated reports
- logs and process ids
- uploaded media
- local database files
- virtual environments and IDE settings

Keep only source code, docs, scripts, and lightweight placeholders in git.
