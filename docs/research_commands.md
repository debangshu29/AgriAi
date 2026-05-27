# Research Commands

Run all commands from the project root.

## Train Four Classifiers

Fast controlled run:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\train_all_classifiers.py --dataset-dir data\sources\Plant-Diseases-Dataset --epochs 1 --batch-size 64 --image-size 128 --max-train-samples 20000 --max-val-samples 4000 --pretrained --log-every 50
```

Higher-accuracy CPU overnight run for the CNN models:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\train_all_classifiers.py --dataset-dir data\sources\Plant-Diseases-Dataset --output-dir ml\weights_high_accuracy --archs mobilenet_v3_small resnet18 efficientnet_b0 --freeze-epochs 1 --epochs 6 --batch-size 32 --image-size 160 --learning-rate 0.0002 --head-learning-rate 0.001 --weight-decay 0.0001 --label-smoothing 0.03 --optimizer adamw --scheduler cosine --augmentation standard --pretrained --log-every 100
```

GPU paper-quality run for all four models:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\train_all_classifiers.py --dataset-dir data\sources\Plant-Diseases-Dataset --output-dir ml\weights_paper --archs mobilenet_v3_small resnet18 efficientnet_b0 vit_b_16 --freeze-epochs 2 --epochs 12 --batch-size 32 --image-size 224 --learning-rate 0.00015 --head-learning-rate 0.001 --weight-decay 0.0001 --label-smoothing 0.05 --optimizer adamw --scheduler cosine --augmentation standard --pretrained --log-every 100
```

## Evaluate A Classifier

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\evaluate_classifier.py --dataset-dir data\sources\Plant-Diseases-Dataset --split val --arch mobilenet_v3_small --output-dir ml\reports
```

Outputs:

- summary JSON
- classification report JSON
- predictions CSV
- confusion matrix CSV
- confusion matrix PNG

## Generate Grad-CAM

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\generate_gradcam.py --image data\sources\Plant-Diseases-Dataset\val\Tomato___Early_blight\example.JPG --arch mobilenet_v3_small
```

## Train U-Net Segmenter

Prepare data in this structure:

- `data/segmenter_ready/train/images`
- `data/segmenter_ready/train/masks`
- `data/segmenter_ready/val/images`
- `data/segmenter_ready/val/masks`

Then run:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\train_segmenter.py --dataset-dir data\segmenter_ready --epochs 20 --batch-size 8 --image-size 256
```

The Django app will automatically use `ml/weights/segmenter_unet.pt` if it exists.

## External Dataset Evaluation

Put an external dataset in ImageFolder format:

- `data/external_dataset/val/<class_name>/*.jpg`

Or export a Hugging Face dataset:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\export_hf_imagefolder.py --dataset-id GVJahnavi/PlantVillage_dataset --split test --output-dir data\external_plantvillage_hf --output-split val --max-samples 4000
```

Then run:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\evaluate_classifier.py --dataset-dir data\external_dataset --split val --arch mobilenet_v3_small --output-dir ml\reports\external
```

## Ablation Study

First produce predictions:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\evaluate_classifier.py --dataset-dir data\sources\Plant-Diseases-Dataset --split val --arch mobilenet_v3_small --output-dir ml\reports
```

Then run:

```powershell
D:\django\plant\.venv\Scripts\python.exe -u ml\run_ablation.py --predictions-csv ml\reports\mobilenet_v3_small_val_predictions.csv --output-dir ml\reports\ablation
```
