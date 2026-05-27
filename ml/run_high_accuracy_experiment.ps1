$ErrorActionPreference = "Stop"

$Python = "D:\django\plant\.venv\Scripts\python.exe"
$Dataset = "data\sources\Plant-Diseases-Dataset"
$WeightsDir = "ml\weights_high_accuracy"
$ReportsDir = "ml\reports_high_accuracy"
$Archs = @("mobilenet_v3_small", "resnet18", "efficientnet_b0")

New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

& $Python -u ml\train_all_classifiers.py `
    --dataset-dir $Dataset `
    --output-dir $WeightsDir `
    --archs $Archs `
    --freeze-epochs 1 `
    --epochs 6 `
    --batch-size 32 `
    --image-size 160 `
    --learning-rate 0.0002 `
    --head-learning-rate 0.001 `
    --weight-decay 0.0001 `
    --label-smoothing 0.03 `
    --optimizer adamw `
    --scheduler cosine `
    --augmentation standard `
    --pretrained `
    --log-every 100

foreach ($Arch in $Archs) {
    & $Python -u ml\evaluate_classifier.py `
        --dataset-dir $Dataset `
        --split val `
        --weights-dir $WeightsDir `
        --output-dir $ReportsDir `
        --arch $Arch `
        --batch-size 64
}

& $Python -u ml\summarize_reports.py `
    --weights-dir $WeightsDir `
    --reports-dir $ReportsDir `
    --archs $Archs
