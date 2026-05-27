from __future__ import annotations

from monitoring.models import ScanSession

from .classifier import classify_crop_health
from .explainability import generate_visual_explanation
from .risk import compute_risk_score
from .segmentation import segment_stress_regions


def run_scan_analysis(scan: ScanSession) -> ScanSession:
    classification = classify_crop_health(scan.image.path, backbone=scan.classifier_backbone)
    segmentation = segment_stress_regions(scan.image.path)
    explanation = generate_visual_explanation(scan.image.path, backbone=scan.classifier_backbone)
    risk = compute_risk_score(
        label=classification.label,
        confidence=classification.confidence,
        affected_area_pct=segmentation.affected_area_pct,
        temperature_c=scan.temperature_c,
        humidity_pct=scan.humidity_pct,
    )

    scan.classifier_label = classification.label
    scan.classifier_confidence = classification.confidence
    scan.classifier_backbone = classification.backbone
    scan.inference_source = f'{classification.source} + {segmentation.source}'
    scan.affected_area_pct = segmentation.affected_area_pct
    scan.vegetation_cover_pct = segmentation.vegetation_cover_pct
    scan.processed_image = segmentation.overlay_relative_path
    scan.explanation_image = explanation.overlay_relative_path
    scan.risk_score = risk.risk_score
    scan.health_status = risk.health_status
    scan.probability_map = classification.probabilities
    scan.stress_breakdown = segmentation.stress_breakdown
    scan.recommendations = risk.recommendations
    scan.narrative = f'{classification.notes} Explanation source: {explanation.source}. {risk.narrative}'
    scan.save()
    return scan
