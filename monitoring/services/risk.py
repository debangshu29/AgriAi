from __future__ import annotations

from dataclasses import dataclass

from monitoring.models import ScanSession
from monitoring.services.labels import humanize_label, infer_disease_key


SEVERITY_MAP = {
    'healthy': 0.05,
    'nutrient_stress': 0.45,
    'pest_pressure': 0.52,
    'bacterial_spot': 0.58,
    'leaf_mold': 0.60,
    'early_blight': 0.67,
    'late_blight': 0.76,
    'mosaic_virus': 0.72,
    'yellow_leaf_curl_virus': 0.76,
    'powdery_mildew': 0.57,
    'rust': 0.56,
    'scab': 0.57,
    'black_rot': 0.64,
    'frog_eye_leaf_spot': 0.60,
    'leaf_scorch': 0.52,
    'leaf_spot': 0.55,
    'target_spot': 0.59,
    'greening': 0.78,
    'esca': 0.68,
    'complex': 0.67,
    'multi_disease': 0.70,
    'spot': 0.54,
    'rot': 0.61,
    'virus': 0.70,
    'mildew': 0.55,
}


@dataclass
class RiskResult:
    risk_score: float
    health_status: str
    recommendations: list[str]
    narrative: str


def compute_risk_score(
    label: str,
    confidence: float,
    affected_area_pct: float,
    temperature_c: float,
    humidity_pct: float,
) -> RiskResult:
    disease_key = infer_disease_key(label)
    disease_pressure = SEVERITY_MAP.get(disease_key, 0.40) * max(confidence, 0.35)
    humidity_pressure = min(max((humidity_pct - 55) / 40, 0), 1)
    heat_pressure = min(abs(temperature_c - 27) / 15, 1)
    area_pressure = min(affected_area_pct / 45, 1)

    risk_score = round(
        100
        * (
            disease_pressure * 0.45
            + humidity_pressure * 0.20
            + heat_pressure * 0.10
            + area_pressure * 0.25
        ),
        2,
    )

    if risk_score < 28:
        health_status = ScanSession.STATUS_HEALTHY
    elif risk_score < 48:
        health_status = ScanSession.STATUS_WATCH
    elif risk_score < 70:
        health_status = ScanSession.STATUS_ALERT
    else:
        health_status = ScanSession.STATUS_CRITICAL

    recommendations = _build_recommendations(label, disease_key, health_status, humidity_pct, temperature_c, affected_area_pct)
    narrative = (
        f'Risk is driven by {humanize_label(label)} signals, {affected_area_pct:.1f}% stressed area, '
        f'{humidity_pct:.1f}% humidity, and {temperature_c:.1f}C field temperature.'
    )
    return RiskResult(
        risk_score=risk_score,
        health_status=health_status,
        recommendations=recommendations,
        narrative=narrative,
    )


def _build_recommendations(
    label: str,
    disease_key: str,
    health_status: str,
    humidity_pct: float,
    temperature_c: float,
    affected_area_pct: float,
) -> list[str]:
    display_label = humanize_label(label)
    recommendations = [
        'Re-scan the same zone within 48 hours to confirm whether symptoms are expanding.',
        'Inspect the highlighted region on-ground and capture 3 to 5 close-up leaf images for verification.',
    ]

    if humidity_pct >= 75:
        recommendations.append('High humidity favors fungal spread, so improve canopy airflow and monitor irrigation timing.')
    if temperature_c >= 32:
        recommendations.append('Heat stress is elevated. Check soil moisture, mulch coverage, and irrigation uniformity.')
    if disease_key != 'healthy':
        recommendations.append(f'Prepare a targeted intervention plan for suspected {display_label} symptoms.')
    if affected_area_pct >= 20:
        recommendations.append('The affected footprint is substantial. Prioritize scouting this block before field-wide treatment.')
    if health_status == ScanSession.STATUS_CRITICAL:
        recommendations.append('Escalate to agronomy review today and isolate hotspot zones if operationally possible.')

    return recommendations
