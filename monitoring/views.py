import json

from django.contrib import messages
from django.db.models import Avg
from django.shortcuts import get_object_or_404, redirect, render

from .forms import ScanUploadForm
from .models import ScanSession
from .services.labels import humanize_label


def home(request):
    if request.method == 'POST':
        form = ScanUploadForm(request.POST, request.FILES)
        if form.is_valid():
            scan = form.save()
            try:
                from .services.pipeline import run_scan_analysis

                run_scan_analysis(scan)
                messages.success(request, 'Scan processed successfully. Review the crop health report below.')
            except Exception as exc:
                messages.warning(
                    request,
                    f'The image was saved, but analysis fell back with a recoverable error: {exc}',
                )
            return redirect('monitoring:scan_detail', pk=scan.pk)
    else:
        form = ScanUploadForm()

    recent_scans = ScanSession.objects.all()[:6]
    summary = ScanSession.objects.aggregate(
        avg_risk=Avg('risk_score'),
        avg_affected=Avg('affected_area_pct'),
        avg_confidence=Avg('classifier_confidence'),
    )
    context = {
        'form': form,
        'recent_scans': recent_scans,
        'summary': {
            'total_scans': ScanSession.objects.count(),
            'critical_scans': ScanSession.objects.filter(health_status=ScanSession.STATUS_CRITICAL).count(),
            'avg_risk': round(summary['avg_risk'] or 0, 1),
            'avg_affected': round(summary['avg_affected'] or 0, 1),
            'avg_confidence': round((summary['avg_confidence'] or 0) * 100, 1),
        },
    }
    return render(request, 'monitoring/home.html', context)


def scan_detail(request, pk):
    scan = get_object_or_404(ScanSession, pk=pk)
    related_scans = ScanSession.objects.exclude(pk=scan.pk)[:4]
    chart_data = {
        'risk': round(scan.risk_score, 1),
        'health': round(scan.health_percent, 1),
        'affected': round(scan.affected_area_pct, 1),
        'vegetation': round(scan.vegetation_cover_pct, 1),
        'confidence': round(scan.classifier_confidence * 100, 1),
        'temperature': round(scan.temperature_c, 1),
        'humidity': round(scan.humidity_pct, 1),
    }
    context = {
        'scan': scan,
        'related_scans': related_scans,
        'chart_json': json.dumps(chart_data),
        'label_display': humanize_label(scan.classifier_label),
    }
    return render(request, 'monitoring/detail.html', context)
