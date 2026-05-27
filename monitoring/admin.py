from django.contrib import admin

from .models import ScanSession


@admin.register(ScanSession)
class ScanSessionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'crop_type',
        'source_type',
        'classifier_label',
        'health_status',
        'risk_score',
        'affected_area_pct',
        'created_at',
    )
    list_filter = ('source_type', 'health_status', 'crop_type')
    search_fields = ('crop_type', 'classifier_label', 'notes')
    readonly_fields = (
        'classifier_confidence',
        'affected_area_pct',
        'risk_score',
        'health_status',
        'probability_map',
        'recommendations',
        'created_at',
        'updated_at',
    )
