from django.db import models


class ScanSession(models.Model):
    SOURCE_LEAF = 'leaf'
    SOURCE_DRONE = 'drone'
    SOURCE_SATELLITE = 'satellite'
    SOURCE_CHOICES = (
        (SOURCE_LEAF, 'Leaf / close-up'),
        (SOURCE_DRONE, 'Drone / aerial'),
        (SOURCE_SATELLITE, 'Satellite / remote sensing'),
    )

    STATUS_HEALTHY = 'healthy'
    STATUS_WATCH = 'watch'
    STATUS_ALERT = 'alert'
    STATUS_CRITICAL = 'critical'
    STATUS_CHOICES = (
        (STATUS_HEALTHY, 'Healthy'),
        (STATUS_WATCH, 'Watchlist'),
        (STATUS_ALERT, 'Alert'),
        (STATUS_CRITICAL, 'Critical'),
    )

    title = models.CharField(max_length=120, blank=True)
    crop_type = models.CharField(max_length=80, default='Unknown crop')
    source_type = models.CharField(max_length=16, choices=SOURCE_CHOICES, default=SOURCE_LEAF)
    image = models.ImageField(upload_to='uploads/')
    processed_image = models.ImageField(upload_to='processed/', blank=True)
    explanation_image = models.ImageField(upload_to='explanations/', blank=True)
    notes = models.TextField(blank=True)

    temperature_c = models.FloatField(default=28.0)
    humidity_pct = models.FloatField(default=60.0)

    classifier_backbone = models.CharField(max_length=40, default='mobilenet_v3_small')
    inference_source = models.CharField(max_length=40, default='heuristic')
    classifier_label = models.CharField(max_length=80, blank=True)
    classifier_confidence = models.FloatField(default=0.0)
    affected_area_pct = models.FloatField(default=0.0)
    vegetation_cover_pct = models.FloatField(default=0.0)
    risk_score = models.FloatField(default=0.0)
    health_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_WATCH)

    probability_map = models.JSONField(default=dict, blank=True)
    stress_breakdown = models.JSONField(default=dict, blank=True)
    recommendations = models.JSONField(default=list, blank=True)
    narrative = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title or f'{self.crop_type} scan #{self.pk}'

    @property
    def health_percent(self):
        return max(0.0, round(100 - self.risk_score, 1))
