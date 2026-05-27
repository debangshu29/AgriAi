# Generated manually for the AgriVision AI starter project.

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='ScanSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(blank=True, max_length=120)),
                ('crop_type', models.CharField(default='Unknown crop', max_length=80)),
                ('source_type', models.CharField(choices=[('leaf', 'Leaf / close-up'), ('drone', 'Drone / aerial'), ('satellite', 'Satellite / remote sensing')], default='leaf', max_length=16)),
                ('image', models.ImageField(upload_to='uploads/')),
                ('processed_image', models.ImageField(blank=True, upload_to='processed/')),
                ('notes', models.TextField(blank=True)),
                ('temperature_c', models.FloatField(default=28.0)),
                ('humidity_pct', models.FloatField(default=60.0)),
                ('classifier_backbone', models.CharField(default='mobilenet_v3_small', max_length=40)),
                ('inference_source', models.CharField(default='heuristic', max_length=40)),
                ('classifier_label', models.CharField(blank=True, max_length=80)),
                ('classifier_confidence', models.FloatField(default=0.0)),
                ('affected_area_pct', models.FloatField(default=0.0)),
                ('vegetation_cover_pct', models.FloatField(default=0.0)),
                ('risk_score', models.FloatField(default=0.0)),
                ('health_status', models.CharField(choices=[('healthy', 'Healthy'), ('watch', 'Watchlist'), ('alert', 'Alert'), ('critical', 'Critical')], default='watch', max_length=20)),
                ('probability_map', models.JSONField(blank=True, default=dict)),
                ('stress_breakdown', models.JSONField(blank=True, default=dict)),
                ('recommendations', models.JSONField(blank=True, default=list)),
                ('narrative', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]
