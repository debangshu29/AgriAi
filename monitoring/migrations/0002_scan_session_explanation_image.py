from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('monitoring', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='scansession',
            name='explanation_image',
            field=models.ImageField(blank=True, upload_to='explanations/'),
        ),
    ]
