from django import forms

from .models import ScanSession


class ScanUploadForm(forms.ModelForm):
    class Meta:
        model = ScanSession
        fields = (
            'title',
            'crop_type',
            'source_type',
            'image',
            'temperature_c',
            'humidity_pct',
            'notes',
        )
        widgets = {
            'title': forms.TextInput(
                attrs={
                    'placeholder': 'North field stress scan',
                }
            ),
            'crop_type': forms.TextInput(
                attrs={
                    'placeholder': 'Tomato, potato, rice, wheat...',
                }
            ),
            'source_type': forms.Select(),
            'image': forms.ClearableFileInput(
                attrs={
                    'accept': 'image/*',
                }
            ),
            'temperature_c': forms.NumberInput(
                attrs={
                    'step': '0.1',
                    'placeholder': '29.5',
                }
            ),
            'humidity_pct': forms.NumberInput(
                attrs={
                    'step': '0.1',
                    'placeholder': '78.0',
                }
            ),
            'notes': forms.Textarea(
                attrs={
                    'rows': 4,
                    'placeholder': 'Optional field notes, growth stage, visible symptoms...',
                }
            ),
        }

    def clean_humidity_pct(self):
        humidity = self.cleaned_data['humidity_pct']
        if humidity is not None and not 0 <= humidity <= 100:
            raise forms.ValidationError('Humidity must be between 0 and 100 percent.')
        return humidity
