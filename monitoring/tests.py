from django.test import TestCase
from django.urls import reverse

from .models import ScanSession
from .services.risk import compute_risk_score


class HomePageTests(TestCase):
    def test_home_page_loads(self):
        response = self.client.get(reverse('monitoring:home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Autonomous crop monitoring')


class ScanSessionModelTests(TestCase):
    def test_health_percent_property(self):
        scan = ScanSession(risk_score=34.5)
        self.assertEqual(scan.health_percent, 65.5)


class RiskServiceTests(TestCase):
    def test_risk_score_returns_valid_status(self):
        result = compute_risk_score(
            label='late_blight',
            confidence=0.84,
            affected_area_pct=32.0,
            temperature_c=30.0,
            humidity_pct=86.0,
        )
        self.assertGreater(result.risk_score, 0)
        self.assertIn(result.health_status, dict(ScanSession.STATUS_CHOICES))
        self.assertTrue(result.recommendations)
