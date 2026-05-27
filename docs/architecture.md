# Architecture overview

## Pipeline
1. User uploads a leaf image or aerial crop image in the Django dashboard.
2. `monitoring.services.classifier` predicts the most likely crop health class.
3. `monitoring.services.segmentation` estimates stressed pixels and generates an overlay image.
4. `monitoring.services.risk` combines disease severity, humidity, temperature, and affected area into a final risk score.
5. The detail page renders metrics, probabilities, recommendations, and visual evidence.

## Design choices
- The project separates classification and segmentation because leaf disease datasets and aerial drone datasets usually belong to different domains.
- The dashboard works before full model training by using deterministic heuristics as a baseline.
- PyTorch was chosen for the training scripts because it is student-friendly, flexible, and pairs well with transfer learning.
- Django keeps the project easy to demo, document, and expand into admin workflows later.

## Suggested future upgrades
- Add Celery or Django Q for asynchronous image processing.
- Add a REST API for mobile or drone ingestion.
- Replace the baseline segmenter with DeepLabV3 or a stronger U-Net variant.
- Add geospatial heatmaps and longitudinal field trend analysis.
