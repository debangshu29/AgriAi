document.addEventListener('DOMContentLoaded', () => {
    const scanDataNode = document.getElementById('scan-data');
    if (!scanDataNode) {
        return;
    }

    const data = JSON.parse(scanDataNode.textContent);
    drawRiskMeter(data);
    fillBars(data);
});

function drawRiskMeter(data) {
    const canvas = document.getElementById('risk-meter');
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 78;
    const startAngle = Math.PI * 0.8;
    const endAngle = Math.PI * 2.2;
    const progress = Math.min(Math.max(data.risk / 100, 0), 1);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';

    ctx.beginPath();
    ctx.strokeStyle = 'rgba(21, 49, 36, 0.12)';
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.stroke();

    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#2e6a3b');
    gradient.addColorStop(0.5, '#e0a74a');
    gradient.addColorStop(1, '#a43824');

    ctx.beginPath();
    ctx.strokeStyle = gradient;
    ctx.arc(centerX, centerY, radius, startAngle, startAngle + (endAngle - startAngle) * progress);
    ctx.stroke();

    ctx.fillStyle = '#153124';
    ctx.textAlign = 'center';
    ctx.font = '700 42px Trebuchet MS';
    ctx.fillText(`${Math.round(data.risk)}%`, centerX, centerY + 10);
    ctx.font = '600 14px Trebuchet MS';
    ctx.fillStyle = '#5f6e61';
    ctx.fillText('Risk Score', centerX, centerY + 36);
}

function fillBars(data) {
    const normalization = {
        confidence: 100,
        affected: 100,
        humidity: 100,
        temperature: 45,
    };

    document.querySelectorAll('[data-bar]').forEach((element) => {
        const key = element.dataset.bar;
        const value = Math.min((data[key] / normalization[key]) * 100, 100);
        const fill = element.querySelector('.bar-fill');
        if (fill) {
            fill.style.width = `${Math.max(value, 4)}%`;
        }
    });
}
