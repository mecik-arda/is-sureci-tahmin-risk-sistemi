/* Dashboard data loader (S23) */
(function () {
    'use strict';

    function setText(id, value) {
        var el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function formatPercent(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        return (value * 100).toFixed(1).replace('.', ',') + '%';
    }

    function formatHours(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        var n = Number(value);
        return n.toFixed(1).replace('.', ',') + ' saat';
    }

    function renderKpis(data) {
        var pred = data.prediction_kpis || {};
        var actual = data.actual_kpis || {};
        var hasPred = pred.total_predictions > 0;

        setText('kpi-total-predictions', pred.total_predictions != null ? String(pred.total_predictions) : '—');
        setText('kpi-high-risk', pred.high_risk_count != null ? String(pred.high_risk_count) : '—');
        setText('kpi-avg-prob', hasPred ? formatPercent(pred.avg_delay_probability) : '—');
        setText('kpi-avg-hours', hasPred ? formatHours(pred.avg_predicted_hours) : '—');

        var hasActual = actual.total_completed > 0;
        setText('kpi-completed', actual.total_completed != null ? String(actual.total_completed) : '—');
        setText('kpi-on-time', actual.on_time != null ? String(actual.on_time) : '—');
        setText('kpi-delayed', actual.actually_delayed != null ? String(actual.actually_delayed) : '—');
        var rateOk = actual.total_completed > 0 && (actual.on_time + actual.actually_delayed) > 0;
        setText('kpi-delay-rate', rateOk ? formatPercent(actual.actual_delay_rate) : '—');

        var empty = document.getElementById('dashboard-empty');
        if (empty) empty.style.display = (!hasPred && !hasActual) ? '' : 'none';
    }

    function drawChart(daily, canvasId, emptyId, title, compactLabels) {
        var canvas = document.getElementById(canvasId);
        var emptyEl = document.getElementById(emptyId);
        if (!canvas) return;

        var valid = Array.isArray(daily) && daily.length > 0 &&
                    daily.some(function (d) { return (d.count || d.value || 0) > 0; });

        if (!valid) {
            canvas.style.display = 'none';
            if (emptyEl) emptyEl.style.display = '';
            return;
        }
        canvas.style.display = '';
        if (emptyEl) emptyEl.style.display = 'none';

        var ctx = canvas.getContext('2d');
        var ratio = window.devicePixelRatio || 1;
        var cssW = canvas.clientWidth || canvas.width;
        var cssH = canvas.clientHeight || canvas.height || 320;
        canvas.width = Math.floor(cssW * ratio);
        canvas.height = Math.floor(cssH * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        if (compactLabels) {
            var items = daily.slice(0, 6);
            var compactMax = Math.max.apply(null, items.map(function (item) { return Number(item.count) || 0; })) || 1;
            var labelWidth = Math.min(190, Math.max(120, cssW * 0.38));
            var rowHeight = (cssH - 34) / items.length;
            var barHeight = Math.max(16, rowHeight - 12);
            var barWidth = cssW - labelWidth - 58;

            ctx.font = '11px -apple-system, Segoe UI, Roboto, Arial, sans-serif';
            ctx.textBaseline = 'middle';
            for (var compactIndex = 0; compactIndex < items.length; compactIndex++) {
                var item = items[compactIndex];
                var itemLabel = String(item.label || '').replace(/_/g, ' ');
                if (itemLabel.length > 22) itemLabel = itemLabel.slice(0, 21) + '…';
                var y = 28 + compactIndex * rowHeight;
                var width = (Number(item.count) || 0) / compactMax * barWidth;

                ctx.fillStyle = '#64748b';
                ctx.textAlign = 'right';
                ctx.fillText(itemLabel, labelWidth - 10, y + barHeight / 2);
                ctx.fillStyle = '#e2e8f0';
                ctx.fillRect(labelWidth, y, barWidth, barHeight);
                var gradient = ctx.createLinearGradient(labelWidth, y, labelWidth + barWidth, y);
                gradient.addColorStop(0, '#2563eb');
                gradient.addColorStop(1, '#7c3aed');
                ctx.fillStyle = gradient;
                ctx.fillRect(labelWidth, y, width, barHeight);
                ctx.fillStyle = '#475569';
                ctx.textAlign = 'left';
                ctx.fillText(String(item.count), Math.min(labelWidth + width + 8, cssW - 18), y + barHeight / 2);
            }

            ctx.fillStyle = '#475569';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(title + ' (ilk 6)', labelWidth, 4);
            return;
        }

        var padL = 48, padR = 18, padT = 18, padB = 42;
        var plotW = cssW - padL - padR;
        var plotH = cssH - padT - padB;

        var counts = daily.map(function (d) { return Number(d.count != null ? d.count : d.value) || 0; });
        var maxV = Math.max.apply(null, counts);
        if (maxV <= 0) maxV = 1;
        var niceMax = Math.ceil(maxV / 5) * 5 || 5;
        var n = daily.length;
        var gap = 10;
        var barW = Math.max(8, (plotW - gap * (n - 1)) / n);

        ctx.font = '11px -apple-system, Segoe UI, Roboto, Arial, sans-serif';
        ctx.textBaseline = 'middle';
        var steps = 4;
        for (var s = 0; s <= steps; s++) {
            var val = (niceMax / steps) * s;
            var y = padT + plotH - (val / niceMax) * plotH;
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(padL + plotW, y);
            ctx.stroke();
            ctx.fillStyle = '#64748b';
            ctx.textAlign = 'right';
            ctx.fillText(String(Math.round(val)), padL - 8, y);
        }

        // bars + x labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (var i = 0; i < n; i++) {
            var x = padL + i * (barW + gap);
            var h = (counts[i] / niceMax) * plotH;
            var yBar = padT + plotH - h;

            var grad = ctx.createLinearGradient(0, yBar, 0, padT + plotH);
            grad.addColorStop(0, '#2563eb');
            grad.addColorStop(1, '#7c3aed');
            ctx.fillStyle = grad;
            ctx.fillRect(x, yBar, barW, Math.max(h, 0));

            ctx.fillStyle = '#64748b';
            var label = String(daily[i].date || daily[i].day || daily[i].label || '');
            if (label.length === 10) label = label.slice(5);
            ctx.save();
            ctx.translate(x + barW / 2, padT + plotH + 6);
            if (n > 10) { ctx.rotate(-Math.PI / 4); ctx.textAlign = 'right'; }
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }

        ctx.fillStyle = '#475569';
        ctx.font = '12px -apple-system, Segoe UI, Roboto, Arial, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(title, padL, 0);
    }

    async function load() {
        try {
            var resp = await fetch('/api/dashboard');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();
            renderKpis(data);
            var daily = data.daily_volume || [];
            drawChart(daily, 'prediction-chart', 'prediction-chart-empty', 'Günlük tahmin adedi', false);
            drawChart(data.process_type_distribution || [], 'process-type-chart', 'process-type-chart-empty', 'Süreç sayısı', true);
            var range = document.getElementById('prediction-chart-range');
            if (range) {
                range.textContent = daily.length ? daily[0].date + ' – ' + daily[daily.length - 1].date : '';
            }
        } catch (e) {
            console.error('Dashboard load failed:', e);
            ['kpi-total-predictions', 'kpi-high-risk', 'kpi-avg-prob', 'kpi-avg-hours',
             'kpi-completed', 'kpi-on-time', 'kpi-delayed', 'kpi-delay-rate'].forEach(function (id) {
                setText(id, '—');
            });
            var canvas = document.getElementById('prediction-chart');
            var emptyEl = document.getElementById('prediction-chart-empty');
            if (canvas) canvas.style.display = 'none';
            if (emptyEl) {
                emptyEl.style.display = '';
                emptyEl.textContent = 'Veri yüklenemedi.';
            }
            var typeCanvas = document.getElementById('process-type-chart');
            var typeEmpty = document.getElementById('process-type-chart-empty');
            if (typeCanvas) typeCanvas.style.display = 'none';
            if (typeEmpty) {
                typeEmpty.style.display = '';
                typeEmpty.textContent = 'Veri yüklenemedi.';
            }
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
