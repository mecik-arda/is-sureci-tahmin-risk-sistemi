/* Process list loader (S24) — single bundle scope, pagination, filters */
(function () {
    'use strict';

    var state = {
        page: 1,
        per_page: 20,
        status: '',
        risk: '',
        modelAvailable: false,
        selectedProcessIds: new Set()
    };

    function $(id) { return document.getElementById(id); }

    function formatPercent(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        return (value * 100).toFixed(1).replace('.', ',') + '%';
    }

    function formatHours(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        return Number(value).toFixed(1).replace('.', ',') + ' saat';
    }

    function riskPill(p) {
        if (!p || !p.has_prediction) return '<span class="pill pill--muted">Tahmin yok</span>';
        return p.predicted_is_delayed
            ? '<span class="pill pill--danger">Yüksek risk</span>'
            : '<span class="pill pill--success">Düşük risk</span>';
    }
    function riskScore(p) {
        if (!p || !p.has_prediction || p.risk_score == null) return '—';
        var labels = { low: 'Düşük', medium: 'Orta', high: 'Yüksek' };
        return p.risk_score + ' / 100 (' + (labels[p.risk_level] || '—') + ')';
    }

    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/`/g, '&#96;');
    }

    function renderRows(processes) {
        var tbody = $('process-tbody');
        if (!tbody) return;
        if (!processes.length) {
            tbody.innerHTML = '';
            return;
        }
        var html = processes.map(function (p) {
            var prob = (p.has_prediction && p.delay_probability != null) ? formatPercent(p.delay_probability) : '—';
            var hours = (p.has_prediction && p.predicted_hours != null) ? formatHours(p.predicted_hours) : '—';
            return ''
                + '<tr class="row-clickable" tabindex="0" role="link" data-href="/processes/' + encodeURIComponent(p.id) + '">'
                + '<td class="nowrap">' + escapeHtml(p.external_id || ('#' + p.id)) + '</td>'
                + '<td><input class="process-select" type="checkbox" value="' + escapeHtml(p.id) + '" aria-label="' + escapeHtml(p.external_id || p.id) + ' sürecini seç"' + (state.selectedProcessIds.has(p.id) ? ' checked' : '') + '></td>'
                + '<td>' + escapeHtml(p.process_type || '—') + '</td>'
                + '<td>' + escapeHtml(p.current_status || (p.completed_at ? 'Kapalı' : 'Açık')) + '</td>'
                + '<td>' + prob + '</td>'
                + '<td>' + escapeHtml(riskScore(p)) + '</td>'
                + '<td>' + riskPill(p) + '</td>'
                + '<td class="nowrap">' + hours + '</td>'
                + '</tr>';
        }).join('');
        tbody.innerHTML = html;

    }

    function openRow(row) {
        var href = row && row.getAttribute('data-href');
        if (href) window.location.href = href;
    }

    function selectedProcessIds() {
        return Array.from(state.selectedProcessIds);
    }

    function updateBatchButton() {
        var button = $('batch-predict');
        if (!button) return;
        var count = selectedProcessIds().length;
        button.disabled = !state.modelAvailable || count === 0;
        button.textContent = count
            ? 'Seçili Süreçler İçin Tahmin Üret (' + count + ')'
            : 'Seçili Süreçler İçin Tahmin Üret';
    }

    var lastBatchResults = null;

    function runBatchPrediction() {
        var button = $('batch-predict');
        var exportBtn = $('export-batch');
        var processIds = selectedProcessIds();
        if (!button || !processIds.length) return;
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        if (exportBtn) exportBtn.style.display = 'none';
        setText('batch-status', 'Seçili süreçler için tahmin üretiliyor...');
        fetch('/api/predictions/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ process_ids: processIds })
        })
            .then(function (response) {
                if (!response.ok) throw new Error('Batch prediction request failed');
                return response.json();
            })
            .then(function (data) {
                lastBatchResults = data;
                setText('batch-status', data.succeeded + ' süreçte tahmin üretildi, ' + data.failed + ' süreçte üretilemedi.');
                if (exportBtn && data.succeeded > 0) exportBtn.style.display = '';
                load();
            })
            .catch(function () {
                setText('batch-status', 'Toplu tahmin üretilemedi. Lütfen yeniden deneyin.');
            })
            .finally(function () {
                button.removeAttribute('aria-busy');
                updateBatchButton();
            });
    }

    function exportBatchToExcel() {
        if (!lastBatchResults || !lastBatchResults.results) return;
        var rows = [];
        lastBatchResults.results.forEach(function (r) {
            if (r.ok && r.prediction) {
                rows.push([
                    r.process_id,
                    r.prediction.delay_probability != null ? Math.round(r.prediction.delay_probability * 100) + '%' : '—',
                    r.prediction.predicted_is_delayed ? 'Risk var' : 'Risk yok',
                    r.prediction.predicted_hours != null ? Math.round(r.prediction.predicted_hours) + 'h' : '—',
                ]);
            } else {
                rows.push([r.process_id, 'Hata: ' + (r.error_code || 'bilinmeyen'), '', '']);
            }
        });

        var csv = '\uFEFFSüreç No,Gecikme Olasılığı,Tahmin,Tahmini Süre\n'
            + rows.map(function (r) { return r.join(','); }).join('\n');

        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'toplu_tahmin_sonuclari_' + new Date().toISOString().slice(0, 10) + '.csv';
        a.click();
        URL.revokeObjectURL(url);
    }

    function bindRowNavigation() {
        var tbody = $('process-tbody');
        if (!tbody) return;
        tbody.addEventListener('click', function (event) {
            if (event.target.closest('.process-select')) {
                return;
            }
            openRow(event.target.closest('tr.row-clickable'));
        });
        tbody.addEventListener('change', function (event) {
            var input = event.target.closest('.process-select');
            if (!input) return;
            var processId = Number(input.value);
            if (input.checked && state.selectedProcessIds.size >= 50) {
                input.checked = false;
                setText('batch-status', 'En fazla 50 süreç seçebilirsiniz.');
            } else if (input.checked) {
                state.selectedProcessIds.add(processId);
            } else {
                state.selectedProcessIds.delete(processId);
            }
            updateBatchButton();
        });
        tbody.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' || event.key === ' ') {
                if (event.target.closest('.process-select')) return;
                event.preventDefault();
                openRow(event.target.closest('tr.row-clickable'));
            }
        });
    }

    function renderPagination(total) {
        var nav = $('pagination');
        var info = $('page-info');
        var prev = $('prev-page');
        var next = $('next-page');
        var perPage = state.per_page;
        var totalPages = Math.max(1, Math.ceil(total / perPage));

        if (nav) {
            nav.style.display = total > 0 ? '' : 'none';
        }
        if (info) {
            var first = (state.page - 1) * perPage + 1;
            var last = Math.min(total, state.page * perPage);
            info.textContent = total > 0
                ? (first + '–' + last + ' / ' + total + ' kayıt  ·  Sayfa ' + state.page + ' / ' + totalPages)
                : '';
        }
        if (prev) prev.disabled = state.page <= 1;
        if (next) next.disabled = state.page >= totalPages;
    }

    function toggleEmpty(total) {
        var empty = $('empty-state');
        var loading = $('loading-state');
        if (loading) loading.style.display = 'none';
        if (empty) empty.style.display = total > 0 ? 'none' : '';
    }

    function renderModelContext(banner) {
        var detail = $('model-context-detail');
        var ctxBar = $('model-context');
        if (!detail || !banner) return;
        if (!banner.model_available) {
            detail.textContent = 'Model aktif değil (degrade modu)';
            if (ctxBar) ctxBar.classList.add('model-context--degraded');
            return;
        }
        state.modelAvailable = true;
        var parts = [];
        if (banner.model_version) parts.push('Sürüm ' + banner.model_version);
        if (banner.bundle_stage) parts.push(banner.bundle_stage);
        if (banner.threshold != null) parts.push('Eşik %' + (banner.threshold * 100).toFixed(0));
        detail.textContent = parts.length ? parts.join(' · ') : 'Aktif';
    }

    async function load() {
        var loading = $('loading-state');
        if (loading) loading.style.display = '';
        var params = new URLSearchParams();
        params.set('page', state.page);
        params.set('per_page', state.per_page);
        if (state.status) params.set('status', state.status);
        if (state.risk) params.set('risk', state.risk);

        try {
            var resp = await fetch('/api/processes?' + params.toString());
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();
            var processes = data.processes || [];
            renderRows(processes);
            renderPagination(data.total != null ? data.total : processes.length);
            toggleEmpty(data.total != null ? data.total : processes.length);
            if (data.banner) renderModelContext(data.banner);
            updateBatchButton();
        } catch (e) {
            console.error('Process list load failed:', e);
            renderRows([]);
            renderPagination(0);
            if (loading) loading.style.display = 'none';
            var empty = $('empty-state');
            if (empty) {
                empty.textContent = 'Kayıtlar yüklenemedi.';
                empty.style.display = '';
            }
        }
    }

    function bindFilters() {
        var form = $('filter-bar');
        var reset = $('filter-reset');
        var prev = $('prev-page');
        var next = $('next-page');
        var batch = $('batch-predict');
        var exportBatch = $('export-batch');

        if (form) {
            form.addEventListener('submit', function (ev) {
                ev.preventDefault();
                var status = $('status-filter') ? $('status-filter').value : '';
                var risk = $('risk-filter') ? $('risk-filter').value : '';
                state.status = ['open', 'closed'].indexOf(status) >= 0 ? status : '';
                state.risk = ['high_risk', 'low_risk'].indexOf(risk) >= 0 ? risk : '';
                state.page = 1;
                load();
            });
        }
        if (reset) {
            reset.addEventListener('click', function () {
                if ($('status-filter')) $('status-filter').value = '';
                if ($('risk-filter')) $('risk-filter').value = '';
                state.status = '';
                state.risk = '';
                state.page = 1;
                load();
            });
        }
        if (prev) prev.addEventListener('click', function () { if (state.page > 1) { state.page--; load(); } });
        if (next) next.addEventListener('click', function () { state.page++; load(); });
        if (batch) batch.addEventListener('click', runBatchPrediction);
        if (exportBatch) exportBatch.addEventListener('click', exportBatchToExcel);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { bindFilters(); bindRowNavigation(); load(); });
    } else {
        bindFilters();
        bindRowNavigation();
        load();
    }
})();
