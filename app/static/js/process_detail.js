/* Process detail loader (S22 dual card + S25 prediction + S26 XAI + S27 similar) */
(function () {
    'use strict';

    /* ---------- helpers ---------- */
    function $(id) { return document.getElementById(id); }
    function setText(id, value) { var el = $(id); if (el) el.textContent = value; }
    function show(el) { if (el) el.style.display = ''; }
    function hide(el) { if (el) el.style.display = 'none'; }
    function escapeHtml(v) {
        if (v === null || v === undefined) return '';
        return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;').replace(/`/g, '&#96;');
    }
    function fmtPct(v) {
        if (v === null || v === undefined || Number.isNaN(v)) return '—';
        return (Number(v) * 100).toFixed(1).replace('.', ',') + '%';
    }
    function riskLevelLabel(value) {
        var labels = { low: 'Düşük', medium: 'Orta', high: 'Yüksek' };
        return labels[value] || '—';
    }
    function fmtHours(v) {
        if (v === null || v === undefined || Number.isNaN(v)) return '—';
        return Number(v).toFixed(1).replace('.', ',') + ' saat';
    }
    function fmtNum(v) {
        if (v === null || v === undefined || Number.isNaN(v)) return '—';
        return String(v);
    }
    function fmtDateTime(iso) {
        if (!iso) return null;
        var d = new Date(iso);
        if (Number.isNaN(d.getTime())) return iso;
        var pad = function (n) { return (n < 10 ? '0' : '') + n; };
        return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
            ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
    }

    var MONTHS_TR = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
        'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık'];
    var WEEKDAYS_TR = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'];

    var FEATURE_ORDER;
    var simulationOptions = {};
    var FALLBACK_LABELS = {
        'source': 'Bildirim Kaynağı',
        'subject': 'Konu',
        'reason': 'Neden',
        'type': 'Süreç Tipi',
        'neighborhood': 'Mahalle',
        'open_month': 'Açılış Ayı',
        'open_weekday': 'Açılış Günü',
        'open_hour': 'Açılış Saati',
        'is_weekend': 'Haftasonu Açılış',
        'sla_duration_hours': 'SLA Süresi (saat)'
    };
    var FEATURE_KEYS = ['source', 'subject', 'reason', 'type', 'neighborhood',
        'open_month', 'open_weekday', 'open_hour', 'is_weekend', 'sla_duration_hours'];

    function buildFeatureOrder(labelsFromCatalog) {
        var catalog = labelsFromCatalog && labelsFromCatalog.feature_labels ? labelsFromCatalog.feature_labels : {};
        return FEATURE_KEYS.map(function (key) {
            return [key, catalog[key] || FALLBACK_LABELS[key] || key];
        });
    }

    function initFeatureOrder() {
        if (FEATURE_ORDER) return;
        FEATURE_ORDER = FEATURE_KEYS.map(function (key) {
            return [key, FALLBACK_LABELS[key] || key];
        });
    }

    async function loadLabelCatalog() {
        try {
            var resp = await fetch('/api/label-catalog');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();
            FEATURE_ORDER = buildFeatureOrder(data);
        } catch (e) {
            initFeatureOrder();
        }
    }

    async function loadSimulationOptions() {
        try {
            var resp = await fetch('/api/simulation-options');
            if (!resp.ok) throw new Error('Simulation options request failed');
            simulationOptions = await resp.json();
        } catch (e) {
            simulationOptions = {};
        }
    }

    function featureDisplay(key, value) {
        if (value === null || value === undefined) return '—';
        if (key === 'open_month') {
            var m = Number(value);
            return (m >= 1 && m <= 12) ? (MONTHS_TR[m - 1] + ' (' + m + ')') : fmtNum(value);
        }
        if (key === 'open_weekday') {
            var w = Number(value);
            return (w >= 0 && w <= 6) ? WEEKDAYS_TR[w] : fmtNum(value);
        }
        if (key === 'is_weekend') {
            return value === 1 || value === true ? 'Evet' : 'Hayır';
        }
        if (key === 'open_hour') {
            return Number.isFinite(Number(value)) ? (value + ':00') : fmtNum(value);
        }
        if (key === 'sla_duration_hours') {
            return Number.isNaN(Number(value)) ? 'Belirsiz' : fmtHours(value);
        }
        if (value === 'missing' || value === '') return 'Belirtilmemiş';
        return String(value);
    }

    function getProcessId() {
        var m = window.location.pathname.match(/\/processes\/(\d+)/);
        if (m) return parseInt(m[1], 10);
        return null;
    }

    /* ---------- S22 dual card ---------- */
    function renderOpening(openingFields) {
        var tbody = $('opening-tbody');
        var empty = $('opening-empty');
        if (!openingFields || Object.keys(openingFields).length === 0) {
            if (tbody) tbody.innerHTML = '';
            show(empty);
            return;
        }
        hide(empty);
        var html = FEATURE_ORDER.map(function (pair) {
            var key = pair[0], label = pair[1];
            return '<tr><td>' + escapeHtml(label) + '</td><td>' +
                escapeHtml(featureDisplay(key, openingFields[key])) + '</td></tr>';
        }).join('');
        if (tbody) tbody.innerHTML = html;
    }

    function renderCurrent(currentFields) {
        var tbody = $('current-tbody');
        if (!tbody || !currentFields) return;
        var cf = currentFields;
        var status = cf.current_status || '—';
        var completed = cf.completed_at ? fmtDateTime(cf.completed_at) : 'Devam ediyor';
        var deadline = cf.deadline ? fmtDateTime(cf.deadline) : 'Tanımsız';
        var reason = cf.closure_reason && cf.closure_reason !== '' ? cf.closure_reason : '—';
        var rows = [
            ['Güncel Durum', status],
            ['Tamamlanma Tarihi', completed],
            ['Son Başlama (SLA) Tarihi', deadline],
            ['Kapanış Nedeni', reason]
        ];
        tbody.innerHTML = rows.map(function (r) {
            return '<tr><td>' + escapeHtml(r[0]) + '</td><td>' + escapeHtml(r[1]) + '</td></tr>';
        }).join('');
    }

    /* ---------- S25 prediction ---------- */
    function renderPrediction(data, threshold) {
        var hero = $('pred-probability');
        var verdict = $('pred-verdict');
        var pred = data.prediction;
        var hasSla = data.has_sla === true;
        var noSla = $('prediction-no-sla');
        var empty = $('prediction-empty');

        setText('pred-threshold', threshold != null ? fmtPct(threshold) : '—');

        if (!hasSla) show(noSla);
        else hide(noSla);

        if (!pred) {
            if (hero) { hero.textContent = '—'; hero.classList.remove('is-risk', 'is-safe'); }
            if (verdict) { verdict.textContent = ''; verdict.classList.remove('is-risk', 'is-safe'); }
            setText('pred-decision', '—');
            setText('pred-risk-score', '—');
            setText('pred-risk-level', '—');
            setText('pred-hours', '—');
            setText('pred-at', '—');
            setText('pred-model-version', '—');
            show(empty);
            return;
        }
        hide(empty);

        var prob = pred.delay_probability;
        if (hero) {
            hero.textContent = fmtPct(prob);
            hero.classList.remove('is-risk', 'is-safe');
            if (verdict) verdict.classList.remove('is-risk', 'is-safe');
            if (prob !== null && prob !== undefined && !Number.isNaN(prob)) {
                if (pred.predicted_is_delayed) {
                    hero.classList.add('is-risk');
                    if (verdict) { verdict.classList.add('is-risk'); verdict.textContent = 'Yüksek gecikme riski'; }
                } else {
                    hero.classList.add('is-safe');
                    if (verdict) { verdict.classList.add('is-safe'); verdict.textContent = 'Düşük gecikme riski'; }
                }
            }
        }
        setText('pred-decision', pred.predicted_is_delayed
            ? 'Gecikme riski var'
            : (prob !== null && prob !== undefined ? 'Gecikme riski yok' : '—'));
        setText('pred-risk-score', pred.risk_score != null ? pred.risk_score + ' / 100' : '—');
        setText('pred-risk-level', riskLevelLabel(pred.risk_level));
        setText('pred-hours', fmtHours(pred.predicted_hours));
        setText('pred-at', fmtDateTime(pred.predicted_at) || '—');
        setText('pred-model-version', pred.model_version || '—');
    }

    async function loadPredictionHistory(processId) {
        var tbody = $('history-tbody');
        var empty = $('history-empty');
        if (!tbody) return;
        try {
            var response = await fetch('/api/processes/' + processId + '/prediction-history');
            if (!response.ok) throw new Error('History request failed');
            var data = await response.json();
            var predictions = data.predictions || [];
            if (!predictions.length) {
                tbody.innerHTML = '';
                show(empty);
                return;
            }
            hide(empty);
            tbody.innerHTML = predictions.map(function (prediction) {
                var context = prediction.prediction_context === 'simulation' ? 'Varsayımsal senaryo' : 'Açılış tahmini';
                var decision = prediction.predicted_is_delayed === true ? 'Gecikme riski var' : (prediction.predicted_is_delayed === false ? 'Gecikme riski yok' : '—');
                var score = prediction.risk_score != null ? prediction.risk_score + ' / 100 (' + riskLevelLabel(prediction.risk_level) + ')' : '—';
                return '<tr><td>' + escapeHtml(fmtDateTime(prediction.predicted_at) || '—') + '</td><td>' + escapeHtml(context) + '</td><td>' + escapeHtml(prediction.model_version || '—') + '</td><td>' + escapeHtml(fmtPct(prediction.delay_probability)) + '</td><td>' + escapeHtml(score) + '</td><td>' + escapeHtml(decision) + '</td></tr>';
            }).join('');
        } catch (e) {
            tbody.innerHTML = '';
            if (empty) { empty.textContent = 'Tahmin geçmişi yüklenemedi.'; show(empty); }
        }
    }

    function initPredictionAction(processId, prediction, modelAvailable) {
        var action = $('prediction-action');
        var button = $('btn-predict');
        if (!action || !button) return;
        if (prediction || !modelAvailable) {
            hide(action);
            return;
        }
        show(action);
        button.onclick = function () {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            setText('prediction-create-status', 'Tahmin üretiliyor...');
            hide($('prediction-create-error'));
            fetch('/api/processes/' + encodeURIComponent(processId) + '/predictions', { method: 'POST' })
                .then(function (response) {
                    if (!response.ok) throw new Error('Prediction request failed');
                    return response.json();
                })
                .then(function () {
                    window.location.reload();
                })
                .catch(function () {
                    button.disabled = false;
                    button.removeAttribute('aria-busy');
                    setText('prediction-create-status', '');
                    var error = $('prediction-create-error');
                    if (error) {
                        error.textContent = 'Tahmin üretilemedi. Lütfen daha sonra yeniden deneyin.';
                        show(error);
                    }
                });
        };
    }

    /* ---------- S26 XAI ---------- */
    async function loadXai(processId) {
        var globalContainer = $('xai-bars');
        var globalEmpty = $('xai-empty');
        var shapPanel = $('shap-panel');
        var shapContainer = $('shap-bars');
        var shapEmpty = $('shap-empty');
        try {
            var resp = await fetch('/api/processes/' + processId + '/xai');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();

            // Global importances
            var imp = data.importances || [];
            if (!data.available || imp.length === 0) {
                if (globalContainer) globalContainer.innerHTML = '';
                show(globalEmpty);
            } else {
                hide(globalEmpty);
                renderImportanceBars(imp, globalContainer);
            }

            // Per-instance SHAP
            var shap = data.shap_values || [];
            if (shap.length === 0) {
                if (shapPanel) shapPanel.style.display = 'none';
            } else {
                if (shapPanel) shapPanel.style.display = '';
                hide(shapEmpty);
                if (shapContainer) renderShapBars(shap, shapContainer);
            }
        } catch (e) {
            console.error('XAI load failed:', e);
            if (globalContainer) globalContainer.innerHTML = '';
            if (shapPanel) shapPanel.style.display = 'none';
        }
    }

    function renderImportanceBars(imp, container) {
        if (!container) return;
        var maxAbs = 0;
        imp.forEach(function (f) { var a = Math.abs(f.importance || 0); if (a > maxAbs) maxAbs = a; });
        if (maxAbs <= 0) maxAbs = 1;
        var html = imp.slice(0, 10).map(function (f) {
            var pct = Math.max(2, (Math.abs(f.importance || 0) / maxAbs) * 100);
            var label = f.label_tr || 'Tanımlanmamış değer';
            var val = Number(f.importance || 0).toFixed(4).replace('.', ',');
            return ''
                + '<div class="importance-item">'
                + '  <div class="importance-item__head">'
                + '    <span class="importance-item__label">' + escapeHtml(label) + '</span>'
                + '    <span class="importance-item__value">' + escapeHtml(val) + '</span>'
                + '  </div>'
                + '  <div class="importance-item__track">'
                + '    <div class="importance-item__fill" style="width:' + pct.toFixed(1) + '%"></div>'
                + '  </div>'
                + '</div>';
        }).join('');
        container.innerHTML = html;
    }

    function renderShapBars(shap, container) {
        if (!container) return;
        var maxAbs = 0;
        shap.forEach(function (s) { var a = Math.abs(s.shap_value || 0); if (a > maxAbs) maxAbs = a; });
        if (maxAbs <= 0) maxAbs = 1;
        var html = shap.slice(0, 10).map(function (s) {
            var pct = Math.max(2, (Math.abs(s.shap_value || 0) / maxAbs) * 100);
            var label = s.label_tr || 'Tanımlanmamış değer';
            var val = Number(s.shap_value || 0).toFixed(4).replace('.', ',');
            var dirClass = (s.shap_value > 0) ? 'importance-item__fill--risk' : 'importance-item__fill--safe';
            var dirLabel = (s.shap_value > 0) ? 'Riski artıran' : 'Riski azaltan';
            return ''
                + '<div class="importance-item">'
                + '  <div class="importance-item__head">'
                + '    <span class="importance-item__label">' + escapeHtml(label) + '</span>'
                + '    <span class="importance-item__dir">' + escapeHtml(dirLabel) + '</span>'
                + '    <span class="importance-item__value">' + escapeHtml(val) + '</span>'
                + '  </div>'
                + '  <div class="importance-item__track">'
                + '    <div class="importance-item__fill ' + dirClass + '" style="width:' + pct.toFixed(1) + '%"></div>'
                + '  </div>'
                + '</div>';
        }).join('');
        container.innerHTML = html;
    }
            show(empty);
            if (empty) empty.textContent = 'Özellik önem bilgisi yüklenemedi.';
        }
    }

    /* ---------- S27 similar ---------- */
    async function loadSimilar(processId) {
        var tbody = $('similar-tbody');
        var empty = $('similar-empty');
        if (!tbody) return;
        try {
            var resp = await fetch('/api/processes/' + processId + '/similar');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();
            var neighbors = data.neighbors || [];
            if (!data.available || neighbors.length === 0) {
                tbody.innerHTML = '';
                show(empty);
                return;
            }
            hide(empty);
            tbody.innerHTML = neighbors.map(function (n) {
                var status = n.is_delayed === true
                    ? '<span class="pill pill--danger">Geciken</span>'
                    : (n.is_delayed === false ? '<span class="pill pill--success">Zamanında</span>' : '—');
                var duration = (n.total_duration_hours != null) ? fmtHours(n.total_duration_hours) : '—';
                return '<tr>'
                    + '<td class="nowrap">' + escapeHtml(n.external_id || '—') + '</td>'
                    + '<td>' + escapeHtml(n.process_type || '—') + '</td>'
                    + '<td>' + status + '</td>'
                    + '<td class="nowrap">' + duration + '</td>'
                    + '</tr>';
            }).join('');
        } catch (e) {
            console.error('Similar load failed:', e);
            tbody.innerHTML = '';
            show(empty);
            if (empty) empty.textContent = 'Benzer süreçler yüklenemedi.';
        }
    }

    /* ---------- S30 simulation ---------- */
    var simulationPredictionId = null;
    var simulationOrigProb = null;
    var simulationOrigHours = null;
    var simulationOrigDelayed = null;

    function buildSimFields(openingFields) {
        if (!openingFields) return;
        var fields = $('simulation-override-fields');
        var wrap = $('simulation-form-wrap');
        if (!fields) return;
        var pairs = FEATURE_ORDER;
        var html = pairs.map(function (pair) {
            var key = pair[0], label = pair[1];
            var currentVal = openingFields[key];
            var displayVal = featureDisplay(key, currentVal);
            var isCategorical = (['source', 'subject', 'reason', 'type', 'neighborhood'].indexOf(key) >= 0);
            if (isCategorical) {
                var choices = simulationOptions[key] || [];
                if (currentVal && choices.indexOf(currentVal) < 0) choices = [currentVal].concat(choices);
                var options = '<option value="">Orijinal değerini koru</option>' + choices.map(function (choice) {
                    return '<option value="' + escapeHtml(choice) + '">' + escapeHtml(choice) + '</option>';
                }).join('');
                return ''
                    + '<div class="sim-field">'
                    + '  <label for="sim-' + key + '" class="sim-field__label">' + escapeHtml(label) + ' <span class="sim-field__current">(mevcut: ' + escapeHtml(displayVal) + ')</span></label>'
                    + '  <select class="sim-field__input" id="sim-' + key + '">' + options + '</select>'
                    + '</div>';
            } else {
                var step = (key === 'is_weekend') ? '1' : 'any';
                var min = '';
                var placeholder = 'Değiştirmek için girin...';
                if (key === 'open_month') { min = 'min="1" max="12"'; step = '1'; placeholder = '1-12'; }
                if (key === 'open_weekday') { min = 'min="0" max="6"'; step = '1'; placeholder = '0-6'; }
                if (key === 'open_hour') { min = 'min="0" max="23"'; step = '1'; placeholder = '0-23'; }
                if (key === 'is_weekend') { min = 'min="0" max="1"'; placeholder = '0 veya 1'; }
                if (key === 'sla_duration_hours') { min = 'min="0" max="87600"'; placeholder = 'Saat cinsinden'; }
                return ''
                    + '<div class="sim-field">'
                    + '  <label for="sim-' + key + '" class="sim-field__label">' + escapeHtml(label) + ' <span class="sim-field__current">(mevcut: ' + escapeHtml(displayVal) + ')</span></label>'
                    + '  <input class="sim-field__input" type="number" id="sim-' + key + '" ' + min + ' step="' + step + '" placeholder="' + placeholder + '">'
                    + '</div>';
            }
        }).join('');
        fields.innerHTML = html;
        show(wrap);
    }

    function getSimOverrides() {
        var overrides = {};
        var keys = ['source', 'subject', 'reason', 'type', 'neighborhood', 'open_month', 'open_weekday', 'open_hour', 'is_weekend', 'sla_duration_hours'];
        keys.forEach(function (k) {
            var el = document.getElementById('sim-' + k);
            if (!el) return;
            var v = el.value.trim();
            if (v === '') return;
            if (k === 'source' || k === 'subject' || k === 'reason' || k === 'type' || k === 'neighborhood') {
                overrides[k] = v;
            } else {
                var n = Number(v);
                if (!Number.isNaN(n)) overrides[k] = n;
            }
        });
        return overrides;
    }

    function runSimulation(processId) {
        if (simulationPredictionId === null) {
            setText('sim-status', 'Önce bir tahmin gereklidir.');
            return;
        }
        var overrides = getSimOverrides();
        var hasOverrides = false;
        for (var _k in overrides) { if (Object.prototype.hasOwnProperty.call(overrides, _k)) { hasOverrides = true; break; } }
        if (!hasOverrides) {
            setText('sim-status', 'En az bir alanı değiştirmelisiniz.');
            return;
        }
        var btn = $('sim-run-btn');
        var statusEl = $('sim-status');
        if (btn) { btn.disabled = true; btn.setAttribute('aria-busy', 'true'); }
        if (statusEl) statusEl.textContent = 'Çalıştırılıyor...';
        hide($('simulation-result'));
        hide($('simulation-error'));

        fetch('/api/processes/' + encodeURIComponent(processId) + '/simulations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_prediction_id: simulationPredictionId, overrides: overrides })
        })
            .then(function (resp) {
                if (!resp.ok) throw new Error('Simulation request failed');
                return resp.json();
            })
            .then(function (data) {
                if (btn) { btn.disabled = false; btn.removeAttribute('aria-busy'); }
                if (statusEl) statusEl.textContent = '';
                show($('simulation-result'));
                hide($('simulation-error'));

                setText('sim-orig-prob', fmtPct(simulationOrigProb));
                setText('sim-orig-hours', fmtHours(simulationOrigHours));
                var origDec = (simulationOrigDelayed === true) ? 'Gecikme riski var' : ((simulationOrigDelayed === false) ? 'Gecikme riski yok' : '—');
                setText('sim-orig-decision', origDec);

                setText('sim-new-prob', fmtPct(data.delay_probability));
                setText('sim-new-hours', fmtHours(data.predicted_hours));
                var newDec = (data.predicted_is_delayed === true) ? 'Gecikme riski var' : ((data.predicted_is_delayed === false) ? 'Gecikme riski yok' : '—');
                setText('sim-new-decision', newDec);
                setText('sim-threshold', fmtPct(data.threshold));

                if (simulationOrigProb != null && data.delay_probability != null) {
                    var diff = Math.abs(Number(data.delay_probability) - Number(simulationOrigProb)) * 100;
                    var direction = Number(data.delay_probability) < Number(simulationOrigProb) ? 'daha düşük' : 'daha yüksek';
                    setText('sim-diff-text', 'Bu varsayımsal girdide model çıktısı ' + diff.toFixed(1).replace('.', ',') + ' yüzde puanı ' + direction + 'tür.');
                }
            })
            .catch(function () {
                if (btn) { btn.disabled = false; btn.removeAttribute('aria-busy'); }
                if (statusEl) statusEl.textContent = '';
                hide($('simulation-result'));
                var errEl = $('simulation-error');
                if (errEl) { errEl.textContent = 'Senaryo çalıştırılamadı. Girdileri kontrol edip yeniden deneyin.'; show(errEl); }
            });
    }

    function initSimulation(processId, openingFields, prediction) {
        var btn = document.getElementById('sim-run-btn');
        if (btn) {
            btn.addEventListener('click', function () { runSimulation(processId); });
        }
        if (prediction && prediction.prediction_id) {
            simulationPredictionId = prediction.prediction_id;
            simulationOrigProb = prediction.delay_probability;
            simulationOrigHours = prediction.predicted_hours;
            simulationOrigDelayed = prediction.predicted_is_delayed;
            buildSimFields(openingFields);
            hide($('simulation-no-prediction'));
        } else {
            hide($('simulation-form-wrap'));
            show($('simulation-no-prediction'));
        }
    }

    /* ---------- S31 feedback ---------- */
    var feedbackPredictionId = null;
    function initFeedback(processId, prediction, hasSla) {
        feedbackPredictionId = prediction && prediction.prediction_id ? prediction.prediction_id : null;
        if (!feedbackPredictionId) {
            show($('feedback-no-prediction'));
            return;
        }
        hide($('feedback-no-prediction'));
        if (prediction.prediction_context === 'simulation') {
            show($('feedback-simulation'));
            hide($('feedback-accuracy-section'));
            hide($('feedback-usefulness-section'));
            return;
        }
        hide($('feedback-simulation'));

        var accSection = $('feedback-accuracy-section');
        if (!hasSla) {
            setText('fb-accuracy-status', 'SLA tanımlı olmadığı için gecikme tahmininin doğruluğu değerlendirilemez.');
            show(accSection);
        } else {
            show(accSection);
            setText('fb-accuracy-prediction', prediction.predicted_is_delayed ? 'Gecikme riski var' : 'Gecikme riski yok');
            fetch('/api/predictions/' + feedbackPredictionId + '/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ feedback_type: 'accuracy' })
            })
                .then(function (resp) { if (!resp.ok) throw new Error('Feedback request failed'); return resp.json(); })
                .then(function (data) {
                    if (data.actual_outcome === null || data.actual_outcome === undefined) {
                        setText('fb-accuracy-status', 'Bu tahminin doğruluğu henüz değerlendirilemiyor.');
                        setText('fb-accuracy-actual', 'Henüz çözümlenmedi');
                    } else {
                        var outcomeText = data.actual_outcome === 1 ? 'Geciken' : 'Zamanında';
                        var matchText = (prediction.predicted_is_delayed && data.actual_outcome === 1) || (!prediction.predicted_is_delayed && data.actual_outcome === 0) ? 'Uyuştu' : 'Uyuşmadı';
                        setText('fb-accuracy-status', 'Sonuç: ' + matchText);
                        setText('fb-accuracy-actual', outcomeText);
                    }
                })
                .catch(function () {
                    setText('fb-accuracy-status', 'Doğruluk bilgisi alınamadı.');
                });
        }

        var useSection = $('feedback-usefulness-section');
        show(useSection);
        var btn = document.getElementById('fb-usefulness-btn');
        if (btn) {
            btn.addEventListener('click', function () {
                var comment = document.getElementById('fb-usefulness-comment');
                if (!comment || !comment.value.trim()) {
                    setText('fb-usefulness-status', 'Yorum alanı boş bırakılamaz.');
                    return;
                }
                btn.disabled = true;
                btn.setAttribute('aria-busy', 'true');
                setText('fb-usefulness-status', 'Gönderiliyor...');
                fetch('/api/predictions/' + feedbackPredictionId + '/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ feedback_type: 'usefulness', comment: comment.value.trim() })
                })
                    .then(function (resp) {
                        if (!resp.ok) throw new Error('Feedback request failed');
                        return resp.json();
                    })
                    .then(function () {
                        btn.disabled = false;
                        btn.removeAttribute('aria-busy');
                        setText('fb-usefulness-status', 'Gönderildi.');
                        hide($('feedback-error'));
                    })
                    .catch(function () {
                        btn.disabled = false;
                        btn.removeAttribute('aria-busy');
                        setText('fb-usefulness-status', 'Gönderilemedi.');
                        var feedbackError = $('feedback-error');
                        if (feedbackError) { feedbackError.textContent = 'Geri bildirim gönderilemedi. Lütfen yeniden deneyin.'; show(feedbackError); }
                    });
            });
        }
    }

    /* ---------- main ---------- */
    async function load() {
        var supportData = Promise.all([loadLabelCatalog(), loadSimulationOptions()]);

        var processId = getProcessId();
        if (processId === null) {
            show($('process-not-found'));
            hide($('process-loading'));
            hide($('process-body'));
            return;
        }

        try {
            var resp = await fetch('/api/processes/' + processId);
            if (resp.status === 404) {
                show($('process-not-found'));
                hide($('process-loading'));
                hide($('process-body'));
                return;
            }
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();
            await supportData;

            setText('process-external-id', data.external_id || ('#' + data.id));
            setText('process-type', data.process_type || '');

            renderOpening(data.opening_fields);
            renderCurrent(data.current_fields);

            var threshold = (data.banner && data.banner.threshold != null) ? data.banner.threshold : null;
            renderPrediction(data, threshold);
            initPredictionAction(processId, data.prediction, data.banner && data.banner.model_available === true);

            initSimulation(processId, data.opening_fields, data.prediction);
            initFeedback(processId, data.prediction, data.has_sla);

            hide($('process-loading'));
            show($('process-body'));
        } catch (e) {
            await supportData;
            console.error('Process detail load failed:', e);
            hide($('process-loading'));
            show($('process-not-found'));
            setText('process-not-found', 'Süreç bilgileri yüklenemedi.');
            hide($('process-body'));
            return;
        }

        // secondary panels (best-effort, independent)
        loadXai(processId);
        loadSimilar(processId);
        loadPredictionHistory(processId);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
