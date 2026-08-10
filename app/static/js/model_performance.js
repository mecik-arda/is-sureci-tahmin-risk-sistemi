/* Model performance page loader (S32) */
(function () {
    'use strict';

    function $(id) { return document.getElementById(id); }
    function setText(id, value) {
        var el = $(id);
        if (el) el.textContent = (value === null || value === undefined || value === '') ? '—' : String(value);
    }
    function fmt(num) {
        if (num === null || num === undefined || Number.isNaN(num)) return '—';
        return Number(num).toFixed(4).replace('.', ',');
    }
    function fmtInt(num) {
        if (num === null || num === undefined || Number.isNaN(num)) return '—';
        return String(Math.round(num));
    }
    function show(el) { if (el) el.style.display = ''; }
    function hide(el) { if (el) el.style.display = 'none'; }

    async function load() {
        try {
            var resp = await fetch('/api/model-performance');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            var data = await resp.json();

            if (!data.available) {
                show($('perf-error'));
                setText('perf-error', 'Model performans bilgisi mevcut değil.');
                hide($('perf-loading'));
                return;
            }

            hide($('perf-loading'));
            show($('perf-body'));

            var bundle = data.bundle || {};
            setText('perf-model-version', bundle.model_version);
            setText('perf-model-type', bundle.model_type);
            setText('perf-stage', bundle.stage);
            var thr = bundle.threshold;
            setText('perf-threshold', thr != null ? '%' + (Number(thr) * 100).toFixed(0) : '—');
            setText('perf-calibration', bundle.calibration_method);

            var clf = data.classification || {};
            setText('perf-pr-auc', fmt(clf.validation_pr_auc));
            setText('perf-roc-auc', fmt(clf.validation_roc_auc));
            setText('perf-brier', fmt(clf.validation_brier));
            setText('perf-f1', fmt(clf.validation_f1));
            setText('perf-precision', fmt(clf.validation_precision));
            setText('perf-recall', fmt(clf.validation_recall));
            setText('perf-clf-count', fmtInt(clf.classification_validation_row_count));
            var matrix = clf.validation_confusion_matrix;
            if (Array.isArray(matrix) && matrix.length === 2 && Array.isArray(matrix[0]) && Array.isArray(matrix[1]) && matrix[0].length === 2 && matrix[1].length === 2) {
                setText('perf-tn', fmtInt(matrix[0][0]));
                setText('perf-fp', fmtInt(matrix[0][1]));
                setText('perf-fn', fmtInt(matrix[1][0]));
                setText('perf-tp', fmtInt(matrix[1][1]));
                show($('perf-confusion'));
                hide($('perf-confusion-empty'));
            } else {
                hide($('perf-confusion'));
                show($('perf-confusion-empty'));
            }

            var reg = data.regression || {};
            setText('perf-mae', fmt(reg.validation_mae));
            setText('perf-medae', fmt(reg.validation_median_ae));
            setText('perf-rmse', fmt(reg.validation_rmse));
            setText('perf-p90ae', fmt(reg.validation_p90_ae));
            setText('perf-reg-count', fmtInt(reg.regression_validation_row_count));

            setText('perf-cv-clf-mean', fmt(clf.cv_pr_auc_mean));
            setText('perf-cv-clf-std', fmt(clf.cv_pr_auc_std));
            setText('perf-cv-reg-mean', fmt(reg.cv_mae_mean));
            setText('perf-cv-reg-std', fmt(reg.cv_mae_std));
        } catch (e) {
            console.error('Performance load failed:', e);
            hide($('perf-loading'));
            var errEl = $('perf-error');
            if (errEl) { errEl.textContent = 'Performans bilgileri yüklenemedi.'; show(errEl); }
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
