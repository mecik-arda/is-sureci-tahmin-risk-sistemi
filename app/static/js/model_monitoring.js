(function () {
    'use strict';

    function $(id) { return document.getElementById(id); }
    function setText(id, value) { var element = $(id); if (element) element.textContent = value == null || value === '' ? '—' : String(value); }
    function formatBytes(value) { return value ? (Number(value) / (1024 * 1024)).toFixed(1).replace('.', ',') + ' MB' : '—'; }
    function formatThreshold(value) { return value == null ? '—' : '%' + (Number(value) * 100).toFixed(0); }

    async function load() {
        var loading = $('monitoring-loading');
        var body = $('monitoring-body');
        var error = $('monitoring-error');
        try {
            var response = await fetch('/api/model-monitoring');
            if (!response.ok) throw new Error('Monitoring request failed');
            var data = await response.json();
            if (!data.available) throw new Error('Model unavailable');
            setText('monitor-version', data.model_version);
            setText('monitor-type', data.model_type);
            setText('monitor-trained-at', data.trained_at);
            setText('monitor-stage', data.stage);
            setText('monitor-schema', data.feature_schema_version);
            setText('monitor-mapping', data.canonical_mapping_version);
            setText('monitor-threshold', formatThreshold(data.threshold));
            setText('monitor-hash', data.artifact_hash);
            var cache = data.analysis_cache || {};
            setText('monitor-cache-built-at', cache.built_at);
            setText('monitor-cache-build-count', cache.build_count);
            setText('monitor-cache-hit-count', cache.cache_hit_count);
            setText('monitor-cache-rows', cache.cached_rows);
            setText('monitor-cache-memory', formatBytes(cache.cached_memory_bytes));
            if (loading) loading.style.display = 'none';
            if (body) body.style.display = '';
        } catch (e) {
            if (loading) loading.style.display = 'none';
            if (error) { error.textContent = 'Model izleme bilgileri şu anda kullanılamıyor.'; error.style.display = ''; }
        }
    }

    document.addEventListener('DOMContentLoaded', load);
})();
