(function () {
    'use strict';

    function $(id) { return document.getElementById(id); }
    function setText(id, value) { var element = $(id); if (element) element.textContent = value == null ? '—' : String(value); }
    function escapeHtml(value) { return String(value == null ? '' : value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/`/g, '&#96;'); }

    function renderRows(id, rows, render, emptyId) {
        var body = $(id);
        var empty = $(emptyId);
        if (body) body.innerHTML = rows.map(render).join('');
        if (empty) empty.style.display = rows.length ? 'none' : '';
    }

    async function load() {
        var loading = $('quality-loading');
        var body = $('quality-body');
        var error = $('quality-error');
        try {
            var response = await fetch('/api/data-quality');
            if (!response.ok) throw new Error('Data quality request failed');
            var data = await response.json();
            setText('quality-run-count', data.total_import_runs);
            setText('quality-total-rows', data.total_rows);
            setText('quality-quarantined', data.quarantined_rows);
            setText('quality-issue-count', data.issue_count);
            renderRows('quality-issues', data.issues_by_code || [], function (issue) {
                return '<tr><td>' + escapeHtml(issue.code) + '</td><td>' + escapeHtml(issue.count) + '</td></tr>';
            }, 'quality-issues-empty');
            renderRows('quality-runs', data.recent_runs || [], function (run) {
                return '<tr><td>' + escapeHtml(run.file_name) + '</td><td>' + escapeHtml(run.status) + '</td><td>' + escapeHtml(run.total_rows) + '</td><td>' + escapeHtml(run.quarantined_rows) + '</td><td>' + escapeHtml(run.error_rows) + '</td><td>' + escapeHtml(run.completed_at || '—') + '</td></tr>';
            }, 'quality-runs-empty');
            if (loading) loading.style.display = 'none';
            if (body) body.style.display = '';
        } catch (e) {
            if (loading) loading.style.display = 'none';
            if (error) { error.textContent = 'Veri kalite bilgileri yüklenemedi.'; error.style.display = ''; }
        }
    }

    document.addEventListener('DOMContentLoaded', load);
})();
