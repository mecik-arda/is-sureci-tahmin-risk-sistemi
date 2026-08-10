(function () {
    'use strict';

    function $(id) { return document.getElementById(id); }
    function setText(id, value) { var element = $(id); if (element) element.textContent = value == null ? '—' : String(value); }
    function show(element) { if (element) element.style.display = ''; }
    function hide(element) { if (element) element.style.display = 'none'; }

    function statusLabel(status) {
        var labels = {
            completed: 'Tamamlandı',
            completed_with_issues: 'Sorunlarla tamamlandı',
            duplicate_file: 'Aynı dosya daha önce işlendi',
            failed: 'Tamamlanamadı'
        };
        return labels[status] || status;
    }

    function renderResult(data) {
        var counts = data.counts || {};
        setText('import-result-status', statusLabel(data.status));
        setText('import-total-rows', counts.total_rows);
        setText('import-inserted-rows', counts.inserted_rows);
        setText('import-updated-rows', counts.updated_rows);
        setText('import-skipped-rows', counts.skipped_duplicate_rows);
        setText('import-quarantined-rows', counts.quarantined_rows);
        setText('import-error-rows', counts.error_rows);
        setText('import-warning-count', counts.warning_count);
        setText('import-run-id', data.import_run_id);
        show($('import-result'));
    }

    function bindForm() {
        var form = $('import-form');
        var input = $('import-file');
        var button = $('import-submit');
        if (!form || !input || !button) return;
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            var file = input.files && input.files[0];
            if (!file) return;
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            setText('import-status', 'Dosya doğrulanıyor ve aktarılıyor...');
            hide($('import-error'));
            hide($('import-result'));
            var data = new FormData();
            data.append('file', file);
            fetch('/api/imports', { method: 'POST', body: data })
                .then(function (response) {
                    if (!response.ok) throw new Error('Import request failed');
                    return response.json();
                })
                .then(function (result) {
                    renderResult(result);
                    setText('import-status', 'Aktarım tamamlandı.');
                    form.reset();
                })
                .catch(function () {
                    var error = $('import-error');
                    if (error) {
                        error.textContent = 'Dosya aktarılamadı. Dosya türünü ve zorunlu kolonları kontrol edin.';
                        show(error);
                    }
                    setText('import-status', '');
                })
                .finally(function () {
                    button.disabled = false;
                    button.removeAttribute('aria-busy');
                });
        });
    }

    document.addEventListener('DOMContentLoaded', bindForm);
})();
