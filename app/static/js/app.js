async function loadBanner() {
    try {
        const resp = await fetch('/api/banner');
        if (!resp.ok) throw new Error('Banner request failed');
        const data = await resp.json();
        const area = document.getElementById('banner-area');
        if (!area) return;
        let banners = [];
        if (data.app_mode === 'demo') {
            banners.push('<div class="banner banner-demo"><strong>Demo Modu</strong> — Bu ortam demo/sentetik veriler kullanmaktadır. Gösterilen sonuçlar gerçek operasyon verisi değildir.</div>');
        }
        if (!data.model_available) {
            banners.push('<div class="banner banner-degraded"><strong>Model Kullanılamıyor</strong> — Yeni tahmin üretilemez. Mevcut süreç bilgilerini görüntüleyebilirsiniz.</div>');
        }
        if (data.bundle_stage === 'production_candidate') {
            banners.push('<div class="banner banner-candidate"><strong>Aday Model</strong> — Bu model nihai test değerlendirmesinden henüz geçirilmemiştir.</div>');
        }
        area.innerHTML = banners.join('');
    } catch(e) {
        console.error('Banner load failed:', e);
    }
}
document.addEventListener('DOMContentLoaded', loadBanner);
window.addEventListener('unhandledrejection', function () {
    const area = document.getElementById('banner-area');
    if (!area || area.querySelector('[data-runtime-error]')) return;
    const banner = document.createElement('div');
    banner.className = 'banner banner-degraded';
    banner.dataset.runtimeError = 'true';
    banner.textContent = 'Sayfanın bazı bölümleri yüklenemedi. Lütfen sayfayı yenileyin.';
    area.appendChild(banner);
});
