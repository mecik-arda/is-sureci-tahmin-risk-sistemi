# AI Destekli İş Süreci Tahmin ve Gecikme Risk Sistemi

Geçmiş süreç kayıtlarından gecikme riski, tahmini tamamlanma süresi ve
açıklanabilir karar desteği sağlayan tamamen yerel çalışan full stack
uygulama.

> **Önemli:** Bu uygulama yalnızca `127.0.0.1` üzerinde çalışır.
> Gerçek veriler harici ortamlara (bulut, drive, repo) yüklenmez.

## Gereksinimler

- Python 3.11 veya uyumlu sürüm
- Windows / macOS / Linux

## Sıfırdan Kurulum

### 1. Sanal ortam ve bağımlılıklar

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Ortam değişkenleri

```bash
copy .env.example .env
```

`.env` dosyasında `APP_MODE=demo` (sentetik veri) veya `APP_MODE=local` (gerçek veri) seçilir.

### 3. Veritabanı

```bash
python scripts/init_db.py
```

Bu komut mevcut veritabanını **siler** ve migration'ları sıfırdan uygular. Veri kaybına dikkat edin.

### 4. Veri içe aktarma

**Demo modunda:**
```bash
python scripts/generate_demo_data.py
```

**Local modda (gerçek veri):** Web arayüzünden `/data-import` sayfasından CSV/XLSX dosyası yükleyin veya komut satırından:
```bash
python scripts/import_process_data.py "data/raw/boston_311_2024.csv"
```

İçe aktarma sırasında:
- Dosya SHA256 parmak iziyle mükerrer yükleme engellenir
- Kategorik değerler kanonik snake_case kodlara dönüştürülür
- Bilinmeyen kategoriler `unknown` olarak işaretlenir
- Açılış verisi değişmez snapshot olarak saklanır
- Veri kalite sorunları `data_quality_issues` tablosuna kaydedilir

### 5. Model eğitimi

```bash
python scripts/run_faz5_training.py
```

Bu komut:
- Train (Ocak-Ağustos) / Validation (Eylül) üzerinde TimeSeriesSplit CV
- Sınıflandırma (RandomForest + sigmoid calibration) ve Regresyon (ElasticNet log1p) modellerini eğitir
- Sonuçları `ml/results/faz5_metrics.json` dosyasına yazar

### 6. Model bundle oluşturma

```bash
python scripts/build_faz5_bundle.py
```

Eğitilen modelleri `artifacts/` dizinine `.joblib` olarak paketler ve `model_bundles` tablosuna kaydeder.

### 7. Modeli aktif hale getirme

```bash
python scripts/enrich_active_bundle_metrics.py
```

Bu komut son eğitilen bundle'ı aktif (`is_active=1`) olarak işaretler ve metriklerini günceller.

### 8. Uygulamayı başlatma

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Uygulama açıldıktan sonra:
- Ana sayfa: http://127.0.0.1:8000
- API dokümantasyonu (Swagger): http://127.0.0.1:8000/docs
- Sağlık kontrolü: http://127.0.0.1:8000/health
- Hazırlık kontrolü: http://127.0.0.1:8000/ready

## Arayüz Turu

### Dashboard (`/`)
- Tahmin KPI'ları (son 30 günlük tahminler)
- Gerçekleşen sonuçlar (tamamlanan, geciken, SLA içinde)
- Günlük tahmin hacmi grafiği
- Süreç türü dağılımı

### Süreç Listesi (`/processes`)
- Durum ve risk filtresi
- Sayfalama
- Çoklu seçim ve toplu tahmin butonu
- Tahmin sonuçları Excel/CSV'e aktarma

### Süreç Detayı (`/processes/{id}`)
- **Tahmin kartı:** Kalibre edilmiş gecikme olasılığı, tahmini süre
- **İkili bilgi kartı:** Açılış anı bilgileri / Güncel durum
- **XAI paneli:** Per-instance SHAP (bu sürece özel risk faktörleri) + Global permutation importance
- **Benzer süreçler:** KNN ile opening-v1 feature uzayında en yakın 5 geçmiş süreç
- **Senaryo simülasyonu (What-If):** 10 opening-v1 feature değiştirilerek varsayımsal tahmin
- **Geri bildirim:** Doğruluk (backend türetir) ve fayda (kullanıcı yorumu)
- **Tahmin Üret:** Aktif model varken görünen kontrollü aksiyon

### Veri Aktar (`/data-import`)
- CSV ve XLSX dosyası yükleme (maks. 512 MB)
- İçe aktarma sonuç özeti

### Model Performansı (`/model-performance`)
- Aktif model bilgisi (versiyon, eşik, kalibrasyon)
- Sınıflandırma Validation metrikleri (PR-AUC, ROC-AUC, F1, Precision, Recall)
- Regresyon Validation metrikleri (MAE, MedAE, RMSE, p90 AE)
- CV metrikleri
- Karmaşıklık matrisi (confusion matrix)
- Model kartı: `MODEL_KARTI.md`

### Model İzleme (`/model-monitoring`)
- Aktif bundle sürümü, eğitim zamanı
- Schema/mapping sürümü, karar eşiği
- Artifact hash, analysis cache metrikleri

### Veri Kalitesi (`/data-quality`)
- Son import özeti (toplam, başarılı, atlanan, hatalı)
- Issue dağılımı (issue_code'a göre gruplu)
- İçe aktarma geçmişi

## Degraded Mod (Modelsiz Çalışma)

Uygulama başladığında aktif model aranır. Model bulunamazsa uygulama kapanmaz; **degraded** modda çalışmaya devam eder.

- `/health` → her zaman HTTP 200 (uygulama canlı)
- `/ready` → model yoksa HTTP 503 `MODEL_UNAVAILABLE`
- Model gerektiren işlemler → HTTP 503 `MODEL_UNAVAILABLE`

## Hata Kodları

| Kod | Anlam |
|-----|-------|
| `MODEL_UNAVAILABLE` | Aktif model bulunamadı veya yüklenemedi |
| `PROCESS_NOT_FOUND` | Belirtilen süreç kaydı bulunamadı |
| `INVALID_SIMULATION_FIELD` | Simülasyon override'ında izin verilmeyen değer |
| `FEEDBACK_REJECTED` | Geri bildirim reddedildi (örn. simülasyon kaydına) |
| `SNAPSHOT_NOT_FOUND` | Süreç için açılış snapshot'ı bulunamadı |
| `PREDICTION_COMPATIBILITY_ERROR` | Model-snapshot şema uyumsuzluğu |

Tüm hata yanıtları `{"error_code": "...", "message": "...", "details": {}, "request_id": "uuid"}` formatındadır.

## Çalışma Ortamları

`.env` dosyasındaki `APP_MODE` ile seçilir:

| Ayar | `APP_MODE=demo` | `APP_MODE=local` |
|---|---|---|
| Kullanım | Sentetik veri, geliştirme | Gerçek veri, üretim |
| Veritabanı | `data/demo.db` | `data/process_risk.db` |
| Model dizini | `artifacts/demo/` | `artifacts/production/` |

Demo ve local verileri tamamen ayrıdır; karışmaz.

## Testler

```bash
python -m pytest tests/ -v
```

Testler harici ağ bağlantısı olmadan çalışır. 373 test, Windows ACL sorunu giderilmiştir.

## Veri Profilleme ve Temizleme

```bash
# Veri profilleme raporu (JSON + Markdown)
python scripts/data_profiling.py

# Veri temizleme raporu
python scripts/clean_data.py

# Canlı model drift değerlendirmesi (son 30 gün)
python scripts/evaluate_live.py --days 30

# Güvenlik denetimi
python scripts/security_audit.py
```

## Sık Sorulan Sorular

**S: Uygulama neden 503 dönüyor?**
C: Aktif model yüklenmemiş. `scripts/build_faz5_bundle.py` ve `scripts/enrich_active_bundle_metrics.py` adımlarını tamamladığınızdan emin olun.

**S: Veritabanına nasıl veri yüklerim?**
C: Demo modunda `scripts/generate_demo_data.py`, local modda web arayüzünden `/data-import` sayfasından veya `scripts/import_process_data.py` ile.

**S: Tahminler neden üretilmiyor?**
C: SLA tanımlı olmayan süreçlerde sınıflandırma yapılamaz. Ayrıca modelin yüklü ve aktif olduğundan emin olun.

**S: Model eğitimi ne kadar sürer?**
C: 175 bin kayıtla yaklaşık 5-15 dakika (donanıma bağlı). Demo verisiyle (8 bin kayıt) 1-2 dakika.

**S: SHAP açıklamaları neden bazı süreçlerde görünmüyor?**
C: SHAP yalnızca RandomForest sınıflandırıcı ile çalışır. Ayrıca snapshot veya model yoksa per-instance XAI üretilemez.

## Proje Yapısı

```
app/                    # FastAPI uygulaması
├── api/                # REST endpoint'ler (health, imports, predictions)
├── core/               # Config, database, errors, runtime
├── models/             # ORM modelleri (7 tablo)
├── repositories/       # Veritabanı CRUD
├── schemas/            # Pydantic şemaları
├── services/           # İş mantığı (prediction, import, simulation, XAI...)
├── templates/          # Jinja2 HTML şablonları
├── static/             # CSS, JS
└── web/                # Frontend HTML route'ları + API route'ları
ml/                     # Makine öğrenmesi
├── config/             # Feature schema, label catalog, cleaning config
├── datasets/           # Dataset builder, target builder
├── features/           # Feature derivation, preprocessing, schema loader
├── training/           # Classifier/regressor trainer
├── evaluation/         # Metrics, comparison
├── xai/                # Global permutation + per-instance SHAP
└── similarity/         # KNN similarity
alembic/                # Migration altyapısı (005 migration)
scripts/                # Kurulum, eğitim, denetim betikleri
tests/                  # 373 test (birim + entegrasyon + E2E)
data/                   # SQLite veritabanları (gitignore)
artifacts/              # Model bundle .joblib dosyaları (gitignore)
```

## İlgili Belgeler

- Model Kartı: `MODEL_KARTI.md`
- Geliştirme Planı: `dosyalar/plan.md`
- Mimari Sözleşmeler: `dosyalar/kurallar.md`
- Gereksinim Denetimi: `dosyalar/plan_vs_gereksinim_denetimi.md`
- Çalışma Günlüğü: `dosyalar/rapor.md`
