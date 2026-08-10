# Model Kartı — AI Destekli İş Süreci Tahmin ve Gecikme Risk Sistemi

**Model sürümü:** `faz5-production-candidate-v1`
**Oluşturma tarihi:** 2026-08-10

---

## 1. Modelin Amacı ve Kullanılmaması Gereken Durumlar

**Amaç:** Belediye hizmet kayıtlarının (Boston 311) açılış anındaki özelliklerine bakarak:
- Sürecin SLA hedefini aşıp aşmayacağını (sınıflandırma),
- Sürecin tahmini toplam tamamlanma süresini (regresyon) öngörmek.

**Kullanılmaması gereken durumlar:**
- Kesin hukuki veya İK kararı olarak kullanılamaz.
- Personel performans değerlendirmesi için kullanılamaz.
- Farklı şehir/ülke verilerinde yeniden eğitilmeden kullanılamaz.
- Süreç içi (mid-process) tahmin için tasarlanmamıştır; yalnızca açılış anı tahminidir.
- SLA tanımlı olmayan süreçler için sınıflandırma yapamaz.

---

## 2. Eğitim Verisi

| Özellik | Değer |
|---------|-------|
| Veri kaynağı | Boston 311 Service Requests, 2024 |
| Toplam ham kayıt | 282.836 |
| Sınıflandırma kohortu | 256.712 kayıt (%72 negatif / %28 gecikmeli) |
| Regresyon kohortu | 242.033 kayıt (kapanmış ve geçerli süreli) |
| Train dönemi | Ocak – Ağustos 2024 (175.365 kayıt) |
| Validation dönemi | Eylül 2024 (25.246 kayıt) |
| Test dönemi | Ekim – Kasım 2024 (38.232 kayıt) — MÜHÜRLÜ |
| Audit dönemi | Aralık 2024 (17.869 kayıt) — MÜHÜRLÜ |
| Observation cutoff | 2025-01-13T00:00:00 (Analyze Boston metadata) |

---

## 3. Kullanılan Özellikler (10 Feature, Opening-V1)

**Kategorik (5):** `source`, `subject`, `reason`, `type`, `neighborhood`
- Ham İngilizce değerler kanonik `snake_case` kodlara dönüştürülür.
- Bilinmeyen değerler `"unknown"` olarak işaretlenir.

**Sayısal (5 — `open_dt` ve `sla_target_dt` türevi):**
| Feature | Aralık |
|---------|--------|
| `open_month` | 1–12 |
| `open_weekday` | 0–6 |
| `open_hour` | 0–23 |
| `is_weekend` | 0–1 |
| `sla_duration_hours` | ≥ 0 (NaN = SLA yok) |

**V1'de kullanılmayan alanlar (leakage önlemi):**
- `case_status`, `on_time`, `closure_reason`, `closed_dt`, `completed_at` → outcome alanları
- `department`, `queue` → opening-time değişmezliği kanıtlanmamış
- `latitude`, `longitude` → `neighborhood` mekânsal sinyali zaten taşıyor

---

## 4. Train/Validation/Test Ayrımı

- **Yöntem:** Kronolojik (zamana dayalı), rastgele split değil.
- **Split sınırları:** Ay bazında hardcode edilmiş tarih aralıkları.
- **Preprocessing:** Yalnızca Train kümesi üzerinde fit edilir; Validation/Test/Audit'e transform uygulanır.
- **Sealed holdout:** Test (Ekim–Kasım) ve Audit (Aralık) kümeleri model geliştirme boyunca mühürlü kalmıştır; metrik hesaplanmamıştır.

---

## 5. Model Algoritması ve Hiperparametreler

### Sınıflandırma: RandomForestClassifier

| Hiperparametre | Değer |
|---------------|-------|
| `n_estimators` | 300 |
| `max_depth` | None |
| `class_weight` | balanced |
| `min_samples_leaf` | 10 |

**Kalibrasyon:** Sigmoid (Platt scaling), Train OOF probability'ler üzerinde fit edilmiştir.

### Regresyon: ElasticNet (log1p dönüşümü)

| Hiperparametre | Değer |
|---------------|-------|
| `alpha` | 0.5 |
| `l1_ratio` | 0.5 |

**Dönüşüm:** `TransformedTargetRegressor` ile `log1p` / `expm1` kullanılmıştır. Winsorization yapılmamıştır.

### Tuning protokolü:
- Train içinde `TimeSeriesSplit(n_splits=5)` expanding-window CV
- Random KFold/shuffle kullanılmaz
- Hyperparameter search her fold'da fit
- Calibration yalnız Train OOF predictions üzerinde
- Test ve Audit hiçbir aşamada açılmamıştır

---

## 6. Performans Metrikleri

### Sınıflandırma (Validation — Eylül 2024)

| Metrik | CV (mean ± std) | Validation |
|--------|-----------------|-----------|
| PR-AUC | 0.7718 ± 0.023 | **0.8175** |
| ROC-AUC | 0.9082 ± 0.011 | 0.9247 |
| Brier Score | — | **0.1009** |
| F1 Score | — | 0.7724 |
| Precision | — | 0.681 |
| Recall | — | 0.893 |

**Karar eşiği:** `threshold = 0.35` (Validation F1 maksimizasyonu ile seçilmiştir)

### Regresyon (Validation — Eylül 2024)

| Metrik | CV (mean ± std) | Validation |
|--------|-----------------|-----------|
| MAE | 191.64h ± 6.4 | **104.99h** |
| MedianAE | — | 12.56h |
| RMSE | 685.20h ± 25.8 | 332.64h |
| p90 AE | — | 215.02h |

---

## 7. Risk Eşikleri ve Seçim Gerekçesi

**Mevcut durum (V1):**
- `delay_probability`: Kalibre edilmiş gecikme olasılığı (0–1 arası)
- `predicted_is_delayed`: `delay_probability >= 0.35` → binary karar
- `risk_score`: `delay_probability * 100` (0–100 arası)

**V1'de 3-seviyeli risk bandı (Düşük/Orta/Yüksek) kaldırılmıştır.** Gerekçe:
- Binary karar daha net operasyonel sinyal sağlar.
- 3-seviyeli bant keyfî eşik dayatmasıdır (0–39 / 40–69 / 70–100).
- Bu sapma bilinçlidir; mentor onayına tabidir.

---

## 8. Bilinen Sınırlamalar

1. **Yalnızca açılış anı tahmini:** Süreç ilerledikçe değişen riski ölçemez.
2. **Boston 311 verisine özgü:** Farklı şehir/kurum verisinde yeniden eğitilmeden kullanılamaz.
3. **SLA zorunluluğu:** SLA tanımlı olmayan süreçlerde sınıflandırma yapılamaz (~%8).
4. **Per-instance XAI sınırlı:** SHAP değerleri yalnız feature katkısını gösterir; neden-sonuç ilişkisi değildir.
5. **Test/Audit mühürlü:** Nihai genelleme performansı Validation metrikleriyle sınırlıdır.
6. **Medyan süre 12.2h:** Çok kısa süreli işler baskın; MAE 105h ile yüksek görünebilir ancak medyan hata 12.6h'dir.
7. **Dengesiz sınıf (%28 pozitif):** `class_weight=balanced` ile ele alınmıştır.

---

## 9. Model Sürümü ve Artifact

| Özellik | Değer |
|---------|-------|
| Model versiyonu | `faz5-production-candidate-v1` |
| Stage | `production_candidate` |
| Eğitim tarihi | Faz 5 kapanışı |
| Artifact SHA256 | Doğrulanmış, secure loader'dan geçmiş |
| Feature schema | `opening-v1` |
| Canonical mapping | `1.0.0` |
| Label catalog | `1.0.0` (Türkçe sunum) |
| scikit-learn sürümü | 1.7.2 |

---

## 10. Veri Gizliliği

- Tüm veriler yerel ortamda işlenir.
- Gerçek veriler hiçbir harici depo veya servise gönderilmemiştir.
- Harici AI/LLM servislerine gerçek satır verisi gönderilmez.
- Sunum ve raporlarda yalnızca anonimleştirilmiş veya sentetik örnekler kullanılır.
