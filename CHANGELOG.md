# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-05-16

### Added

- `notebooks/08_full_pipeline.ipynb` — Bölüm 5b eklendi: `cv_candidates` listesinden tüm 320 adayı CV MCC'ye göre sıralayan ve her model tipinin en iyi varyantını gösteren hücre.
  - Reason: baseline run sonuçlarını rapor JSON'ından okuyup tablo halinde karşılaştırmak için.

- `notebooks/09_top_3_comparison.ipynb` — Baseline run'ın en iyi 3 model tipini (GRU, CNN1D, Transformer) seçip her birinin katman sayısı, boyut ve dropout parametrelerini 6 varyant × 3 model = 18 kombinasyon üzerinde sweep eden yeni notebook.
  - Her sweep `top_3_results/artifacts/` altına izole çıktı üretir; Drive'da `Top_3_results/` klasörüne kopyalanır.
  - Bölüm 4-5: her model kendi içinde ve genel sıralama (CV MCC azalan).
  - Bölüm 6: best-per-model bar chart + CV vs Test MCC scatter plot.
  - Bölüm 7: fold bazlı `train_loss` / `val_loss` eğrileri; `plot_loss(model, variant)` yardımcı fonksiyonu.

- `notebooks/10_threshold_free_validation.ipynb` — Notebook 09'da `test_mcc = 0` çıkan adayları sabit 0.50 eşikle yeniden değerlendiren notebook.
  - `decision.threshold_search.enabled = False` override'ı ile MCC-tabanlı eşik araması devre dışı.
  - Bölüm 3: nb09 vs nb10 yan yana karşılaştırma; `delta_test_mcc` hesaplanır, "eşik gürültüsü suçlu / aynı / sabit eşik daha kötü" etiketi üretir.
  - Çıktılar `threshold_free_results/` altına izole; Drive'da `Threshold_Free_Results/` klasörüne kopyalanır.

- `src/models/cnn1d_model.py` — `CNN1DClassifier` ve `build_cnn1d_model` fonksiyonlarına `num_conv_layers` parametresi eklendi (varsayılan 2, geriye dönük uyumlu).
  - Reason: notebook 09 sweep'lerinde CNN1D için katman sayısı varyasyonu yapılabilmesi.

### Fixed

- `notebooks/08_full_pipeline.ipynb` — cell-11'de `report` değişkeni Colab branch'inde (`if IN_COLAB: show_results()`) tanımlanmıyordu; cell-17 (confusion matrix) `NameError: name 'report' is not defined` fırlatıyordu.
  - Fix: JSON dosyası varsa `report` her zaman yükleniyor; Colab/lokal ayrımı sadece görüntüleme tarafında kalıyor.
  - cell-17 guard'ı `if not json_files` → `if report is None` olarak güncellendi.

- `toolkit.py` — `show_results()` içindeki `report.get("cv_results", [])` çağrısı rapor şemasındaki gerçek anahtar olan `cv_candidates` ile eşleşmiyordu; tüm CV tablosu "bulunamadı" yazıyordu.
  - Fix: `cv_results or cv_candidates` fallback zinciri eklendi.

### Findings (Deney Bulguları)

- **Baseline (v0.2.0) run — 320 aday, tüm modeller:**
  - En iyi CV MCC: GRU / robust / q20_lb42 → **0.0840** (std: 0.091).
  - Tüm 320 adayın CV MCC ≤ 0.084; varyans ortalamayla aynı mertebede — sinyal rastgelelikten istatistiksel olarak ayırt edilemiyor.
  - F1 yüksek (≈0.74) ama MCC düşük: model Bull sınıfına eğilim gösteriyor; F1 sınıf dengesizliğini yansıtıyor.

- **v0.1.0 → v0.2.0 karşılaştırması:**
  - Eski pipeline GRU'da CV MCC 0.209, test MCC 0.174 raporlamıştı.
  - Yeni pipeline'da aynı (model, scaler, quantile) kombinasyonu top-20'ye bile giremiyor.
  - Label-endpoint guardrail (split sınırında forward return label'ının test verisine sızması) eklenince skor düştü — eski 0.17 figürü metodolojik artefakt.

- **Notebook 09 — mimari sweep (18 varyant):**
  - **Katman sayısı artıkça CV MCC düşüyor** (GRU: l1 > l2 > l3; Transformer: l1 > l2 > l4). ~2000 eğitim örneğiyle derin modeller overfit ediyor.
  - **CNN1D c128_k3_l2** en yüksek CV MCC (0.1346) ama test_mcc = 0: MCC-tabanlı eşik araması degenerate noktaya (hepsi Bull) kaymış.
  - Degenerate satırların ortak işareti: test_f1 ≈ 0.747, test_bal_acc = 0.500, test_mcc = 0.
  - **PR-AUC eşikten bağımsız** ve en güvenilir metrik; CNN1D k5 varyantları PR-AUC'ta öne çıkıyor (0.651).
  - **En tutarlı sonuç:** GRU h64 l2 (CV=0.041, Test=0.043) — iki metrik birbirine yakın.
  - Tree-based modeller (XGBoost, CatBoost, RF) robust vs standard scaler'da **identik** skor üretiyor — beklendik, ölçeğe duyarsızlar; pipeline sanity check geçti.

- **Eşik çökmesi mekanizması:**
  - Model olasılıklarını ör. 0.55-0.62 aralığında üretiyor.
  - MCC-tabanlı arama internal val'de 0.58 eşiğini seçiyor.
  - Test setinde dağılım 0.59-0.65'e kayıyor → tüm örnekler ≥ 0.58 → hepsi Bull → MCC = 0.
  - Düzeltme: `threshold_search.enabled = False` (sabit 0.50) veya arama bandını 0.35-0.65'e daralt.

- **Pipeline bütünlük kontrolleri (split ve normalizasyon):**
  - Val her zaman train'in kronolojik devamı (`make_expanding_folds` expanding window).
  - Scaler her fold'da sadece train'de fit ediliyor (`scale_sequence_data`).
  - `shuffle_train = True`: batch sırası rastgele, split sınırları değil — leakage yok.
  - Feature window split sınırını geçebilir (v0.2.0 endpoint-based); label endpoint kendi split'inde kalıyor.
  - Train/val/test oranları: test %15 (sabit blok) | final train %76.5 | internal val %8.5 | CV val %8.5 (4 fold, expanding).

- **Öğrenememe kök nedenleri:**
  1. T+1 günlük yön EMH'nin en güçlü olduğu frekans — makro göstergelerden öğrenilecek stabil örüntü yok.
  2. Feature–label zaman ölçeği uyumsuzluğu: Yield Curve, VIX Term Structure haftalık–aylık etki ediyor; T+1'e sıkıştırılınca sinyal/gürültü çöküyor.
  3. Bilgi zaten fiyatlanmış — mevcut seviye değil, beklentiden sapma (surprise) taşıyıcı olurdu.
  4. ~1600-1800 effective eğitim örneği: sliding window'dan gelen örnekler birbirleriyle örtüşüyor, bağımsız örnek sayısı çok daha az.

- **Gelecek adaylar (tartışıldı, henüz implement edilmedi):**
  - Horizon'u T+1 → T+5 / T+21'e uzatmak (feature–label uyumu için en kritik değişiklik).
  - Cross-asset features: diğer endekslerin momentumu, relative strength, sektör rotasyon sinyali.
  - Feature türevleri: seviye yerine değişim hızı (`Yield_Curve_5d_change` gibi).
  - Multi-stock dataset augmentation: 4-5 ek hisse kendi labelıyla eğitim havuzuna eklenerek efektif örnek sayısı artırılabilir; ancak T+1 sinyal sorunu hisse başına devam eder.

## [0.2.0] - 2026-05-15

### Added
- Endpoint-based sequence construction for validation and test windows.
  - Reason: validation/test samples should be allowed to use already-known historical feature rows before the split boundary, while keeping label endpoints inside their own split.
- MCC-based probability threshold search using internal validation data.
  - Reason: a fixed `0.50` decision threshold can suppress MCC/F1 on weak and imbalanced financial signals; threshold selection must still avoid final-test leakage.
- Labeling and lookback experiment grids in `configs/base.yaml`.
  - Reason: `threshold_quantile` and `lookback` materially change neutral-drop rate and signal horizon, so they should be selected by cross-validation rather than hand-picked.
- Naive baseline reporting for `always_bull`, `always_bear`, `random`, and `previous_day_sign`.
  - Reason: low MCC is only meaningful when compared against simple non-learning strategies on the same endpoints.
- PR-AUC reporting alongside ROC-AUC, MCC, F1, balanced accuracy, and confusion matrix.
  - Reason: PR-AUC gives extra visibility when class balance shifts after neutral filtering.
- Focused tests for endpoint windows, split-safe label horizons, train-only threshold computation, and MCC threshold search.
  - Reason: the most important leakage and metric-selection rules should be locked down with regression tests.

### Changed
- Cross-validation now evaluates candidate combinations of label quantile and lookback before selecting the final model.
  - Reason: this makes model selection more defensible for noisy T+1 market-direction prediction.
- Sklearn final training now uses internal development validation only to choose the probability threshold, then retrains on all development samples before final test evaluation.
  - Reason: threshold tuning should not reduce final training data, but the final test set must remain untouched.
- Torch training can shuffle training batches via config while preserving chronological split boundaries.
  - Reason: batch order is not leakage once each sample is already a past-only sequence, and shuffling can improve neural optimization.
- Experiment summaries now include selected variant, selected probability threshold, PR-AUC, lookback, and threshold quantile.
  - Reason: final reports should explain which data/decision regime produced the selected MCC/F1.
- Notebook walkthroughs were synchronized with the latest endpoint-based pipeline, threshold-search flow, report schema, and model config keys.
  - Reason: the notebooks were added while the core training code was changing, so their examples needed to match the current source modules.

### Dependencies
- Added optional tree ensemble and test dependencies: `xgboost`, `lightgbm`, `catboost`, and `pytest`.
  - Reason: the comparison plan requires tree-based financial baselines and local regression tests.

## [0.1.0] — 2026-05-11

### Added
- Initial project scaffolding
- Directory structure: `configs/`, `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/utils/`
- `.gitignore` for Python ML projects
- `requirements.txt` with core dependencies
- `PROJECT_MAP.md` — agent context and project map
- `CHANGELOG.md` — this file
- `configs/base.yaml` — default experiment configuration
