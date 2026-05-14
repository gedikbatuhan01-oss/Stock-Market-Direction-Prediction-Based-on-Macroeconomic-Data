# Building Scheme — Bull/Bear Market Classifier

> Projenin mimari iskeletini, görev listesini ve tamamlanma durumunu tek belgede gösterir.
> Gösterge: ✅ tamamlandı · ⏳ bekliyor · ❌ henüz başlanmadı

---

## 1. Proje Mimarisi — Klasör Yapısı

```
Proje_Antigrav/
│
├── configs/
│   └── ✅ base.yaml                     ← Tüm model/veri/eğitim parametreleri
│
├── codes/                               ← Orijinal referans kodlar (değiştirilmez)
│   ├── ✅ V6_ANN.py                     ← 8890 satır Colab all-in-one script
│   └── ✅ final_model_comparison.py     ← Referans karşılaştırma scripti
│
├── data/
│   ├── raw/
│   │   ├── ✅ market_data_10y_enriched.csv   ← Ana veri (10 yıl Nasdaq, 7 feature)
│   │   └── ✅ market_data_10y_zscore.csv     ← Z-score normalize edilmiş versiyon
│   └── processed/
│       ├── ✅ X_tensor.npy                   ← Hazır feature tensörü
│       └── ✅ y_tensor.npy                   ← Hazır label tensörü
│
├── src/
│   ├── utils/
│   │   ├── ✅ config.py                 ← YAML yükleme, override, doğrulama
│   │   ├── ✅ seed.py                   ← Global seed yönetimi (seed=42)
│   │   ├── ✅ io.py                     ← Klasör, JSON, CSV, artifact path yardımcıları
│   │   ├── ✅ npy_opener.py             ← NumPy dosya inceleme aracı
│   │   └── ✅ __init__.py
│   │
│   ├── data/
│   │   ├── ✅ load_data.py              ← CSV yükleme, tarih parse, kronolojik sıralama
│   │   ├── ✅ preprocess.py             ← NaN politikası, kolon seçimi
│   │   ├── ✅ splitters.py              ← Dev/test split, expanding window fold üretici
│   │   ├── ✅ labeling.py               ← Forward return hesaplama, threshold, etiket üretimi
│   │   ├── ✅ sequence_builder.py       ← Lookback pencere oluşturma, neutral label eleme
│   │   └── ✅ __init__.py
│   │
│   ├── models/
│   │   ├── ✅ logistic_regression.py    ← sklearn LogisticRegression wrapper
│   │   ├── ✅ gru_model.py              ← PyTorch GRU
│   │   ├── ✅ cnn1d_model.py            ← PyTorch 1D CNN
│   │   ├── ✅ lstm_model.py             ← PyTorch LSTM
│   │   ├── ✅ tcn_model.py              ← PyTorch Temporal Convolutional Network
│   │   ├── ✅ transformer_encoder_model.py ← PyTorch Transformer Encoder
│   │   ├── ✅ tree_models.py            ← RandomForest, XGBoost, LightGBM, CatBoost
│   │   ├── ✅ model_factory.py          ← Model seçim merkezi (10 model)
│   │   └── ✅ __init__.py
│   │
│   ├── training/
│   │   ├── ✅ trainer.py                ← sklearn + PyTorch birleşik eğitim döngüsü
│   │   ├── ✅ losses.py                 ← Weighted BCE / Cross-Entropy kayıp fonksiyonları
│   │   ├── ✅ callbacks.py              ← EarlyStopping callback
│   │   └── ✅ __init__.py
│   │
│   ├── evaluation/
│   │   ├── ✅ metrics.py                ← MCC, F1, Balanced Accuracy, Confusion Matrix
│   │   ├── ✅ reports.py                ← JSON rapor üretimi, CSV özet
│   │   ├── ✅ final_model_comparison.py ← Model karşılaştırma ve seçim
│   │   ├── ✅ visiulation_artificial.py ← Görselleştirme araçları
│   │   └── ✅ __init__.py
│   │
│   ├── ✅ run_experiment.py             ← Ana orkestrasyon scripti
│   └── ✅ __init__.py
│
├── artifacts/                           ← Çalıştırma sonrası otomatik dolacak
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   └── plots/
│
├── ✅ .gitignore
├── ✅ requirements.txt
├── ✅ CHANGELOG.md
├── ✅ PROJECT_MAP.md
├── ✅ Building_scheme.md               ← (bu dosya)
├── ⏳ README.md                        ← Proje açıklaması yazılacak
├── ⏳ LICENSE                          ← Lisans seçimi ve eklenmesi
└── ❌ experiments_summary.csv          ← İlk çalıştırmada otomatik üretilecek
```

---

## 2. Sprint Görev Listesi

| Sprint | Hedef | Durum |
|--------|-------|-------|
| **Sprint 0** | Proje altyapısı + dokümantasyon + `.gitignore` + `requirements.txt` | ✅ Tamamlandı |
| **Sprint 1** | `configs/base.yaml` + tüm `utils/` + tüm `data/` + LogReg baseline | ✅ Tamamlandı |
| **Sprint 2** | `gru_model.py` + `trainer.py` + `losses.py` + `callbacks.py` | ✅ Tamamlandı |
| **Sprint 3** | CNN1D + LSTM + TCN + Transformer + Tree modeller + evaluation + `run_experiment.py` | ✅ Tamamlandı |
| **GitHub** | Repo oluşturma + tüm dosyaları push etme | ⏳ Bekliyor |
| **README** | Proje açıklaması, kurulum, kullanım talimatları | ⏳ Bekliyor |

---

## 3. Kaynak Dosya Tamamlanma Listesi (25 dosya)

| # | Dosya | Satır | Durum |
|---|-------|-------|-------|
| 1 | `configs/base.yaml` | ~80 | ✅ |
| 2 | `src/utils/config.py` | ~60 | ✅ |
| 3 | `src/utils/seed.py` | ~25 | ✅ |
| 4 | `src/utils/io.py` | ~65 | ✅ |
| 5 | `src/data/load_data.py` | ~30 | ✅ |
| 6 | `src/data/preprocess.py` | ~41 | ✅ |
| 7 | `src/data/splitters.py` | ~90 | ✅ |
| 8 | `src/data/labeling.py` | ~83 | ✅ |
| 9 | `src/data/sequence_builder.py` | ~102 | ✅ |
| 10 | `src/evaluation/metrics.py` | ~107 | ✅ |
| 11 | `src/models/logistic_regression.py` | ~24 | ✅ |
| 12 | `src/models/gru_model.py` | ~75 | ✅ |
| 13 | `src/models/cnn1d_model.py` | ~81 | ✅ |
| 14 | `src/models/lstm_model.py` | ~59 | ✅ |
| 15 | `src/models/tcn_model.py` | ~102 | ✅ |
| 16 | `src/models/transformer_encoder_model.py` | ~72 | ✅ |
| 17 | `src/models/tree_models.py` | ~86 | ✅ |
| 18 | `src/models/model_factory.py` | ~103 | ✅ |
| 19 | `src/training/trainer.py` | ~612 | ✅ |
| 20 | `src/training/losses.py` | ~42 | ✅ |
| 21 | `src/training/callbacks.py` | ~68 | ✅ |
| 22 | `src/evaluation/reports.py` | ~56 | ✅ |
| 23 | `src/evaluation/final_model_comparison.py` | ~562 | ✅ |
| 24 | `src/evaluation/visiulation_artificial.py` | — | ✅ |
| 25 | `src/run_experiment.py` | ~635 | ✅ |

---

## 4. Bekleyen Maddeler

| Madde | Açıklama | Durum |
|-------|----------|-------|
| `README.md` | Proje tanıtımı, kurulum adımları, kullanım örnekleri | ⏳ Bekliyor |
| `LICENSE` | Lisans türü seçimi (MIT, Apache 2.0 vb.) ve dosya eklenmesi | ⏳ Bekliyor |
| GitHub Push | Remote repo oluşturulması ve tüm kaynak kodun yüklenmesi | ⏳ Kullanıcıdan bilgi bekleniyor |
| `experiments_summary.csv` | `run_experiment.py` ilk çalıştırıldığında `artifacts/` altında otomatik üretilir | ❌ Henüz üretilmedi |

---

## 5. Modeller

| Model | Tür | Durum |
|-------|-----|-------|
| Logistic Regression | sklearn | ✅ |
| GRU | PyTorch | ✅ |
| CNN1D | PyTorch | ✅ |
| LSTM | PyTorch | ✅ |
| TCN | PyTorch | ✅ |
| Transformer Encoder | PyTorch | ✅ |
| Random Forest | sklearn | ✅ |
| XGBoost | xgboost | ✅ |
| LightGBM | lightgbm | ✅ |
| CatBoost | catboost | ✅ |

---

## 6. Kritik Kurallar (İhlal Edilemez)

1. **Shuffle YOK** — Kronolojik sıra her adımda korunmalı
2. **Random split YOK** — Yalnızca zaman serisi tabanlı expanding window split
3. **Final test en başta ayrılır** — Model seçimi veya hiperparametre optimizasyonunda kullanılmaz
4. **Threshold yalnızca fold train'den hesaplanır** — Aynı sabit val/test'e uygulanır
5. **Scaler yalnızca fold train'de fit edilir** — Val/test yalnızca transform görür
6. **Neutral label'lar çıkarılır** — Etiket üretiminden sonra, sequence öncesinde
7. **Leakage yasak** — Val/test seti bilgisi train aşamasına sızmaz

---

## 7. Temel Konfigürasyon

| Parametre | Değer |
|-----------|-------|
| Ana metrik | MCC (Matthews Correlation Coefficient) |
| Lookback penceresi | 21 gün |
| Expanding window fold sayısı | 4 |
| Final test oranı | %15 |
| Batch size | 64 |
| Early stopping patience | 10 epoch |
| Global seed | 42 |

---

*Son güncelleme: 2026-05-11 — Sprint 0–3 tamamlandı, GitHub push ve README bekliyor.*
