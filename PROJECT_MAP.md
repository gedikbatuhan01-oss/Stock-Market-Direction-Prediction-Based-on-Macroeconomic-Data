# 🗺️ PROJECT MAP — Agent Context

> Bu dosya projenin haritasıdır. Her sprint sonunda güncellenir.
> AI agent bu dosyayı okuyarak projenin neresinde olduğumuzu anlar.

---

## 📌 Proje Bilgileri

| Alan | Değer |
|------|-------|
| **Proje Adı** | Bull/Bear Market Classifier |
| **Problem** | t+1 yön tahmini (ikili sınıflandırma) |
| **Ana Metrik** | MCC (Matthews Correlation Coefficient) |
| **Veri** | Nasdaq 10-yıllık piyasa verisi, 7 finansal feature |
| **Modeller** | LogReg, GRU, CNN1D, LSTM, TCN, Transformer, RF, XGBoost, LightGBM, CatBoost |
| **Config** | YAML + override sistemi |
| **Seed** | 42 (global) |

---

## 📂 Klasör Yapısı ve Durumu

```
project/
├── configs/
│   └── base.yaml              ← [x] Tam config (tüm modeller dahil)
│
├── codes/                     ← Orijinal monolitik kaynak kodlar (referans)
│   ├── V6_ANN.py              ← 8890 satır — Colab all-in-one script
│   └── final_model_comparison.py
│
├── data/
│   ├── raw/
│   │   ├── market_data_10y_enriched.csv  ← [x] Ana veri dosyası
│   │   └── market_data_10y_zscore.csv    ← [x] Z-score versiyonu
│   └── processed/
│       ├── X_tensor.npy                  ← [x] Önceden işlenmiş feature tensor
│       └── y_tensor.npy                  ← [x] Önceden işlenmiş label tensor
│
├── src/
│   ├── utils/
│   │   ├── config.py          ← [x] YAML yükleme, override, validate
│   │   ├── seed.py            ← [x] Global seed ayarlama
│   │   ├── io.py              ← [x] Klasör, JSON, CSV, artifact paths
│   │   └── npy_opener.py      ← [x] NumPy dosya inceleme aracı
│   │
│   ├── data/
│   │   ├── load_data.py       ← [x] CSV yükleme, tarih parse, sıralama
│   │   ├── preprocess.py      ← [x] NaN politikası, kolon seçimi
│   │   ├── splitters.py       ← [x] Dev/test split, expanding folds
│   │   ├── labeling.py        ← [x] Forward return, threshold, etiket
│   │   └── sequence_builder.py← [x] Lookback pencere, neutral eleme
│   │
│   ├── models/
│   │   ├── logistic_regression.py    ← [x] sklearn LogReg wrapper
│   │   ├── gru_model.py             ← [x] PyTorch GRU
│   │   ├── cnn1d_model.py           ← [x] PyTorch CNN1D
│   │   ├── lstm_model.py            ← [x] PyTorch LSTM
│   │   ├── tcn_model.py             ← [x] PyTorch TCN
│   │   ├── transformer_encoder_model.py ← [x] PyTorch Transformer
│   │   ├── tree_models.py           ← [x] RF, XGBoost, LightGBM, CatBoost
│   │   └── model_factory.py         ← [x] Model seçim hub'ı (tüm modeller)
│   │
│   ├── training/
│   │   ├── trainer.py         ← [x] sklearn + torch training loop
│   │   ├── losses.py          ← [x] Weighted BCE/CE
│   │   └── callbacks.py       ← [x] EarlyStopping
│   │
│   ├── evaluation/
│   │   ├── metrics.py                    ← [x] MCC, F1, BalAcc, confusion
│   │   ├── reports.py                    ← [x] JSON rapor, CSV özet
│   │   ├── final_model_comparison.py     ← [x] Model karşılaştırma & seçim
│   │   └── visiulation_artificial.py     ← [x] Görselleştirme
│   │
│   └── run_experiment.py      ← [x] Ana orkestrasyon (sklearn + torch)
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── predictions/
│   └── plots/
│
├── .gitignore                 ← [x]
├── requirements.txt           ← [x]
├── CHANGELOG.md               ← [x]
├── PROJECT_MAP.md             ← [x] (bu dosya)
├── README.md                  ← [ ] Bekliyor
├── LICENSE                    ← [ ] Bekliyor (lisans seçimi)
└── experiments_summary.csv    ← [ ] Otomatik üretilecek (ilk çalıştırmada)
```

**Gösterge:** `[x]` tamamlandı · `[/]` devam ediyor · `[ ]` henüz başlanmadı

---

## 🏃 Sprint Durumu

| Sprint | Hedef | Durum |
|--------|-------|-------|
| **Sprint 0** | Proje altyapısı + dokümantasyon | ✅ Tamamlandı |
| **Sprint 1** | Config + Utils + Data + LogReg baseline | ✅ Tamamlandı |
| **Sprint 2** | GRU + Neural training loop | ✅ Tamamlandı |
| **Sprint 3** | CNN1D + LSTM + TCN + Transformer + Tree models | ✅ Tamamlandı |
| **GitHub** | Repo oluşturma + push | ⏳ Kullanıcıdan bilgi bekleniyor |
| **README** | Proje dokümantasyonu | ⏳ Bekliyor |

---

## 🔒 Kritik Kurallar (İhlal Edilemez)

1. **Shuffle YOK** — Kronolojik sıra korunmalı
2. **Random split YOK** — Zaman serisi split kullanılacak
3. **Final test en başta ayrılır** — Hiçbir seçimde kullanılmaz
4. **Threshold sadece fold train'den** — Val/test'e aynı sabit uygulanır
5. **Scaler sadece fold train'de fit** — Val/test sadece transform
6. **Neutral çıkarılır** — Label üretiminden sonra
7. **Leakage yasak** — Val/test bilgisi train'e sızamaz

---

## 📝 Tamamlanan Dosya Listesi (25 kaynak dosya)

| # | Dosya | Satır | Kaynak |
|---|-------|-------|--------|
| 1 | `configs/base.yaml` | ~80 | PDF spec |
| 2 | `src/utils/config.py` | ~60 | Yeni yazıldı |
| 3 | `src/utils/seed.py` | ~25 | Yeni yazıldı |
| 4 | `src/utils/io.py` | ~65 | Yeni yazıldı |
| 5 | `src/data/load_data.py` | ~30 | Yeni yazıldı |
| 6 | `src/data/preprocess.py` | ~41 | V6_ANN.py |
| 7 | `src/data/splitters.py` | ~90 | V6_ANN.py |
| 8 | `src/data/labeling.py` | ~83 | V6_ANN.py |
| 9 | `src/data/sequence_builder.py` | ~102 | V6_ANN.py |
| 10 | `src/evaluation/metrics.py` | ~107 | V6_ANN.py |
| 11 | `src/models/logistic_regression.py` | ~24 | V6_ANN.py |
| 12 | `src/models/gru_model.py` | ~75 | V6_ANN.py |
| 13 | `src/models/cnn1d_model.py` | ~81 | V6_ANN.py |
| 14 | `src/models/lstm_model.py` | ~59 | V6_ANN.py |
| 15 | `src/models/tcn_model.py` | ~102 | V6_ANN.py |
| 16 | `src/models/transformer_encoder_model.py` | ~72 | V6_ANN.py |
| 17 | `src/models/tree_models.py` | ~86 | V6_ANN.py |
| 18 | `src/models/model_factory.py` | ~103 | V6_ANN.py |
| 19 | `src/training/trainer.py` | ~612 | V6_ANN.py |
| 20 | `src/training/losses.py` | ~42 | V6_ANN.py |
| 21 | `src/training/callbacks.py` | ~68 | V6_ANN.py |
| 22 | `src/evaluation/reports.py` | ~56 | V6_ANN.py |
| 23 | `src/evaluation/final_model_comparison.py` | ~562 | codes/ |
| 24 | `src/evaluation/visiulation_artificial.py` | - | Kullanıcıdan |
| 25 | `src/run_experiment.py` | ~635 | V6_ANN.py |

---

## 📚 Referanslar

1. Chung et al. (2014) — GRU evaluation
2. Fischer & Krauss (2018) — Deep learning financial markets
3. Kingma & Ba (2015) — Adam optimizer
4. Boughorbel et al. (2017) — MCC for imbalanced data
5. Chicco & Jurman (2020) — MCC vs F1 vs accuracy
6. Estrella & Mishkin (1996/98) — Yield Curve predictor
7. Johnson (2011) — VIX Term Structure

---

*Son güncelleme: 2026-05-11 — Tüm Sprint'ler tamamlandı, GitHub push bekleniyor*
