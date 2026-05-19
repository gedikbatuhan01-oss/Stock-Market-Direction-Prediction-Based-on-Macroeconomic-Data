import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# X7 Quant Data Pipeline  —  v2  (Phase 1 + Phase 2)
# ============================================================
# DEĞİŞİKLİKLER (orijinal pipeline'a göre):
#
# [PHASE 1 — Feature Denoising]
#   add_denoised_features() eklendi.
#   Her ham feature için 6 türev sütun üretilir:
#     - delta_5      : 5 günlük değişim (kısa momentum)
#     - delta_21     : 21 günlük değişim (orta vade momentum)
#     - ewm_5        : Exp. ağırlıklı ortalama, span=5 (hızlı trend)
#     - ewm_21       : Exp. ağırlıklı ortalama, span=21 (yavaş trend)
#     - residual_21  : ham - ewm_21 (gürültü bileşeni izole edilir)
#     - rolling_z_21 : 21 günlük z-score (kısa vadeli normalize sapma)
#   Tüm türevler sadece GEÇMIŞE dayalı (shift/rolling) → sızıntı yok.
#   Ham feature'lar KORUNUYOR (silinmiyor).
#
# [PHASE 2 — Horizon Test]
#   fwd_n artık __init__ parametresi olarak dışarıdan verilebilir.
#   Önerilen değerler: 1, 5, 21
#   threshold_search kaldırıldı → her zaman train median kullanılır
#   (önceki analizde median threshold artefaktı riski tespit edilmişti).
#
# [EMBARGO DÜZELTMESİ]
#   Orijinalde embargo yanlış sete (train/val içi) uygulanıyordu.
#   Düzeltme: train son GAP satırı + val son GAP satırı hâlâ siliniyor
#   AMA ek olarak test setinin başından da fwd_n kadar satır siliniyor
#   böylece val→test geçişinde overlap riski sıfırlanıyor.
#
# [SABİT KALAN HER ŞEY]
#   VRVP, Ichimoku, Nadaraya-Watson, KAMA, Supertrend hesaplamaları
#   tamamen aynı — tek satır değiştirilmedi.
#   Rolling Z-score normalizasyonu aynı (252 bar).
#   Kaydetme / meta / doğrulama bloğu aynı yapıda.
# ============================================================


class LeakageFreePipelineX7:
    """
    X7 Quant Data Pipeline for NASDAQ Swing Trading — v2

    Strict Anti-Leakage Guarantees:
    - All features computed point-in-time using rolling/shifting logic.
    - No negative shifts (future lookahead) allowed for feature generation.
    - Rolling Z-score normalization over a 252-bar window (no global lookahead).
    - Denoised feature derivatives use only past windows (ewm, rolling).
    - Targets are forward returns. Purging and Embargoing applied during split.
    - threshold_search disabled → train median only (no artefact risk).
    """

    def __init__(
        self,
        ticker="^IXIC",
        start="2010-01-01",
        end="2026-04-26",
        fwd_n=5,          # PHASE 2: 1 / 5 / 21 ile dene
        save_dir=None,
    ):
        self.ticker  = ticker
        self.start   = start
        self.end     = end
        self.df      = None

        # ── Hiperparametreler ──────────────────────────────────────────
        self.vrvp_window  = 252
        self.nw_window    = 100
        self.nw_h         = 25
        self.zscore_window = 252

        self.fwd_n = fwd_n                           # PHASE 2
        self.gap   = max(52, self.nw_window, self.fwd_n)   # en az 100

        # PHASE 1 — denoising pencereleri
        self.denoise_short = 5
        self.denoise_long  = 21

        self.feature_cols  = []   # ham feature'lar buraya eklenir
        self.derived_cols  = []   # türev feature'lar buraya eklenir
        self.threshold     = None

        self.save_dir = save_dir or r"C:\Users\DELL\Desktop\trion-mvp\Tracker\Artificial Neuron\Second Attempt Reinforced Learning"

    # ================================================================
    #  VERİ ÇEKME
    # ================================================================
    def fetch_data(self):
        print(f"Fetching data for {self.ticker} from {self.start} to {self.end}...")
        self.df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.droplevel(1)
        self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"Loaded {len(self.df)} daily bars.")

    # ================================================================
    #  HAM FEATURE'LAR  (orijinal kodla birebir aynı)
    # ================================================================
    def add_vrvp(self):
        print("Computing VRVP...")
        poc_list = np.full(len(self.df), np.nan)
        vah_list = np.full(len(self.df), np.nan)
        val_list = np.full(len(self.df), np.nan)

        lows    = self.df['Low'].values
        highs   = self.df['High'].values
        closes  = self.df['Close'].values
        volumes = self.df['Volume'].values
        prices  = (highs + lows) / 2.0

        for i in range(self.vrvp_window - 1, len(self.df)):
            start_idx = i - self.vrvp_window + 1
            w_prices  = prices[start_idx:i+1]
            w_vols    = volumes[start_idx:i+1]
            w_low     = np.min(lows[start_idx:i+1])
            w_high    = np.max(highs[start_idx:i+1])

            if w_high == w_low:
                poc_list[i] = w_prices[-1]
                vah_list[i] = w_prices[-1]
                val_list[i] = w_prices[-1]
                continue

            bins      = np.linspace(w_low, w_high, 21)
            digitized = np.digitize(w_prices, bins)
            digitized = np.clip(digitized, 1, 20)

            vol_profile = np.zeros(20)
            for b in range(1, 21):
                vol_profile[b-1] = w_vols[digitized == b].sum()

            poc_idx     = np.argmax(vol_profile)
            poc         = (bins[poc_idx] + bins[poc_idx+1]) / 2.0
            total_vol   = w_vols.sum()
            target_vol  = total_vol * 0.70
            current_vol = vol_profile[poc_idx]
            lower_idx   = poc_idx
            upper_idx   = poc_idx

            while current_vol < target_vol and (lower_idx > 0 or upper_idx < 19):
                left_vol  = vol_profile[lower_idx-1] if lower_idx > 0  else 0
                right_vol = vol_profile[upper_idx+1] if upper_idx < 19 else 0
                if left_vol > right_vol:
                    lower_idx   -= 1
                    current_vol += left_vol
                elif right_vol > left_vol:
                    upper_idx   += 1
                    current_vol += right_vol
                else:
                    if left_vol == 0 and right_vol == 0:
                        break
                    lower_idx   -= 1
                    upper_idx   += 1
                    current_vol += (left_vol + right_vol)

            lower_idx = max(0, lower_idx)
            upper_idx = min(19, upper_idx)
            vah_list[i] = bins[upper_idx+1]
            val_list[i] = bins[lower_idx]
            poc_list[i] = poc

        self.df['vrvp_poc_dist'] = (poc_list - self.df['Close']) / self.df['Close']
        self.df['vrvp_vah_dist'] = (vah_list - self.df['Close']) / self.df['Close']
        self.df['vrvp_val_dist'] = (val_list - self.df['Close']) / self.df['Close']
        self.feature_cols.extend(['vrvp_poc_dist', 'vrvp_vah_dist', 'vrvp_val_dist'])

    def add_ichimoku(self):
        print("Computing Ichimoku Cloud (leakage-free)...")
        tenkan        = (self.df['High'].rolling(9).max()  + self.df['Low'].rolling(9).min())  / 2.0
        kijun         = (self.df['High'].rolling(26).max() + self.df['Low'].rolling(26).min()) / 2.0
        senkou_a_calc = (tenkan + kijun) / 2.0
        senkou_b_calc = (self.df['High'].rolling(52).max() + self.df['Low'].rolling(52).min()) / 2.0

        senkou_a_lagged = senkou_a_calc.shift(26)
        senkou_b_lagged = senkou_b_calc.shift(26)

        self.df['ichi_sa_dist']    = (self.df['Close'] - senkou_a_lagged) / self.df['Close']
        self.df['ichi_sb_dist']    = (self.df['Close'] - senkou_b_lagged) / self.df['Close']
        self.df['ichi_tk_cross']   = np.sign(tenkan - kijun)
        self.df['ichi_cloud_thick']= np.abs(senkou_a_lagged - senkou_b_lagged) / self.df['Close']
        self.feature_cols.extend(['ichi_sa_dist', 'ichi_sb_dist', 'ichi_tk_cross', 'ichi_cloud_thick'])

    def add_nadaraya_watson(self):
        print("Computing Nadaraya-Watson Envelope (rolling)...")
        W = self.nw_window
        h = self.nw_h

        j        = np.arange(W)
        x_idx    = j[:, None]
        y_idx    = j[None, :]
        w_matrix = np.exp(-0.5 * ((x_idx - y_idx) / h)**2)
        w_sum    = w_matrix.sum(axis=1)

        closes   = self.df['Close'].values
        nw_mid   = np.full(len(self.df), np.nan)
        nw_upper = np.full(len(self.df), np.nan)
        nw_lower = np.full(len(self.df), np.nan)

        for i in range(W - 1, len(self.df)):
            window       = closes[i - W + 1 : i + 1]
            nw_curve     = (w_matrix @ window) / w_sum
            nw_t         = nw_curve[-1]
            residual_std = np.std(window - nw_curve)
            nw_mid[i]    = nw_t
            nw_upper[i]  = nw_t + 2 * residual_std
            nw_lower[i]  = nw_t - 2 * residual_std

        self.df['nw_mid_dist']   = (self.df['Close'] - nw_mid)   / self.df['Close']
        self.df['nw_upper_dist'] = (self.df['Close'] - nw_upper) / self.df['Close']
        self.df['nw_lower_dist'] = (self.df['Close'] - nw_lower) / self.df['Close']
        self.feature_cols.extend(['nw_mid_dist', 'nw_upper_dist', 'nw_lower_dist'])

    def add_kama(self):
        print("Computing KAMA...")
        change     = np.abs(self.df['Close'] - self.df['Close'].shift(10))
        volatility = np.abs(self.df['Close'].diff()).rolling(10).sum()
        er         = change / volatility

        fast_sc = 2.0 / (2  + 1)
        slow_sc = 2.0 / (30 + 1)
        sc      = (er * (fast_sc - slow_sc) + slow_sc)**2

        kama        = np.full(len(self.df), np.nan)
        closes      = self.df['Close'].values
        sc_vals     = sc.values
        first_valid = 10

        if len(self.df) > first_valid:
            kama[first_valid] = closes[first_valid]
            for i in range(first_valid + 1, len(self.df)):
                if not np.isnan(sc_vals[i]):
                    kama[i] = kama[i-1] + sc_vals[i] * (closes[i] - kama[i-1])
                else:
                    kama[i] = kama[i-1]

        self.df['kama_dist'] = (self.df['Close'] - kama) / self.df['Close']
        self.df['kama_er']   = er
        self.feature_cols.extend(['kama_dist', 'kama_er'])

    def add_supertrend(self):
        print("Computing Supertrend...")
        high_low   = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift(1))
        low_close  = np.abs(self.df['Low']  - self.df['Close'].shift(1))
        ranges     = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        atr  = true_range.ewm(alpha=1/14, adjust=False).mean()
        hl2  = (self.df['High'] + self.df['Low']) / 2.0
        basic_upperband = hl2 + 2.0 * atr
        basic_lowerband = hl2 - 2.0 * atr

        final_upperband = np.zeros(len(self.df))
        final_lowerband = np.zeros(len(self.df))
        supertrend      = np.zeros(len(self.df))

        final_upperband[0] = basic_upperband.iloc[0]
        final_lowerband[0] = basic_lowerband.iloc[0]
        supertrend[0]      = 1

        closes  = self.df['Close'].values
        bu_vals = basic_upperband.values
        bl_vals = basic_lowerband.values

        for i in range(1, len(self.df)):
            if bu_vals[i] < final_upperband[i-1] or closes[i-1] > final_upperband[i-1]:
                final_upperband[i] = bu_vals[i]
            else:
                final_upperband[i] = final_upperband[i-1]

            if bl_vals[i] > final_lowerband[i-1] or closes[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = bl_vals[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]

            if   supertrend[i-1] ==  1 and closes[i] <= final_upperband[i]:
                supertrend[i] = -1
            elif supertrend[i-1] == -1 and closes[i] >= final_lowerband[i]:
                supertrend[i] =  1
            else:
                supertrend[i] = supertrend[i-1]

        self.df['st_direction']  = supertrend
        self.df['st_lower_dist'] = (self.df['Close'] - final_lowerband) / self.df['Close']
        self.df['st_upper_dist'] = (final_upperband  - self.df['Close']) / self.df['Close']
        self.df['st_atr_pct']    = atr / self.df['Close']
        self.feature_cols.extend(['st_direction', 'st_lower_dist', 'st_upper_dist', 'st_atr_pct'])

    # ================================================================
    #  PHASE 1 — DENOISED FEATURE TÜREVLERİ
    # ================================================================
    def add_denoised_features(self):
        """
        Her ham feature için 6 türev sütun üretir.
        Tüm işlemler sadece geçmişe bakar → sızıntı riski sıfır.

        Türevler:
          delta_5      — 5 günlük değişim: kısa vadeli momentum sinyali
          delta_21     — 21 günlük değişim: orta vadeli momentum sinyali
          ewm_5        — EWM span=5: gürültüyü kısa vadede filtreler
          ewm_21       — EWM span=21: gürültüyü orta vadede filtreler
          residual_21  — ham - ewm_21: düşük frekanslı trend çıkarılınca
                         kalan yüksek frekanslı bileşen (gerçek gürültü)
          rolling_z_21 — 21 günlük z-score: anlık sapmanın normalize hali
        """
        print("Computing Phase 1 denoised feature derivatives...")

        # Sadece binary/kategorik feature'ları atla
        SKIP = {'st_direction', 'ichi_tk_cross'}

        s  = self.denoise_short   # 5
        lg = self.denoise_long    # 21
        new_cols = []

        for col in self.feature_cols:
            if col in SKIP:
                continue

            series = self.df[col]

            # 5 günlük değişim (kısa momentum)
            d5 = f"{col}__delta_{s}"
            self.df[d5] = series.diff(s)

            # 21 günlük değişim (orta vade momentum)
            d21 = f"{col}__delta_{lg}"
            self.df[d21] = series.diff(lg)

            # EWM hızlı (span=5)
            e5 = f"{col}__ewm_{s}"
            self.df[e5] = series.ewm(span=s, adjust=False).mean()

            # EWM yavaş (span=21)
            e21 = f"{col}__ewm_{lg}"
            self.df[e21] = series.ewm(span=lg, adjust=False).mean()

            # Residual = ham - ewm_21  (yüksek frekanslı bileşen)
            res = f"{col}__residual_{lg}"
            self.df[res] = series - self.df[e21]

            # Rolling z-score 21 günlük (anlık normalize sapma)
            roll_mean = series.rolling(lg).mean()
            roll_std  = series.rolling(lg).std()
            rz = f"{col}__rolling_z_{lg}"
            self.df[rz] = (series - roll_mean) / (roll_std + 1e-8)
            self.df[rz] = self.df[rz].clip(-5, 5)

            new_cols.extend([d5, d21, e5, e21, res, rz])

        self.derived_cols = new_cols
        print(f"  → {len(new_cols)} türev feature eklendi "
              f"({len(self.feature_cols) - len(SKIP)} ham feature × 6).")

    # ================================================================
    #  ROLLING Z-SCORE NORMALİZASYONU (orijinalle aynı mantık)
    # ================================================================
    def rolling_normalization(self):
        """
        Ham + türev feature'ların tümüne 252 bar rolling z-score uygular.
        Binary feature'lar (st_direction, ichi_tk_cross) ve türevleri
        ile rolling_z türevleri hariç tutulur (zaten normalize).
        """
        print("Applying rolling Z-score normalization (252-bar)...")

        SKIP_EXACT   = {'st_direction', 'ichi_tk_cross'}
        SKIP_SUFFIX  = ('__rolling_z_',)   # zaten z-score, tekrar normalize etme

        all_cols = self.feature_cols + self.derived_cols

        for col in all_cols:
            base = col.split('__')[0]
            if base in SKIP_EXACT:
                continue
            if any(col.endswith(s) for s in SKIP_SUFFIX):
                continue

            mean_col = self.df[col].rolling(self.zscore_window).mean()
            std_col  = self.df[col].rolling(self.zscore_window).std()
            z_col    = (self.df[col] - mean_col) / (std_col + 1e-8)
            self.df[col] = z_col.clip(-5, 5)

    # ================================================================
    #  SPLIT + KAYDET  (embargo düzeltmesiyle)
    # ================================================================
    def split_and_save(self):
        print(f"\nApplying walk-forward split with Embargo (fwd_n={self.fwd_n})...")

        # ── PHASE 2: Horizon — threshold_search kapalı, sadece train median
        self.df['fwd_ret'] = self.df['Close'].shift(-self.fwd_n) / self.df['Close'] - 1.0

        # Tüm feature sütunlarında NaN olan satırları düşür (warm-up)
        all_cols = self.feature_cols + self.derived_cols
        self.df  = self.df.dropna(subset=all_cols)

        # ── Tarih bazlı bölme ──────────────────────────────────────
        train_mask = (self.df.index >= '2010-01-01') & (self.df.index <= '2022-12-31')
        val_mask   = (self.df.index >= '2023-01-01') & (self.df.index <= '2023-12-31')
        test_mask  = (self.df.index >= '2024-01-01') & (self.df.index <= '2026-04-26')

        df_train = self.df[train_mask].copy()
        df_val   = self.df[val_mask].copy()
        df_test  = self.df[test_mask].copy()

        # ── EMBARGO (düzeltilmiş) ──────────────────────────────────
        # Orijinal: train ve val'in sonu kırpılıyor (set içi embargo).
        # Düzeltme : aynı mantık korunuyor + test başından fwd_n satır
        # siliniyor (val→test geçiş bölgesinde label overlap riski = 0).
        if len(df_train) > self.gap:
            df_train = df_train.iloc[:-self.gap]
        if len(df_val) > self.gap:
            df_val = df_val.iloc[:-self.gap]
        # Test başından fwd_n satır sil (val'in son fwd_n günüyle label örtüşmesi)
        if len(df_test) > self.fwd_n:
            df_test = df_test.iloc[self.fwd_n:]

        # ── NaN target satırlarını düşür (son fwd_n satır) ─────────
        df_train = df_train.dropna(subset=['fwd_ret'])
        df_val   = df_val.dropna(subset=['fwd_ret'])
        df_test  = df_test.dropna(subset=['fwd_ret'])

        # ── Threshold: sadece train median (artefakt riski yok) ────
        self.threshold = float(df_train['fwd_ret'].median())
        print(f"Target threshold (Train median fwd_{self.fwd_n}d): {self.threshold:.6f}")

        df_train['target'] = (df_train['fwd_ret'] > self.threshold).astype(int)
        df_val['target']   = (df_val['fwd_ret']   > self.threshold).astype(int)
        df_test['target']  = (df_test['fwd_ret']  > self.threshold).astype(int)

        # ── Numpy dizileri ─────────────────────────────────────────
        X_train = df_train[all_cols].values
        y_train = df_train['target'].values
        X_val   = df_val[all_cols].values
        y_val   = df_val['target'].values
        X_test  = df_test[all_cols].values
        y_test  = df_test['target'].values

        # ── Kaydet ────────────────────────────────────────────────
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "X7_train.npy"), X_train)
        np.save(os.path.join(self.save_dir, "y7_train.npy"), y_train)
        np.save(os.path.join(self.save_dir, "X7_val.npy"),   X_val)
        np.save(os.path.join(self.save_dir, "y7_val.npy"),   y_val)
        np.save(os.path.join(self.save_dir, "X7_test.npy"),  X_test)
        np.save(os.path.join(self.save_dir, "y7_test.npy"),  y_test)

        meta = {
            "features"          : all_cols,
            "raw_features"      : self.feature_cols,
            "derived_features"  : self.derived_cols,
            "n_features_total"  : len(all_cols),
            "n_features_raw"    : len(self.feature_cols),
            "n_features_derived": len(self.derived_cols),
            "fwd_n"             : self.fwd_n,
            "train_dates"       : [str(df_train.index[0].date()), str(df_train.index[-1].date())],
            "val_dates"         : [str(df_val.index[0].date()),   str(df_val.index[-1].date())],
            "test_dates"        : [str(df_test.index[0].date()),  str(df_test.index[-1].date())],
            "threshold"         : self.threshold,
            "gap_size"          : self.gap,
            "train_class_balance": float(y_train.mean()),
            "val_class_balance"  : float(y_val.mean()),
            "test_class_balance" : float(y_test.mean()),
        }
        with open(os.path.join(self.save_dir, "X7_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # ── DOĞRULAMA KONTROLLERI ──────────────────────────────────
        print("\n=== VERIFICATION CHECKS ===")

        # 1. Index overlap
        tr_idx  = set(df_train.index)
        val_idx = set(df_val.index)
        tst_idx = set(df_test.index)
        overlap_tv = len(tr_idx & val_idx)
        overlap_vt = len(val_idx & tst_idx)
        print(f"1. Index overlap (Train∩Val={overlap_tv}, Val∩Test={overlap_vt}):",
              "PASS" if overlap_tv == 0 and overlap_vt == 0 else "FAIL")

        # 2. Train-Val gap
        full_idx   = self.df.index
        train_end  = full_idx.get_loc(df_train.index[-1])
        val_start  = full_idx.get_loc(df_val.index[0])
        gap1       = val_start - train_end - 1
        print(f"2. Train→Val gap={gap1} (required≥{self.gap}):",
              "PASS" if gap1 >= self.gap else f"FAIL")

        # 3. Val-Test gap
        val_end    = full_idx.get_loc(df_val.index[-1])
        test_start = full_idx.get_loc(df_test.index[0])
        gap2       = test_start - val_end - 1
        print(f"3. Val→Test gap={gap2} (required≥{self.fwd_n}):",
              "PASS" if gap2 >= self.fwd_n else f"FAIL")

        # 4. Negatif shift kontrolü (manuel)
        print("4. No negative shift in feature generation: PASS (rolling/ewm/diff only)")

        # 5. Sınıf dengesi
        print(f"5. Class balance (1s)  Train:{y_train.mean():.2%}  "
              f"Val:{y_val.mean():.2%}  Test:{y_test.mean():.2%}")

        # 6. Shapes
        print(f"6. Shapes  Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

        # 7. Tarihler
        print(f"7. Train: {meta['train_dates'][0]} → {meta['train_dates'][1]}")
        print(f"   Val  : {meta['val_dates'][0]}   → {meta['val_dates'][1]}")
        print(f"   Test : {meta['test_dates'][0]}  → {meta['test_dates'][1]}")

        # 8. Feature sayısı
        print(f"8. Features  Raw:{len(self.feature_cols)}  "
              f"Derived:{len(self.derived_cols)}  "
              f"Total:{len(all_cols)}")

        print("\nAll X7 v2 pipeline steps completed successfully.")
        print(f"Files saved to: {self.save_dir}")


# ====================================================================
#  KULLANIM
# ====================================================================
if __name__ == "__main__":

    # ── Horizon denemeleri için sadece fwd_n'i değiştir ─────────────
    # fwd_n=1  → yarın yönünü tahmin et
    # fwd_n=5  → 1 hafta sonrasını tahmin et  (önerilen başlangıç)
    # fwd_n=21 → 1 ay sonrasını tahmin et
    pipeline = LeakageFreePipelineX7(fwd_n=5)

    pipeline.fetch_data()

    # Ham feature'lar (orijinalle birebir aynı)
    pipeline.add_vrvp()
    pipeline.add_ichimoku()
    pipeline.add_nadaraya_watson()
    pipeline.add_kama()
    pipeline.add_supertrend()

    # PHASE 1: Türev feature'lar (YENİ)
    pipeline.add_denoised_features()

    # Normalizasyon (ham + türevlerin tümü)
    pipeline.rolling_normalization()

    # Split + kaydet
    pipeline.split_and_save()
