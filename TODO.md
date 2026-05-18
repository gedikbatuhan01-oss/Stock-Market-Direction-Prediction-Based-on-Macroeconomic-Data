# TODO - Dataset Olusturma Sureci

## Mevcut Kodda Yapilanlar

- [x] Hazir enriched CSV dosyasinin kullanilmasi: `data/raw/market_data_10y_enriched.csv`
- [x] Config uzerinden veri dosyasi yolunun tanimlanmasi: `configs/base.yaml`
- [x] CSV dosyasinin okunmasi
- [x] `Date` kolonunun tarih olarak parse edilmesi
- [x] Verinin kronolojik olarak siralanmasi
- [x] Kronolojik siralamanin kontrol edilmesi
- [x] Gerekli kolonlarin secilmesi
- [x] Kullanilacak feature kolonlarinin config icinde tanimlanmasi
- [x] Hedef fiyat kolonunun `Nasdaq_Close` olarak tanimlanmasi
- [x] Eksik degerler icin forward-fill uygulanmasi
- [x] Forward-fill sonrasi kalan NaN satirlarinin silinmesi
- [x] Final test setinin kronolojik olarak en sondaki %15 bolumden ayrilmasi
- [x] Development set uzerinde expanding-window cross-validation fold'larinin olusturulmasi
- [x] `Nasdaq_Close` uzerinden t+1 forward return hesaplanmasi
- [x] Label threshold degerinin yalnizca train fold uzerinden hesaplanmasi
- [x] Bull / Bear / Neutral label uretimi
- [x] Neutral orneklerin egitim dataset'inden cikarilmasi
- [x] Lookback pencereleriyle sequence dataset olusturulmasi
- [x] Validation/test sequence'lerinin gecmis feature satirlarini kullanmasina izin verilip label endpoint'inin split icinde tutulmasi
- [x] Sequence input'unun neural modeller icin 3D tensor formatinda tutulmasi
- [x] Tabular modeller icin sequence input'unun flatten edilmesi
- [x] Scaler'in yalnizca train set uzerinde fit edilmesi
- [x] Validation/test setlerinin train-fit scaler ile transform edilmesi
- [x] `StandardScaler` ve `RobustScaler` seceneklerinin desteklenmesi
- [x] Label quantile adaylarinin config uzerinden denenmesi
- [x] Lookback adaylarinin config uzerinden denenmesi
- [x] Dataset ve label dagilimi diagnostiklerinin rapora eklenmesi

## Kodda Olmayan veya Eksik Kalanlar

- [ ] Veri kaynaklarindan otomatik veri cekme scripti yazilmasi
- [ ] Nasdaq fiyat verisinin otomatik indirilmesi
- [ ] VIX term structure verisinin otomatik indirilmesi veya uretilmesi
- [ ] Yield curve verisinin otomatik indirilmesi veya uretilmesi
- [ ] SKEW index verisinin otomatik indirilmesi
- [ ] Crude oil verisinin otomatik indirilmesi
- [ ] Risk appetite ratio feature'inin ham kaynaklardan yeniden hesaplanmasi
- [ ] VWAP deviation feature'inin ham OHLCV verisinden yeniden hesaplanmasi
- [ ] Volume momentum feature'inin ham volume verisinden yeniden hesaplanmasi
- [ ] Enriched dataset'i sifirdan ureten `make_dataset` benzeri tek komutluk pipeline eklenmesi
- [ ] Ham veri, ara veri ve final dataset asamalarinin ayri klasorlerde yonetilmesi
- [ ] Dataset versiyonlama sistemi eklenmesi
- [ ] Dataset uretim tarihinin, kaynaklarinin ve parametrelerinin metadata olarak saklanmasi
- [ ] Dataset hash veya checksum bilgisinin kaydedilmesi
- [ ] Duplicate date kontrolu eklenmesi
- [ ] Tarih boslugu / eksik islem gunu kontrolu eklenmesi
- [ ] Feature dtype dogrulamasi eklenmesi
- [ ] Sonsuz deger (`inf`, `-inf`) kontrolu eklenmesi
- [ ] Outlier kontrolu veya winsorization/clipping politikasi eklenmesi
- [ ] Ham feature uretim formullerinin kod seviyesinde dokumante edilmesi
- [ ] `data/processed/X_tensor.npy` ve `data/processed/y_tensor.npy` dosyalarinin ana pipeline tarafindan yeniden uretilebilir hale getirilmesi
- [ ] Dataset olusturma sonrasi otomatik kalite raporu uretilmesi
- [ ] Dataset olusturma adimlari icin ayri test dosyalari eklenmesi
