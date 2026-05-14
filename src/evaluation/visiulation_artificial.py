import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_macro_and_price(file_path):
    print("Veri yükleniyor...")
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # Görselleştirilecek kolonlar (Nasdaq en üstte)
    columns_to_plot = [
        'Nasdaq_Close',
        'VIX_Term_Structure', 
        'Yield_Curve', 
        'SKEW_Index',
        'Risk_Appetite_Ratio',
        'Crude_Oil',
        'VWAP_Deviation',
        'Volume_Momentum'
    ]
    
    # Eksik kolon kontrolü
    missing_cols = [col for col in columns_to_plot if col not in df.columns]
    if missing_cols:
        print(f"Hata: Şu kolonlar bulunamadı: {missing_cols}")
        return

    # Grafik alanını oluştur (8 satır, 1 sütun)
    fig, axes = plt.subplots(nrows=len(columns_to_plot), ncols=1, figsize=(15, 22), sharex=True)
    
    # Başlık
    fig.suptitle('Quant Makro Göstergeleri ve Nasdaq Fiyat Hareketi (2016-2026)', fontsize=16, fontweight='bold', y=0.92)

    # Renk paleti
    colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Önemli piyasa olayları ve tarihleri
    important_events = {
        '2018-02-05': 'Volmageddon',
        '2020-02-24': 'COVID-19 Çöküşü',
        '2022-03-16': 'Fed Faiz Artırımı Başlangıcı',
        '2023-03-10': 'SVB Banka Krizi'
    }

    # Her bir kolonu kendi alt grafiğine (subplot) çizdir
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        
        # Nasdaq fiyatını daha belirgin (kalın) çizdir
        if col == 'Nasdaq_Close':
            ax.plot(df.index, df[col], color=colors[i], linewidth=2.0)
            ax.set_ylabel("Fiyat", fontsize=9)
        else:
            ax.plot(df.index, df[col], color=colors[i], linewidth=1.2)
        
        # Grafik başlıkları ve eksen ayarları
        ax.set_title(col, loc='left', fontsize=12, fontweight='bold', color='#333333')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Yatay kritik seviye uyarı çizgileri
        if col == 'VIX_Term_Structure':
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        elif col == 'SKEW_Index':
            ax.axhline(140, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        elif col == 'VWAP_Deviation':
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Dikey kırmızı kriz çizgilerini her alt grafiğe ekle
        for date_str, label_text in important_events.items():
            event_date = pd.to_datetime(date_str)
            
            # Eğer tarih veri setimizin içindeyse çizgiyi çek
            if event_date >= df.index.min() and event_date <= df.index.max():
                ax.axvline(event_date, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
                
                # Metin etiketini sadece en üstteki (Nasdaq) grafiğe yaz (kalabalık olmaması için)
                if i == 0:
                    ax.text(event_date, ax.get_ylim()[1] * 1.02, label_text, 
                            rotation=25, color='red', fontsize=10, fontweight='bold', 
                            verticalalignment='bottom')

    # X eksenini (Tarih) düzenle
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # Boşlukları ayarla ve göster
    plt.tight_layout(rect=[0, 0, 1, 0.88]) 
    plt.show()

if __name__ == "__main__":
    # Terminal çıktında görünen, verinin bulunduğu gerçek ve doğru dosya yolu
    dosya_yolu = r"C:\Users\DELL\Desktop\trion-mvp\Tracker\Artificial Neuron\market_data_10y_enriched.csv"
    plot_macro_and_price(dosya_yolu)