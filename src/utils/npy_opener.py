import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_fixed_2018_2019_all_visible():
    file_path = r"C:\Users\DELL\Desktop\trion-mvp\Tracker\Artificial Neuron\market_data_10y_zscore.csv"
    print("Veri yükleniyor ve ölçek sorunu düzeltiliyor...")
    
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df = df.loc['2018-01-01':'2019-12-31']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # KRİTİK DÜZELTME: Sadece gerçek Z-Skoru (-3, +3 arası) olan kolonları manuel seçiyoruz.
    # Milyonluk ham verilerin (OI_Proxy_SPY_Volume vb.) grafiği ezmesini engelliyoruz.
    macro_features = [
        'VIX_Term_Structure', 'Yield_Curve', 'SKEW_Index', 
        'Risk_Appetite_Ratio', 'Crude_Oil', 'VWAP_Deviation', 
        'Volume_Momentum'
    ]
    
    # Tüm makro çizgiler varsayılan olarak AÇIK
    for col in macro_features:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col, mode='lines', 
                           line=dict(width=1.5), opacity=0.8),
                secondary_y=False, # Sol eksen (-3, +3)
            )

    # Nasdaq Fiyatı (Sağ Eksen)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Nasdaq_Close'], name='Nasdaq_Close (Ham Fiyat)',
                   mode='lines', line=dict(color='black', width=3.5)),
        secondary_y=True,
    )

    # Önemli Olaylar
    important_events = {
        '2018-02-05': 'Volmageddon (VIX Patlaması)',
        '2018-12-24': '2018 Noel Çöküşü (Fed)'
    }

    for date_str, label in important_events.items():
        event_date = pd.to_datetime(date_str)
        if event_date >= df.index.min() and event_date <= df.index.max():
            fig.add_vline(x=event_date, line_width=2, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=event_date, y=df['Nasdaq_Close'].max(), yref="y2", 
                text=label, showarrow=False, textangle=-90, yanchor='top',
                font=dict(color="red", size=12, family="Arial Black")
            )

    fig.update_layout(
        title='2018-2019: Düzeltilmiş Quant Göstergeleri (Tümü Açık)',
        xaxis_title='Tarih',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    # Eksen isimlerini güncelle (Ölçeğin doğru olduğundan emin olmak için)
    fig.update_yaxes(title_text="Gerçek Z-Skorları (-3 ile +3 Arası)", secondary_y=False)
    fig.update_yaxes(title_text="Nasdaq Fiyatı (USD)", secondary_y=True)

    fig.show()

if __name__ == "__main__":
    plot_fixed_2018_2019_all_visible()