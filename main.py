import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import time

HISSELER = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS",
    "AKBNK.IS", "BIMAS.IS", "FROTO.IS", "PGSUS.IS", "ASELS.IS",
]

def veri_cek(hisseler=HISSELER, yil=3):
    bitis = datetime.today().strftime('%Y-%m-%d')
    baslangic = (datetime.today() - timedelta(days=yil*365)).strftime('%Y-%m-%d')

    satirlar = []
    for hisse in hisseler:
        try:
            time.sleep(1)  # rate limit koruması
            df_h = yf.download(hisse, start=baslangic, end=bitis,
                               auto_adjust=True, progress=False)
            if df_h.empty:
                continue
            df_h.columns = [c[0] if isinstance(c, tuple) else c for c in df_h.columns]
            df_h['ticker'] = hisse
            df_h['tarih'] = df_h.index
            df_h = df_h.dropna(subset=['Close', 'Volume'])
            df_h = df_h.rename(columns={
                'Open':'acilis','High':'yuksek','Low':'dusuk',
                'Close':'kapanis','Volume':'hacim'
            })
            satirlar.append(df_h)
        except Exception as e:
            print(f"{hisse} atlandı: {e}")
            continue

    if not satirlar:
        raise ValueError("Hiç veri çekilemedi. Lütfen daha sonra tekrar deneyin.")

    df = pd.concat(satirlar, ignore_index=True)
    return df

def ozellik_hesapla(df):
    df = df.sort_values(['ticker','tarih']).reset_index(drop=True)
    sonuc = []
    for hisse in df['ticker'].unique():
        d = df[df['ticker'] == hisse].copy()
        d['fiyat_degisim'] = d['kapanis'].pct_change() * 100
        d['hacim_ort_20g'] = d['hacim'].rolling(20).mean()
        d['hacim_oran'] = d['hacim'] / d['hacim_ort_20g']
        d['volatilite'] = (d['yuksek'] - d['dusuk']) / d['kapanis'] * 100
        d['fiyat_zskor'] = (
            d['fiyat_degisim'] - d['fiyat_degisim'].rolling(20).mean()
        ) / d['fiyat_degisim'].rolling(20).std()
        d['hacim_degisim'] = d['hacim'].pct_change() * 100
        sonuc.append(d)
    df = pd.concat(sonuc, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def model_calistir(df, contamination=0.02):
    ozellikler = ['fiyat_degisim','hacim_oran','volatilite','fiyat_zskor','hacim_degisim']
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ozellikler)

    if len(df) == 0:
        raise ValueError("Temizleme sonrası veri kalmadı.")

    X = df[ozellikler].to_numpy().astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    df['anomali_skor'] = model.fit_predict(X_scaled)
    df['anomali_skor_ham'] = model.score_samples(X_scaled)

    return df
