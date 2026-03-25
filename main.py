import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

HISSELER = [
    "THYAO.IS", "GARAN.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS",
    "AKBNK.IS", "BIMAS.IS", "FROTO.IS", "PGSUS.IS", "ASELS.IS",
]


def veri_cek(hisseler=HISSELER, yil=3):
    bitis = datetime.today().strftime('%Y-%m-%d')
    baslangic = (datetime.today() - timedelta(days=yil * 365)).strftime('%Y-%m-%d')

    df_ham = yf.download(hisseler, start=baslangic, end=bitis,
                         group_by='ticker', auto_adjust=True)
    satirlar = []
    for hisse in hisseler:
        try:
            df_h = df_ham[hisse].copy()
            df_h['ticker'] = hisse
            df_h['tarih'] = df_h.index
            df_h = df_h.dropna(subset=['Close', 'Volume'])
            satirlar.append(df_h)
        except:
            pass

    df = pd.concat(satirlar, ignore_index=True)
    df = df.rename(columns={
        'Open': 'acilis', 'High': 'yuksek', 'Low': 'dusuk',
        'Close': 'kapanis', 'Volume': 'hacim'
    })
    return df


def ozellik_hesapla(df):
    df = df.sort_values(['ticker', 'tarih']).reset_index(drop=True)
    for hisse in df['ticker'].unique():
        mask = df['ticker'] == hisse
        df.loc[mask, 'fiyat_degisim'] = df.loc[mask, 'kapanis'].pct_change() * 100
        df.loc[mask, 'hacim_ort_20g'] = df.loc[mask, 'hacim'].rolling(20).mean()
        df.loc[mask, 'hacim_oran'] = df.loc[mask, 'hacim'] / df.loc[mask, 'hacim_ort_20g']
        df.loc[mask, 'volatilite'] = (df.loc[mask, 'yuksek'] - df.loc[mask, 'dusuk']) / df.loc[mask, 'kapanis'] * 100
        df.loc[mask, 'fiyat_zskor'] = (
                                              df.loc[mask, 'fiyat_degisim'] - df.loc[mask, 'fiyat_degisim'].rolling(
                                          20).mean()
                                      ) / df.loc[mask, 'fiyat_degisim'].rolling(20).std()
        df.loc[mask, 'hacim_degisim'] = df.loc[mask, 'hacim'].pct_change() * 100
    return df.dropna()


def model_calistir(df, contamination=0.02):
    ozellikler = ['fiyat_degisim','hacim_oran','volatilite','fiyat_zskor','hacim_degisim']
    
    df_copy = df.copy()
    df_copy[ozellikler] = df_copy[ozellikler].replace([np.inf, -np.inf], np.nan)
    df_copy = df_copy.dropna(subset=ozellikler)
    
    X = df_copy[ozellikler].values  # numpy array'e çevir
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    df_copy['anomali_skor'] = model.fit_predict(X_scaled)
    df_copy['anomali_skor_ham'] = model.score_samples(X_scaled)
    
    return df_copy
