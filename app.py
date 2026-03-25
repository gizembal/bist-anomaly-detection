import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd 
import openpyxl
from main import veri_cek, ozellik_hesapla, model_calistir, HISSELER

st.set_page_config(
    page_title="BIST Anomali Tespit Sistemi",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BIST Borsa Anomali Tespit Sistemi")
st.markdown("Isolation Forest algoritmasıyla anormal fiyat ve hacim hareketlerini tespit eder.")

with st.sidebar:
    st.header("Ayarlar")
    secili_hisse = st.selectbox("Hisse Seç", HISSELER)
    contamination = st.slider("Anomali Oranı", 0.01, 0.10, 0.02, 0.01,
                              help="Verinin yüzde kaçı anomali olarak işaretlensin?")

@st.cache_data(ttl=3600)
def yukle():
    df = pd.read_csv('bist_data.csv')
    df['tarih'] = pd.to_datetime(df['tarih'])
    return df

with st.spinner("Veri çekiliyor, lütfen bekle..."):
    df = yukle()

df_model = model_calistir(df, contamination)
anomaliler = df_model[df_model['anomali_skor'] == -1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Kayıt", f"{len(df_model):,}")
col2.metric("Anomali Sayısı", len(anomaliler))
col3.metric("En Çok Anomali", anomaliler['ticker'].value_counts().index[0])
col4.metric("Anomali Oranı", f"%{len(anomaliler)/len(df_model)*100:.1f}")

st.divider()

df_h = df_model[df_model['ticker'] == secili_hisse]
anomali_h = df_h[df_h['anomali_skor'] == -1]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=(f'{secili_hisse} Kapanış Fiyatı', 'Hacim Oranı'),
    row_heights=[0.6, 0.4], vertical_spacing=0.08
)

fig.add_trace(go.Scatter(
    x=df_h['tarih'], y=df_h['kapanis'],
    name='Kapanış', line=dict(color='#378ADD', width=1.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=anomali_h['tarih'], y=anomali_h['kapanis'],
    mode='markers', name='Anomali',
    marker=dict(color='#E24B4A', size=10, symbol='circle',
                line=dict(color='white', width=1.5)),
    hovertemplate='<b>%{x}</b><br>Fiyat: %{y:.2f}'
), row=1, col=1)

fig.add_trace(go.Bar(
    x=df_h['tarih'], y=df_h['hacim_oran'],
    name='Hacim Oranı',
    marker_color=df_h['anomali_skor'].apply(
        lambda x: '#E24B4A' if x == -1 else '#B5D4F4'
    )
), row=2, col=1)

fig.add_hline(y=3, line_dash="dash", line_color="#E24B4A",
              annotation_text="3x eşik", row=2, col=1)

fig.update_layout(
    height=600,
    plot_bgcolor='#1a1a2e',
    paper_bgcolor='#1a1a2e',
    font=dict(color='white', size=13),
    legend=dict(
        bgcolor='rgba(255,255,255,0.1)',
        bordercolor='rgba(255,255,255,0.2)',
        borderwidth=1
    )
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"{secili_hisse} — Tespit Edilen Anomaliler")
tablo = anomali_h[['tarih','kapanis','fiyat_degisim','hacim_oran','volatilite','anomali_skor_ham']].copy()
tablo.columns = ['Tarih','Kapanış','Fiyat Değişimi %','Hacim Oranı','Volatilite %','Anomali Skoru']
tablo = tablo.sort_values('Anomali Skoru').reset_index(drop=True)
tablo['Fiyat Değişimi %'] = tablo['Fiyat Değişimi %'].round(2)
tablo['Hacim Oranı'] = tablo['Hacim Oranı'].round(2)
tablo['Volatilite %'] = tablo['Volatilite %'].round(2)
tablo['Anomali Skoru'] = tablo['Anomali Skoru'].round(4)
st.dataframe(tablo, use_container_width=True)

st.subheader("Tüm Hisseler — Anomali Özeti")
ozet = anomaliler.groupby('ticker').size().reset_index(name='Anomali Sayısı')
ozet = ozet.sort_values('Anomali Sayısı', ascending=False)
st.bar_chart(ozet.set_index('ticker'))
