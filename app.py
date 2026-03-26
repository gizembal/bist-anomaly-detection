import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd 
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

with st.expander("📖 Metodoloji — Özellikler nasıl hesaplanıyor?"):
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Fiyat Değişimi (%)")
        st.caption("Günlük fiyat hareketini ölçer")
        st.code("(bugünkü kapanış - dünkü kapanış) / dünkü kapanış × 100")
        st.info("Normal: ±2-3% | Şüpheli: ±8-10%")
        
        st.subheader("2. Hacim Oranı")
        st.caption("Bugünkü hacmi 20 günlük ortalama ile karşılaştırır")
        st.code("bugünkü hacim / son 20 günlük ortalama hacim")
        st.info("3x üzeri = normalin 3 katı işlem")
        
        st.subheader("3. Gün İçi Volatilite (%)")
        st.caption("Günün en yüksek ve en düşük fiyatı arasındaki fark")
        st.code("(günün en yükseği - günün en düşüğü) / kapanış × 100")
        st.info("Kapanış sakin görünse bile gün içi oynaklığı yakalar")

    with col_b:
        st.subheader("4. Fiyat Z-Skoru")
        st.caption("Bu hareket bu hisse için ne kadar nadir?")
        st.code("(bugünkü değişim - 20g ortalama) / 20g standart sapma")
        st.info("±2: nadir | ±3: çok nadir | ±4: alarm 🚨")
        
        st.subheader("5. Hacim Değişimi (%)")
        st.caption("Hacmin dünden bugüne değişimi")
        st.code("(bugünkü hacim - dünkü hacim) / dünkü hacim × 100")
        st.info("Anlık hacim sıçramasını yakalar")
        
        st.subheader("Isolation Forest")
        st.caption("5 özellik birlikte değerlendiriliyor")
        st.info("Diğer günlerden kolayca izole edilen günler → anomali\nContamination = %2 → 7.300 günden 146 anomali")

    st.warning("⚠️ Anomali = Manipülasyon değildir. Flaglenen günler istatistiksel olarak sıradışıdır — büyük haberler, makroekonomik olaylar veya gerçek manipülasyon nedeniyle olabilir.")

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
tablo = anomali_h[['tarih','kapanis','fiyat_degisim','hacim_oran',
                    'volatilite','fiyat_zskor','hacim_degisim',
                    'anomali_skor_ham']].copy()
tablo.columns = ['Tarih','Kapanış','Fiyat Değişimi %','Hacim Oranı',
                 'Volatilite %','Fiyat Z-Skoru','Hacim Değişimi %',
                 'Anomali Skoru']
tablo = tablo.sort_values('Anomali Skoru').reset_index(drop=True)
tablo['Fiyat Değişimi %'] = tablo['Fiyat Değişimi %'].round(2)
tablo['Hacim Oranı'] = tablo['Hacim Oranı'].round(2)
tablo['Volatilite %'] = tablo['Volatilite %'].round(2)
tablo['Fiyat Z-Skoru'] = tablo['Fiyat Z-Skoru'].round(2)
tablo['Hacim Değişimi %'] = tablo['Hacim Değişimi %'].round(2)
tablo['Anomali Skoru'] = tablo['Anomali Skoru'].round(4)
st.dataframe(tablo, use_container_width=True)

st.subheader("All Stocks — Anomaly Summary")
ozet = anomaliler.groupby('ticker').size().reset_index(name='Anomaly Count')
ozet = ozet.sort_values('Anomaly Count', ascending=False)

fig2 = go.Figure(go.Bar(
    x=ozet['ticker'],
    y=ozet['Anomaly Count'],
    marker_color='#E24B4A'
))
fig2.update_layout(
    height=300,
    plot_bgcolor='#1a1a2e',
    paper_bgcolor='#1a1a2e',
    font=dict(color='white'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
)
st.plotly_chart(fig2, use_container_width=True)
