# BIST Stock Market Anomaly Detection

An unsupervised machine learning system that detects suspicious price and volume movements in Turkish stock market (BIST) using Isolation Forest algorithm.

## Live Demo
🔗 [Open Dashboard](https://bist-anomaly-detection.streamlit.app/)

## What it does
- Flags abnormal trading days based on price & volume behavior
- Analyzes 10 major BIST stocks over 3 years of market data
- Interactive dashboard to explore anomalies per stock
- Adjustable anomaly sensitivity slider

## How it works
1. **Data Collection** — Historical BIST price & volume data
2. **Feature Engineering** — 5 financial features calculated per stock:
   - Daily price change (%)
   - Volume ratio (vs. 20-day average)
   - Intraday volatility
   - Price z-score
   - Volume change (%)
3. **Anomaly Detection** — Isolation Forest (unsupervised ML)
4. **Visualization** — Interactive Streamlit dashboard

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Plotly
- Streamlit
- yfinance

## Detected Anomaly Types
- **Pump & Dump** — Abnormal volume spike followed by sharp price drop
- **Wash Trading** — High volume with no significant price movement
- **Spoofing** — Unusual intraday volatility patterns

## Results
- 7,300+ trading day records analyzed
- 146 anomalies detected across 10 stocks
- AKBNK.IS showed highest anomaly frequency (25 days)
- May 2023 election period flagged across multiple stocks simultaneously

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
Gizem Bal — [LinkedIn](https://linkedin.com/in/balgizem) | [GitHub](https://github.com/gizembal)
