# BIST Stock Market Anomaly Detection

## What is this project?

Every day, millions of trades happen in the Turkish stock market (BIST). 
Most days are "normal" — prices move slightly, trading volume follows 
its usual pattern.

But some days are different. A stock suddenly drops 9% with 3x its 
normal trading volume. Is this a natural market reaction, or is someone 
manipulating the price?

This project uses machine learning to automatically flag suspicious 
trading days across 10 major BIST stocks — so instead of manually 
reviewing 7,300 days of data, you focus only on the ones that look unusual.

## Live Demo
🔗 [Open Dashboard](https://bist-anomaly-detection.streamlit.app/)

---

## How does it work?

### Step 1 — Data Collection
We pulled 3 years of daily stock data (~750 trading days) for 10 major 
BIST stocks using Yahoo Finance. For each day, we have:
- Opening and closing price
- Highest and lowest price of the day
- Trading volume (number of shares that changed hands)

### Step 2 — Feature Engineering
Raw price and volume data alone doesn't tell us much. So we calculated 
5 indicators for each trading day — think of these as "red flag signals":

| Feature | What it measures | Why it matters |
|---|---|---|
| **Price change (%)** | How much did the price move vs. yesterday? | Normal stocks move ±2-3%. A ±9% move is rare. |
| **Volume ratio** | Today's volume vs. 20-day average | 3x normal volume on a big price day = suspicious |
| **Intraday volatility** | Difference between day's high and low | Wild swings within a day can signal panic or manipulation |
| **Price z-score** | How unusual is today's move for *this specific stock*? | Normalizes each stock against its own history — a 5% drop means different things for different stocks |
| **Volume momentum** | How much did volume change vs. yesterday? | A sudden volume spike (not just high volume) is a strong signal |

### Step 3 — Anomaly Detection with Isolation Forest

**What is Isolation Forest?**
It's a machine learning algorithm that finds "outliers" — data points 
that are very different from everything else.

Think of it like this: imagine plotting all 7,300 trading days on a 
map based on their 5 features. Normal days cluster together in the 
middle. Anomalies are isolated, far from the crowd.

The algorithm works by drawing random lines to separate data points. 
Normal days are hard to isolate (surrounded by similar days — needs 
many lines). Anomalies are easy to isolate (alone, far from others — 
needs very few lines).

**What is "unsupervised" ML?**
We didn't need to tell the model "this day was manipulation" or 
"this day was normal." It figured out what's unusual on its own, 
just by looking at the patterns in the data.

**What is contamination = 0.02?**
This tells the model: "assume about 2% of all trading days are 
anomalies." With 7,300 records → 146 flagged days. You can adjust 
this with the slider in the dashboard.

### Step 4 — Interactive Dashboard
Built with Streamlit — select any stock, adjust anomaly sensitivity, 
and explore flagged days with an interactive chart.

---

## What did we find?

- **7,300+** trading day records analyzed
- **146** anomalies detected across 10 stocks
- **AKBNK.IS** had the most anomalies — 25 flagged days out of ~750
- **ASELS.IS** was the most stable — only 8 flagged days
- **May 2023** — multiple stocks flagged simultaneously during 
Turkey's election period (systemic market stress, not manipulation)

---

## Important caveat

Anomaly ≠ Manipulation. A flagged day means the trading behavior 
was statistically unusual — it could be due to major news, 
macroeconomic events, or actual manipulation. Human judgment is 
still needed to interpret the results.

---

## Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn
- **Plotly** — Interactive charts
- **Streamlit** — Web dashboard
- **yfinance** — Stock data

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's next?
- Add SPK enforcement decisions as labels → supervised learning
- Expand to all BIST 100 stocks
- Real-time data pipeline
- News sentiment analysis

## Author
Gizem Bal — [LinkedIn](https://linkedin.com/in/balgizem) | 
[GitHub](https://github.com/gizembal)
