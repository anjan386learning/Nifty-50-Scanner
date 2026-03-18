# ⬡ Nifty 50 Signal Scanner — Web App

Live Long/Short signal scanner for all 50 Nifty stocks.
Indicators: RSI(14) · Session VWAP · EMA(33) · SL Hunt Detector

---

## 🚀 Deploy to Streamlit Cloud (FREE — get a public URL in 5 minutes)

### Step 1 — Create a GitHub repository

1. Go to https://github.com/new
2. Name it: `nifty50-scanner`
3. Set it to **Public**
4. Click **Create repository**

### Step 2 — Upload these files to GitHub

Upload the following files maintaining this exact structure:

```
nifty50-scanner/
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

**Option A — GitHub web UI (easiest):**
1. Click **Add file → Upload files**
2. Drag and drop `app.py` and `requirements.txt`
3. For `.streamlit/config.toml`: click **Add file → Create new file**,
   type `.streamlit/config.toml` as the filename, paste the contents

**Option B — Git CLI:**
```bash
git clone https://github.com/YOUR_USERNAME/nifty50-scanner
cd nifty50-scanner
cp /path/to/app.py .
cp /path/to/requirements.txt .
mkdir .streamlit
cp /path/to/config.toml .streamlit/
git add .
git commit -m "Initial deploy"
git push
```

### Step 3 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select:
   - **Repository**: `YOUR_USERNAME/nifty50-scanner`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**

### Step 4 — Get your URL

Within 2–3 minutes your app will be live at:
```
https://YOUR_USERNAME-nifty50-scanner-app-XXXXX.streamlit.app
```

Share this URL with anyone — no login required to view.

---

## 🖥 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📋 Features

| Feature | Details |
|---|---|
| Signal scan | All 50 Nifty stocks, every 30 seconds |
| Long criteria | RSI ≥ 55 · Price > VWAP · Price ≥ EMA(33) High |
| Short criteria | RSI ≤ 45 · Price < VWAP · Price ≤ EMA(33) Low |
| SL Hunt — Wick | Flags bars where wick > 55% of range against signal direction |
| SL Hunt — Vol Spike | Flags 2× avg volume bars moving against signal direction |
| SL Hunt — RSI Diverge | Flags RSI diverging 8+ pts from price in last 3 bars |
| Data source | NSE India via yfinance (.NS tickers) |
| Intervals | 9-minute OHLCV resampled from 1-min bars |
| Export | Download signals and SL alerts as CSV |
| Auto-refresh | 30-second automatic rescan |

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only.
It does not constitute financial advice. Always do your own research.
