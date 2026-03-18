"""
╔══════════════════════════════════════════════════════════════╗
║     Nifty 50 Signal Scanner  —  Streamlit Web App           ║
║     Long / Short signals · RSI · VWAP · EMA · SL Hunt       ║
╚══════════════════════════════════════════════════════════════╝
Deploy free: https://streamlit.io/cloud
"""

import threading
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nifty 50 Signal Scanner",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
NIFTY50_TICKERS: dict = {
    "RELIANCE":   ["RELIANCE.NS"],
    "TCS":        ["TCS.NS"],
    "HDFCBANK":   ["HDFCBANK.NS"],
    "ICICIBANK":  ["ICICIBANK.NS"],
    "INFY":       ["INFY.NS"],
    "HINDUNILVR": ["HINDUNILVR.NS"],
    "ITC":        ["ITC.NS"],
    "SBIN":       ["SBIN.NS"],
    "BHARTIARTL": ["BHARTIARTL.NS"],
    "KOTAKBANK":  ["KOTAKBANK.NS"],
    "LT":         ["LT.NS"],
    "AXISBANK":   ["AXISBANK.NS"],
    "ASIANPAINT": ["ASIANPAINT.NS"],
    "MARUTI":     ["MARUTI.NS"],
    "TITAN":      ["TITAN.NS"],
    "WIPRO":      ["WIPRO.NS"],
    "ULTRACEMCO": ["ULTRACEMCO.NS"],
    "BAJFINANCE": ["BAJFINANCE.NS"],
    "ONGC":       ["ONGC.NS"],
    "NTPC":       ["NTPC.NS"],
    "POWERGRID":  ["POWERGRID.NS"],
    "M&M":        ["M&M.NS"],
    "NESTLEIND":  ["NESTLEIND.NS"],
    "TECHM":      ["TECHM.NS"],
    "JSWSTEEL":   ["JSWSTEEL.NS"],
    "HCLTECH":    ["HCLTECH.NS"],
    "TATAMOTORS": ["TMPV.NS", "TATAMOTORS.NS"],
    "COALINDIA":  ["COALINDIA.NS"],
    "INDUSINDBK": ["INDUSINDBK.NS"],
    "SUNPHARMA":  ["SUNPHARMA.NS"],
    "DRREDDY":    ["DRREDDY.NS"],
    "BAJAJFINSV": ["BAJAJFINSV.NS"],
    "DIVISLAB":   ["DIVISLAB.NS"],
    "CIPLA":      ["CIPLA.NS"],
    "HEROMOTOCO": ["HEROMOTOCO.NS"],
    "GRASIM":     ["GRASIM.NS"],
    "TATACONSUM": ["TATACONSUM.NS"],
    "BAJAJ-AUTO": ["BAJAJ-AUTO.NS"],
    "ADANIENT":   ["ADANIENT.NS"],
    "ADANIPORTS": ["ADANIPORTS.NS"],
    "HINDALCO":   ["HINDALCO.NS"],
    "TATASTEEL":  ["TATASTEEL.NS"],
    "APOLLOHOSP": ["APOLLOHOSP.NS"],
    "LTIM":       ["LTIM.NS"],
    "BEL":        ["BEL.NS"],
    "SHRIRAMFIN": ["SHRIRAMFIN.NS"],
    "TRENT":      ["TRENT.NS"],
    "ETERNAL":    ["ETERNAL.NS", "ZOMATO.NS"],
    "BPCL":       ["BPCL.NS"],
    "EICHERMOT":  ["EICHERMOT.NS"],
}
NIFTY50      = list(NIFTY50_TICKERS.keys())
INTERVAL_MIN = 9
RSI_PERIOD   = 14
EMA_PERIOD   = 33

SL_WICK_RATIO     = 0.55
SL_VOL_SPIKE_MULT = 2.0
SL_RSI_DIVERGE    = 8.0

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════
def fetch_ohlcv(sym: str) -> pd.DataFrame:
    REQUIRED = (RSI_PERIOD + EMA_PERIOD) * INTERVAL_MIN
    tickers  = NIFTY50_TICKERS.get(sym, [f"{sym}.NS"])
    for ticker in tickers:
        for period in ("1d", "5d"):
            try:
                df = yf.download(ticker, interval="1m", period=period,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if not {"open","high","low","close","volume"}.issubset(df.columns):
                    continue
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["close","volume"])
                if len(df) >= REQUIRED:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def resample(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.resample(f"{INTERVAL_MIN}min", label="left", closed="left")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),    close=("close","last"),
                   volume=("volume","sum"))
              .dropna())


def calc_rsi(s: pd.Series) -> pd.Series:
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    al = l.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def calc_ema(s: pd.Series) -> pd.Series:
    return s.ewm(span=EMA_PERIOD, adjust=False).mean()


def fmt_vol(v: int) -> str:
    if v >= 1_000_000: return f"{v/1e6:.2f}M"
    if v >= 1_000:     return f"{v/1e3:.0f}K"
    return str(v)


# ═══════════════════════════════════════════════════════════════
# SIGNAL EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate(sym: str):
    df1 = fetch_ohlcv(sym)
    if df1.empty or len(df1) < RSI_PERIOD * INTERVAL_MIN:
        return None, None, None
    df = resample(df1)
    if len(df) < RSI_PERIOD + EMA_PERIOD:
        return None, None, None

    rsi_s  = calc_rsi(df["close"])
    vwap_s = calc_vwap(df)
    emah_s = calc_ema(df["high"])
    emal_s = calc_ema(df["low"])

    price  = float(df["close"].iloc[-1])
    rsi_v  = float(rsi_s.iloc[-1])
    vwap_v = float(vwap_s.iloc[-1])
    emah_v = float(emah_s.iloc[-1])
    emal_v = float(emal_s.iloc[-1])
    prev   = float(df["close"].iloc[-2]) if len(df) > 1 else price
    chg    = (price - prev) / prev * 100
    vol    = int(df["volume"].iloc[-1])

    lc1 = rsi_v >= 55;  lc2 = price > vwap_v;  lc3 = price >= emah_v
    sc1 = rsi_v <= 45;  sc2 = price < vwap_v;  sc3 = price <= emal_v

    sig = dict(
        sym=sym, price=round(price,2), chg=round(chg,2),
        rsi=round(rsi_v,1), vwap=round(vwap_v,2),
        emah=round(emah_v,2), emal=round(emal_v,2), vol=vol,
        lc1=lc1, lc2=lc2, lc3=lc3, long_pass=lc1 and lc2 and lc3,
        sc1=sc1, sc2=sc2, sc3=sc3, short_pass=sc1 and sc2 and sc3,
    )
    return sig, df, rsi_s


# ═══════════════════════════════════════════════════════════════
# SL HUNT DETECTOR
# ═══════════════════════════════════════════════════════════════
def sl_hunt_analyse(sym, sig, df, rsi_s):
    alerts = []
    if df is None or len(df) < 5:
        return alerts
    direction = "LONG" if sig.get("long_pass") else "SHORT"
    ts = datetime.now().strftime("%H:%M:%S")

    # ── 1. Wick reversal ──
    row  = df.iloc[-1]
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng  = h - l
    if rng >= 0.5:
        uw = h - max(o, c)
        lw = min(o, c) - l
        if direction == "LONG" and uw / rng >= SL_WICK_RATIO:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="UPPER WICK REVERSAL",
                detail=f"Wick {uw:.2f} / Range {rng:.2f} ({uw/rng:.0%})  H:{h:.2f} L:{l:.2f} C:{c:.2f}",
                severity="HIGH" if uw/rng >= 0.70 else "MED"))
        if direction == "SHORT" and lw / rng >= SL_WICK_RATIO:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="LOWER WICK REVERSAL",
                detail=f"Wick {lw:.2f} / Range {rng:.2f} ({lw/rng:.0%})  H:{h:.2f} L:{l:.2f} C:{c:.2f}",
                severity="HIGH" if lw/rng >= 0.70 else "MED"))

    # ── 2. Volume spike ──
    if len(df) >= 6:
        avg_vol   = df["volume"].iloc[-6:-1].mean()
        vol_ratio = row["volume"] / avg_vol if avg_vol > 0 else 0
        price_chg = row["close"] - row["open"]
        if direction == "LONG" and vol_ratio >= SL_VOL_SPIKE_MULT and price_chg < 0:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="VOL SPIKE + BEARISH BAR",
                detail=f"Volume {vol_ratio:.1f}× avg  Close {row['close']:.2f} (Open {row['open']:.2f})  Long SL sweep",
                severity="HIGH" if vol_ratio >= SL_VOL_SPIKE_MULT * 1.5 else "MED"))
        if direction == "SHORT" and vol_ratio >= SL_VOL_SPIKE_MULT and price_chg > 0:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="VOL SPIKE + BULLISH BAR",
                detail=f"Volume {vol_ratio:.1f}× avg  Close {row['close']:.2f} (Open {row['open']:.2f})  Short SL sweep",
                severity="HIGH" if vol_ratio >= SL_VOL_SPIKE_MULT * 1.5 else "MED"))

    # ── 3. RSI divergence ──
    if rsi_s is not None and len(rsi_s) >= 4 and len(df) >= 4:
        prices = df["close"].iloc[-4:].values
        rsis   = rsi_s.iloc[-4:].values
        if direction == "LONG" and prices[-1] > prices[-2] and rsis[-1] < rsis[-2] - SL_RSI_DIVERGE:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="BEARISH RSI DIVERGENCE",
                detail=f"Price ↑ {prices[-2]:.2f}→{prices[-1]:.2f}  RSI ↓ {rsis[-2]:.1f}→{rsis[-1]:.1f}  Momentum fading",
                severity="MED"))
        if direction == "SHORT" and prices[-1] < prices[-2] and rsis[-1] > rsis[-2] + SL_RSI_DIVERGE:
            alerts.append(dict(ts=ts, sym=sym, direction=direction,
                pattern="BULLISH RSI DIVERGENCE",
                detail=f"Price ↓ {prices[-2]:.2f}→{prices[-1]:.2f}  RSI ↑ {rsis[-2]:.1f}→{rsis[-1]:.1f}  Momentum fading",
                severity="MED"))

    return alerts


# ═══════════════════════════════════════════════════════════════
# SCANNING  (cached with TTL so all users share one fetch cycle)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=30, show_spinner=False)
def run_scan():
    signals    = []
    sl_alerts  = []
    for sym in NIFTY50:
        sig, df, rsi_s = evaluate(sym)
        if sig:
            signals.append(sig)
            if sig.get("long_pass") or sig.get("short_pass"):
                sl_alerts.extend(sl_hunt_analyse(sym, sig, df, rsi_s))
    return signals, sl_alerts


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS  — bright, professional theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f7 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: #ffffff !important; border-bottom: 2px solid #b0b8d0; }

/* ── title bar ── */
.title-bar {
    background: #ffffff;
    border: 1px solid #b0b8d0;
    border-radius: 10px;
    padding: 18px 28px 14px 28px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.title-bar h1 { margin: 0; font-size: 1.6rem; color: #1a1d2e; font-weight: 700; }
.title-bar p  { margin: 0; font-size: 0.85rem; color: #6b718e; }

/* ── metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.metric-card {
    background: #ffffff;
    border: 1px solid #b0b8d0;
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 140px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card .label { font-size: 0.75rem; color: #6b718e; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1a1d2e; line-height: 1.2; }
.metric-card .sub   { font-size: 0.75rem; color: #9098b0; }

/* ── tab bar ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #e4e8f2;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #b0b8d0;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 6px !important;
    font-weight: 600 !important;
    color: #3a3f5c !important;
    padding: 6px 20px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #0057b8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}

/* ── dataframe / table ── */
[data-testid="stDataFrame"] { border: 1px solid #b0b8d0 !important; border-radius: 8px; overflow: hidden; }

/* ── SL hunt table rows ── */
.sl-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.sl-table th {
    background: #e4e8f2; color: #1a1d2e; font-weight: 700;
    padding: 8px 12px; text-align: left;
    border-bottom: 2px solid #b0b8d0;
}
.sl-table td { padding: 7px 12px; border-bottom: 1px solid #e4e8f2; color: #1a1d2e; }
.sl-table tr:nth-child(even) td { background: #f5f7fc; }
.sl-high td { background: #ffe0b2 !important; color: #b00020 !important; font-weight: 600; }
.sl-med  td { background: #fff3e0 !important; color: #7a4000 !important; }

/* ── badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; margin: 2px;
}
.badge-long  { background: #d0f5e8; color: #065c38; border: 1px solid #0a7c4e; }
.badge-short { background: #fde0e5; color: #8f0d20; border: 1px solid #c0142e; }
.badge-high  { background: #ffe0b2; color: #b00020; border: 1px solid #e65100; }
.badge-med   { background: #fff3e0; color: #7a4000; border: 1px solid #bf6000; }

/* ── info banner ── */
.info-banner {
    background: #e8f0fe; border: 1px solid #0057b8; border-radius: 8px;
    padding: 10px 18px; color: #0057b8; font-size: 0.85rem;
    margin-bottom: 12px;
}
.warn-banner {
    background: #fff3e0; border: 1px solid #bf6000; border-radius: 8px;
    padding: 10px 18px; color: #7a4000; font-size: 0.85rem;
    margin-bottom: 12px;
}

/* ── footer ── */
.footer { text-align: center; color: #9098b0; font-size: 0.78rem; margin-top: 24px; padding: 12px; border-top: 1px solid #b0b8d0; }

/* hide streamlit chrome */
#MainMenu, footer, [data-testid="collapsedControl"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def tick(v): return "✓" if v else "·"

def chg_color(v):
    return f'<span style="color:{"#0a7c4e" if v >= 0 else "#c0142e"};font-weight:600">{("+" if v>=0 else "")}{v:.2f}%</span>'

def signal_badge(s):
    if s.get("long_pass"):
        return '<span class="badge badge-long">▲ LONG</span>'
    if s.get("short_pass"):
        return '<span class="badge badge-short">▼ SHORT</span>'
    return ""


# ═══════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════

# ── Title bar ──
st.markdown("""
<div class="title-bar">
  <div style="font-size:2.2rem">⬡</div>
  <div>
    <h1>Nifty 50 Signal Scanner</h1>
    <p>9-min OHLCV &nbsp;·&nbsp; RSI(14) &nbsp;·&nbsp; Session VWAP &nbsp;·&nbsp; EMA(33 High/Low) &nbsp;·&nbsp; SL Hunt Detector &nbsp;·&nbsp; NSE India via yfinance</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──
with st.sidebar:
    st.markdown("### ⚙️ Scanner Settings")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.markdown("---")
    st.markdown("**Long criteria**")
    st.markdown("- RSI(14) ≥ 55\n- Price > VWAP\n- Price ≥ EMA(33) High")
    st.markdown("**Short criteria**")
    st.markdown("- RSI(14) ≤ 45\n- Price < VWAP\n- Price ≤ EMA(33) Low")
    st.markdown("---")
    st.markdown("**SL Hunt thresholds**")
    st.markdown(f"- Wick ratio: {SL_WICK_RATIO:.0%}\n- Vol spike: {SL_VOL_SPIKE_MULT}×\n- RSI diverge: {SL_RSI_DIVERGE} pts")
    st.markdown("---")
    if st.button("⟳  Scan Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ── Mode — stored in session_state, toggled by buttons only ──
if "is_long" not in st.session_state:
    st.session_state["is_long"] = True   # default: Long
is_long = st.session_state["is_long"]

# ── Run scan ──
with st.spinner("🔄 Scanning all 50 Nifty stocks…"):
    signals, sl_alerts = run_scan()

# ── IST scan time (UTC+5:30) ──
from datetime import timezone, timedelta
ist = timezone(timedelta(hours=5, minutes=30))
scan_time = datetime.now(ist).strftime("%H:%M:%S IST")

# ── Metrics ──
key      = "long_pass" if is_long else "short_pass"
hits     = [s for s in signals if s.get(key)]
avg_rsi  = sum(s["rsi"] for s in hits) / len(hits) if hits else 0
sl_highs = sum(1 for a in sl_alerts if a["severity"] == "HIGH")
sl_meds  = len(sl_alerts) - sl_highs
# Count both long and short for display
long_hits  = [s for s in signals if s.get("long_pass")]
short_hits = [s for s in signals if s.get("short_pass")]

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Scanned</div>
    <div class="value">50</div>
    <div class="sub">Nifty 50 stocks</div>
  </div>
  <div class="metric-card">
    <div class="label">▲ Long Signals</div>
    <div class="value" style="color:#0a7c4e">{len(long_hits)}</div>
    <div class="sub">qualifying</div>
  </div>
  <div class="metric-card">
    <div class="label">▼ Short Signals</div>
    <div class="value" style="color:#c0142e">{len(short_hits)}</div>
    <div class="sub">qualifying</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg RSI — {"Long" if is_long else "Short"}</div>
    <div class="value">{f"{avg_rsi:.1f}" if hits else "—"}</div>
    <div class="sub">9-min bars</div>
  </div>
  <div class="metric-card">
    <div class="label">SL Hunt Alerts</div>
    <div class="value" style="color:#b07800">{len(sl_alerts)}</div>
    <div class="sub">🔴 {sl_highs} HIGH &nbsp; 🟠 {sl_meds} MED</div>
  </div>
  <div class="metric-card">
    <div class="label">Last Scan</div>
    <div class="value" style="font-size:1rem;padding-top:6px">{scan_time}</div>
    <div class="sub">&nbsp;</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Mode toggle buttons ──
col1, col2, col3 = st.columns([1.2, 1.2, 5.6])
with col1:
    long_type = "primary" if is_long else "secondary"
    if st.button(f"▲ Long ({len(long_hits)})", type=long_type, use_container_width=True):
        st.session_state["is_long"] = True
        st.rerun()
with col2:
    short_type = "primary" if not is_long else "secondary"
    if st.button(f"▼ Short ({len(short_hits)})", type=short_type, use_container_width=True):
        st.session_state["is_long"] = False
        st.rerun()

# ── Tabs ──
tab_signals, tab_all, tab_sl = st.tabs([
    f"{'▲ Long' if is_long else '▼ Short'} Signals ({len(hits)})",
    f"All Scanned ({len(signals)})",
    f"⚡ SL Hunt Alerts ({len(sl_alerts)})",
])

# ════════════════════════════════════════════
# TAB 1 — qualifying signals
# ════════════════════════════════════════════
with tab_signals:
    if not hits:
        st.markdown(f"""
        <div class="warn-banner">
        No {'Long' if is_long else 'Short'} signals found in this scan cycle.
        The scanner refreshes every 30 seconds — signals appear when all 3 criteria are met simultaneously.
        </div>""", unsafe_allow_html=True)
    else:
        ema_k = "emah" if is_long else "emal"
        c1k   = "lc1"  if is_long else "sc1"
        c2k   = "lc2"  if is_long else "sc2"
        c3k   = "lc3"  if is_long else "sc3"

        # Build display dataframe
        rows = []
        for s in sorted(hits, key=lambda x: x["rsi"], reverse=is_long):  # Long: high RSI first; Short: low RSI first
            rows.append({
                "Symbol":   s["sym"],
                "Price ₹":  f"₹ {s['price']:,.2f}",
                "Chg %":    f"{'+' if s['chg']>=0 else ''}{s['chg']:.2f}%",
                "RSI(14)":  f'{s["rsi"]:.1f}',
                "VWAP":     f"{s['vwap']:.2f}",
                f"EMA(33)": f"{s[ema_k]:.2f}",
                "Volume":   fmt_vol(s["vol"]),
                "C1 RSI":   tick(s[c1k]),
                "C2 VWAP":  tick(s[c2k]),
                "C3 EMA":   tick(s[c3k]),
                "Signal":   "▲ LONG" if is_long else "▼ SHORT",
            })

        df_display = pd.DataFrame(rows)

        # Style the dataframe
        def style_row(row):
            base = "background-color: #d0f5e8; color: #065c38;" if is_long \
                   else "background-color: #fde0e5; color: #8f0d20;"
            return [base] * len(row)

        styled = (df_display.style
                  .apply(style_row, axis=1)
                  .set_properties(**{"font-weight": "600", "text-align": "center"})
                  .set_table_styles([{
                      "selector": "th",
                      "props": [("background-color","#e4e8f2"),
                                ("color","#1a1d2e"),
                                ("font-weight","700"),
                                ("text-align","center"),
                                ("border","1px solid #b0b8d0")]
                  }]))

        st.dataframe(styled, use_container_width=True, hide_index=True,
                     height=min(60 + len(rows) * 35, 500))

        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv,
                           file_name=f"nifty50_{'long' if is_long else 'short'}_{scan_time.replace(':','')}.csv",
                           mime="text/csv")

# ════════════════════════════════════════════
# TAB 2 — all scanned
# ════════════════════════════════════════════
with tab_all:
    st.markdown(f"""
    <div class="info-banner">
    Showing all {len(signals)} stocks for which data was fetched successfully.
    Stocks highlighted in green/red qualify for Long/Short signals.
    </div>""", unsafe_allow_html=True)

    rows_all = []
    for s in sorted(signals, key=lambda x: x["rsi"], reverse=True):
        sig_label = "▲ LONG" if s.get("long_pass") else ("▼ SHORT" if s.get("short_pass") else "—")
        rows_all.append({
            "Symbol":  s["sym"],
            "Price ₹": f"₹ {s['price']:,.2f}",
            "Chg %":   f"{'+' if s['chg']>=0 else ''}{s['chg']:.2f}%",
            "RSI(14)": f'{s["rsi"]:.1f}',
            "VWAP":    f"{s['vwap']:.2f}",
            "EMA H":   f"{s['emah']:.2f}",
            "EMA L":   f"{s['emal']:.2f}",
            "Volume":  fmt_vol(s["vol"]),
            "Signal":  sig_label,
        })

    def color_all(row):
        if row["Signal"] == "▲ LONG":
            return ["background-color:#d0f5e8;color:#065c38"] * len(row)
        if row["Signal"] == "▼ SHORT":
            return ["background-color:#fde0e5;color:#8f0d20"] * len(row)
        return [""] * len(row)

    df_all = pd.DataFrame(rows_all)
    st.dataframe(
        df_all.style.apply(color_all, axis=1)
              .set_table_styles([{"selector":"th","props":[
                  ("background-color","#e4e8f2"),("color","#1a1d2e"),
                  ("font-weight","700"),("text-align","center")]}]),
        use_container_width=True, hide_index=True,
        height=min(60 + len(rows_all) * 35, 600)
    )

# ════════════════════════════════════════════
# TAB 3 — SL Hunt alerts
# ════════════════════════════════════════════
with tab_sl:
    if not sl_alerts:
        st.markdown("""
        <div class="info-banner">
        ✅ No SL Hunt patterns detected in this scan cycle.
        Alerts appear only for stocks qualifying for Long or Short signals.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warn-banner">
        ⚡ <strong>{len(sl_alerts)} SL Hunt pattern(s) detected</strong> across qualifying stocks.
        🔴 <strong>{sl_highs} HIGH</strong> severity &nbsp;·&nbsp; 🟠 <strong>{sl_meds} MED</strong> severity.
        These are warning signals — confirm with price action before acting.
        </div>""", unsafe_allow_html=True)

        # Build HTML table
        rows_html = ""
        for a in reversed(sl_alerts):
            css = "sl-high" if a["severity"] == "HIGH" else "sl-med"
            dir_badge = "badge-long" if a["direction"] == "LONG" else "badge-short"
            rows_html += f"""
            <tr class="{css}">
              <td>{a['ts']}</td>
              <td><strong>{a['sym']}</strong></td>
              <td><span class="badge {dir_badge}">{"▲ LONG" if a['direction']=="LONG" else "▼ SHORT"}</span></td>
              <td><span class="badge {"badge-high" if a['severity']=="HIGH" else "badge-med"}">{a['severity']}</span></td>
              <td>{a['pattern']}</td>
              <td style="font-size:0.82rem">{a['detail']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="sl-table">
          <thead><tr>
            <th>Time</th><th>Symbol</th><th>Signal</th><th>Severity</th>
            <th>Pattern</th><th>Detail</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # Download
        df_sl = pd.DataFrame(sl_alerts)[["ts","sym","direction","severity","pattern","detail"]]
        st.download_button("⬇ Download SL Hunt CSV", df_sl.to_csv(index=False),
                           file_name=f"sl_hunt_{scan_time.replace(':','')}.csv",
                           mime="text/csv")

# ── Auto-refresh ──
if auto_refresh:
    import time
    st.markdown("""
    <div style="text-align:right;color:#9098b0;font-size:0.8rem;margin-top:8px">
    🔄 Auto-refreshing every 60 seconds
    </div>""", unsafe_allow_html=True)
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ── Footer ──
st.markdown(f"""
<div class="footer">
  Data: NSE India · yfinance (.NS) &nbsp;|&nbsp;
  Indicators: RSI(14) · Session VWAP · EMA(33) &nbsp;|&nbsp;
  Intervals: 9-min OHLCV &nbsp;|&nbsp;
  Last scan: {scan_time} IST &nbsp;|&nbsp;
  ⬡ Nifty 50 Signal Scanner
</div>
""", unsafe_allow_html=True)
