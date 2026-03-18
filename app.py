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

# ════════════════════════════════════════════
# SIDE-BY-SIDE  Long | Short  tables
# ════════════════════════════════════════════

def build_signal_table(hits, mode_long):
    """Build a styled dataframe for either long or short hits."""
    if not hits:
        return None
    ema_k = "emah" if mode_long else "emal"
    c1k   = "lc1"  if mode_long else "sc1"
    c2k   = "lc2"  if mode_long else "sc2"
    c3k   = "lc3"  if mode_long else "sc3"
    rows  = []
    for s in sorted(hits, key=lambda x: x["rsi"], reverse=mode_long):
        rows.append({
            "Symbol":  s["sym"],
            "Price ₹": f"₹{s['price']:,.2f}",
            "Chg %":   f"{'+' if s['chg']>=0 else ''}{s['chg']:.2f}%",
            "RSI":     f'{s["rsi"]:.1f}',
            "VWAP":    f"{s['vwap']:.2f}",
            "EMA(33)": f"{s[ema_k]:.2f}",
            "Vol":     fmt_vol(s["vol"]),
            "C1":      tick(s[c1k]),
            "C2":      tick(s[c2k]),
            "C3":      tick(s[c3k]),
        })
    df = pd.DataFrame(rows)
    bg  = "#d0f5e8" if mode_long else "#fde0e5"
    fg  = "#065c38" if mode_long else "#8f0d20"
    styled = (df.style
                .apply(lambda r: [f"background-color:{bg};color:{fg};font-weight:600"] * len(r), axis=1)
                .set_table_styles([{"selector":"th","props":[
                    ("background-color","#e4e8f2"),("color","#1a1d2e"),
                    ("font-weight","700"),("text-align","center"),
                    ("border","1px solid #b0b8d0"),("font-size","0.8rem")]}])
                .set_properties(**{"font-size":"0.85rem","text-align":"center"}))
    return styled, df

# ── Main two-column view ──
st.markdown("---")
col_long, col_short = st.columns(2, gap="medium")

with col_long:
    st.markdown(f"""
    <div style="background:#d0f5e8;border:1px solid #0a7c4e;border-radius:8px;
                padding:10px 16px;margin-bottom:10px;display:flex;
                align-items:center;justify-content:space-between">
      <span style="color:#065c38;font-weight:700;font-size:1rem">▲ LONG SIGNALS</span>
      <span style="background:#0a7c4e;color:#fff;border-radius:20px;
                   padding:2px 12px;font-weight:700;font-size:0.9rem">{len(long_hits)}</span>
    </div>
    <div style="color:#3a3f5c;font-size:0.78rem;margin-bottom:8px">
      RSI ≥ 55 &nbsp;·&nbsp; Price &gt; VWAP &nbsp;·&nbsp; Price ≥ EMA(33) High
    </div>
    """, unsafe_allow_html=True)
    if long_hits:
        result = build_signal_table(long_hits, True)
        if result:
            styled_l, df_l = result
            st.dataframe(styled_l, use_container_width=True, hide_index=True,
                         height=min(55 + len(long_hits) * 33, 520))
            st.download_button("⬇ Long CSV", df_l.to_csv(index=False),
                               file_name=f"long_{scan_time.replace(' ','_').replace(':','')}.csv",
                               mime="text/csv", key="dl_long")
    else:
        st.markdown("""<div class="warn-banner">No Long signals this cycle.</div>""",
                    unsafe_allow_html=True)

with col_short:
    st.markdown(f"""
    <div style="background:#fde0e5;border:1px solid #c0142e;border-radius:8px;
                padding:10px 16px;margin-bottom:10px;display:flex;
                align-items:center;justify-content:space-between">
      <span style="color:#8f0d20;font-weight:700;font-size:1rem">▼ SHORT SIGNALS</span>
      <span style="background:#c0142e;color:#fff;border-radius:20px;
                   padding:2px 12px;font-weight:700;font-size:0.9rem">{len(short_hits)}</span>
    </div>
    <div style="color:#3a3f5c;font-size:0.78rem;margin-bottom:8px">
      RSI ≤ 45 &nbsp;·&nbsp; Price &lt; VWAP &nbsp;·&nbsp; Price ≤ EMA(33) Low
    </div>
    """, unsafe_allow_html=True)
    if short_hits:
        result = build_signal_table(short_hits, False)
        if result:
            styled_s, df_s = result
            st.dataframe(styled_s, use_container_width=True, hide_index=True,
                         height=min(55 + len(short_hits) * 33, 520))
            st.download_button("⬇ Short CSV", df_s.to_csv(index=False),
                               file_name=f"short_{scan_time.replace(' ','_').replace(':','')}.csv",
                               mime="text/csv", key="dl_short")
    else:
        st.markdown("""<div class="warn-banner">No Short signals this cycle.</div>""",
                    unsafe_allow_html=True)

# ── All Scanned + SL Hunt in expanders below ──
st.markdown("---")
with st.expander(f"📋 All Scanned ({len(signals)} stocks)", expanded=False):
    rows_all = []
    for s in sorted(signals, key=lambda x: x["rsi"], reverse=True):
        sig_label = "▲ LONG" if s.get("long_pass") else ("▼ SHORT" if s.get("short_pass") else "—")
        rows_all.append({
            "Symbol":  s["sym"],
            "Price ₹": f"₹{s['price']:,.2f}",
            "Chg %":   f"{'+' if s['chg']>=0 else ''}{s['chg']:.2f}%",
            "RSI":     f'{s["rsi"]:.1f}',
            "VWAP":    f"{s['vwap']:.2f}",
            "EMA H":   f"{s['emah']:.2f}",
            "EMA L":   f"{s['emal']:.2f}",
            "Volume":  fmt_vol(s["vol"]),
            "Signal":  sig_label,
        })
    def color_all(row):
        if row["Signal"] == "▲ LONG":  return ["background-color:#d0f5e8;color:#065c38"] * len(row)
        if row["Signal"] == "▼ SHORT": return ["background-color:#fde0e5;color:#8f0d20"] * len(row)
        return [""] * len(row)
    df_all = pd.DataFrame(rows_all)
    st.dataframe(
        df_all.style.apply(color_all, axis=1)
              .set_table_styles([{"selector":"th","props":[
                  ("background-color","#e4e8f2"),("color","#1a1d2e"),
                  ("font-weight","700"),("text-align","center")]}]),
        use_container_width=True, hide_index=True,
        height=min(60 + len(rows_all) * 33, 500))

with st.expander(f"⚡ SL Hunt Alerts ({len(sl_alerts)})", expanded=len(sl_alerts) > 0):
    if not sl_alerts:
        st.markdown("""<div class="info-banner">✅ No SL Hunt patterns detected this cycle.</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warn-banner">
        ⚡ <strong>{len(sl_alerts)} alert(s)</strong> &nbsp;·&nbsp;
        🔴 <strong>{sl_highs} HIGH</strong> &nbsp;·&nbsp;
        🟠 <strong>{sl_meds} MED</strong>
        </div>""", unsafe_allow_html=True)
        rows_html = ""
        for a in reversed(sl_alerts):
            css      = "sl-high" if a["severity"] == "HIGH" else "sl-med"
            dir_badge= "badge-long" if a["direction"] == "LONG" else "badge-short"
            dir_text = "▲ LONG" if a["direction"] == "LONG" else "▼ SHORT"
            sev_badge= "badge-high" if a["severity"] == "HIGH" else "badge-med"
            rows_html += f"""<tr class="{css}">
              <td>{a['ts']}</td><td><strong>{a['sym']}</strong></td>
              <td><span class="badge {dir_badge}">{dir_text}</span></td>
              <td><span class="badge {sev_badge}">{a['severity']}</span></td>
              <td>{a['pattern']}</td>
              <td style="font-size:0.82rem">{a['detail']}</td></tr>"""
        st.markdown(f"""<table class="sl-table"><thead><tr>
            <th>Time</th><th>Symbol</th><th>Signal</th><th>Sev</th>
            <th>Pattern</th><th>Detail</th></tr></thead>
            <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
        df_sl = pd.DataFrame(sl_alerts)[["ts","sym","direction","severity","pattern","detail"]]
        st.download_button("⬇ SL Hunt CSV", df_sl.to_csv(index=False),
                           file_name=f"sl_hunt_{scan_time.replace(' ','_').replace(':','')}.csv",
                           mime="text/csv")

# ── Auto-refresh ──
if auto_refresh:
    import time
    st.markdown("""<div style="text-align:right;color:#9098b0;font-size:0.78rem;margin-top:8px">
    🔄 Auto-refreshing every 60 seconds</div>""", unsafe_allow_html=True)
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ── Footer ──
st.markdown(f"""
<div class="footer">
  Data: NSE India · yfinance (.NS) &nbsp;|&nbsp;
  RSI(14) · Session VWAP · EMA(33) &nbsp;|&nbsp;
  9-min OHLCV &nbsp;|&nbsp;
  Last scan: {scan_time} &nbsp;|&nbsp; ⬡ Nifty 50 Signal Scanner
</div>
""", unsafe_allow_html=True)
