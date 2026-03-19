"""
╔══════════════════════════════════════════════════════════════╗
║     Nifty 50 Signal Scanner  —  Streamlit Web App           ║
║     Long / Short signals · RSI · VWAP · EMA · SL Hunt       ║
╚══════════════════════════════════════════════════════════════╝
Deploy free: https://streamlit.io/cloud
"""

import threading
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
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
    """
    Download 1-min OHLCV.  Every yf.download call runs in a daemon thread
    with a hard 20-second timeout so a single hung request cannot stall
    the entire scan on Streamlit Cloud.
    """
    REQUIRED = (RSI_PERIOD + EMA_PERIOD) * INTERVAL_MIN
    tickers  = NIFTY50_TICKERS.get(sym, [f"{sym}.NS"])
    for ticker in tickers:
        for period in ("1d", "5d"):
            try:
                result = []

                def _dl(t=ticker, p=period):
                    try:
                        df = yf.download(t, interval="1m", period=p,
                                         progress=False, auto_adjust=True)
                        result.append(df)
                    except Exception:
                        pass

                thr = threading.Thread(target=_dl, daemon=True)
                thr.start()
                thr.join(timeout=20)          # give up after 20 s

                if not result:
                    continue
                df = result[0]
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                    continue
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["close", "volume"])
                if len(df) >= REQUIRED:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def resample(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.resample(f"{INTERVAL_MIN}min", label="left", closed="left")
              .agg(open=("open", "first"), high=("high", "max"),
                   low=("low", "min"),     close=("close", "last"),
                   volume=("volume", "sum"))
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


def calc_ema_high(df: pd.DataFrame) -> pd.Series:
    return df["high"].ewm(span=EMA_PERIOD, adjust=False).mean()


def calc_ema_low(df: pd.DataFrame) -> pd.Series:
    return df["low"].ewm(span=EMA_PERIOD, adjust=False).mean()


def fmt_vol(v: int) -> str:
    if v >= 1_000_000: return f"{v/1e6:.2f}M"
    if v >= 1_000:     return f"{v/1e3:.0f}K"
    return str(v)


# ═══════════════════════════════════════════════════════════════
# SL HUNT HELPERS
# ═══════════════════════════════════════════════════════════════
def _sl_ts_ist(df_resampled: pd.DataFrame) -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    ts  = df_resampled.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.astimezone(ist).strftime("%H:%M:%S")


def _check_wick(sym, direction, df, ts):
    alerts = []
    latest = df.iloc[-1]
    o, h, l, c = latest["open"], latest["high"], latest["low"], latest["close"]
    bar_range = h - l
    if bar_range < 0.5:
        return alerts
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if direction == "LONG" and upper_wick / bar_range >= SL_WICK_RATIO:
        alerts.append({
            "ts": ts, "sym": sym, "direction": direction,
            "pattern": "UPPER WICK REVERSAL",
            "detail": (f"Upper wick {upper_wick:.2f}  |  Bar range {bar_range:.2f}  "
                       f"|  Wick/Range {upper_wick/bar_range:.0%}  "
                       f"|  Bar H:{h:.2f} L:{l:.2f} C:{c:.2f}"),
            "severity": "HIGH" if upper_wick / bar_range >= 0.70 else "MED",
        })
    if direction == "SHORT" and lower_wick / bar_range >= SL_WICK_RATIO:
        alerts.append({
            "ts": ts, "sym": sym, "direction": direction,
            "pattern": "LOWER WICK REVERSAL",
            "detail": (f"Lower wick {lower_wick:.2f}  |  Bar range {bar_range:.2f}  "
                       f"|  Wick/Range {lower_wick/bar_range:.0%}  "
                       f"|  Bar H:{h:.2f} L:{l:.2f} C:{c:.2f}"),
            "severity": "HIGH" if lower_wick / bar_range >= 0.70 else "MED",
        })
    return alerts


def _check_vol_spike(sym, direction, df, ts):
    alerts = []
    if len(df) < 6:
        return alerts
    avg_vol = df["volume"].iloc[-6:-1].mean()
    if avg_vol <= 0:
        return alerts
    latest    = df.iloc[-1]
    vol_ratio = latest["volume"] / avg_vol
    price_chg = latest["close"] - latest["open"]
    if direction == "LONG" and vol_ratio >= SL_VOL_SPIKE_MULT and price_chg < 0:
        alerts.append({
            "ts": ts, "sym": sym, "direction": direction,
            "pattern": "VOL SPIKE + BEARISH BAR",
            "detail": (f"Volume {vol_ratio:.1f}x avg  |  "
                       f"Bar close {latest['close']:.2f} (open {latest['open']:.2f})  |  "
                       f"Potential long SL sweep"),
            "severity": "HIGH" if vol_ratio >= SL_VOL_SPIKE_MULT * 1.5 else "MED",
        })
    if direction == "SHORT" and vol_ratio >= SL_VOL_SPIKE_MULT and price_chg > 0:
        alerts.append({
            "ts": ts, "sym": sym, "direction": direction,
            "pattern": "VOL SPIKE + BULLISH BAR",
            "detail": (f"Volume {vol_ratio:.1f}x avg  |  "
                       f"Bar close {latest['close']:.2f} (open {latest['open']:.2f})  |  "
                       f"Potential short SL sweep"),
            "severity": "HIGH" if vol_ratio >= SL_VOL_SPIKE_MULT * 1.5 else "MED",
        })
    return alerts


def _check_rsi_divergence(sym, direction, df, rsi_series, ts):
    alerts = []
    if rsi_series is None or len(rsi_series) < 4 or len(df) < 4:
        return alerts
    prices = df["close"].iloc[-4:].values
    rsis   = rsi_series.iloc[-4:].values
    if direction == "LONG":
        if prices[-1] > prices[-2] and rsis[-1] < rsis[-2] - SL_RSI_DIVERGE:
            alerts.append({
                "ts": ts, "sym": sym, "direction": direction,
                "pattern": "BEARISH RSI DIVERGENCE",
                "detail": (f"Price up {prices[-2]:.2f} to {prices[-1]:.2f}  |  "
                           f"RSI down {rsis[-2]:.1f} to {rsis[-1]:.1f}  |  "
                           f"Momentum fading - SL hunt likely"),
                "severity": "MED",
            })
    if direction == "SHORT":
        if prices[-1] < prices[-2] and rsis[-1] > rsis[-2] + SL_RSI_DIVERGE:
            alerts.append({
                "ts": ts, "sym": sym, "direction": direction,
                "pattern": "BULLISH RSI DIVERGENCE",
                "detail": (f"Price down {prices[-2]:.2f} to {prices[-1]:.2f}  |  "
                           f"RSI up {rsis[-2]:.1f} to {rsis[-1]:.1f}  |  "
                           f"Momentum fading - SL hunt likely"),
                "severity": "MED",
            })
    return alerts


# ═══════════════════════════════════════════════════════════════
# SIGNAL EVALUATION  (single fetch covers signals + SL Hunt)
# ═══════════════════════════════════════════════════════════════
def evaluate_with_sl_hunt(sym: str):
    """Returns (sig_dict, df_resampled, rsi_series) or (None, None, None)."""
    df1 = fetch_ohlcv(sym)
    if df1.empty or len(df1) < RSI_PERIOD * INTERVAL_MIN:
        return None, None, None
    df = resample(df1)
    if len(df) < RSI_PERIOD + EMA_PERIOD:
        return None, None, None

    rsi_s  = calc_rsi(df["close"])
    vwap_s = calc_vwap(df)
    emah_s = calc_ema_high(df)
    emal_s = calc_ema_low(df)

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
        sym=sym, price=round(price, 2), chg=round(chg, 2),
        rsi=round(rsi_v, 1), vwap=round(vwap_v, 2),
        emah=round(emah_v, 2), emal=round(emal_v, 2), vol=vol,
        lc1=lc1, lc2=lc2, lc3=lc3, long_pass=lc1 and lc2 and lc3,
        sc1=sc1, sc2=sc2, sc3=sc3, short_pass=sc1 and sc2 and sc3,
    )
    return sig, df, rsi_s


# ═══════════════════════════════════════════════════════════════
# SCANNING  — one cached function does EVERYTHING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=55, show_spinner=False)
def run_scan():
    """
    Fetches every Nifty 50 stock ONCE.
    Signals + SL Hunt both run on the same DataFrame — no second fetch.

    BUG FIXED: The original code called run_sl_hunt() separately which
    re-downloaded every qualifying stock a second time.  That doubled the
    yfinance calls and caused Streamlit Cloud to time out.

    TTL is 55 s (just under the 60 s browser refresh) so every page reload
    always gets a fresh scan instead of stale cached data.
    """
    signals   = []
    sl_alerts = []

    for sym in NIFTY50:
        try:
            sig, df, rsi_s = evaluate_with_sl_hunt(sym)
            if sig is None:
                continue
            signals.append(sig)

            if sig.get("long_pass"):
                direction = "LONG"
            elif sig.get("short_pass"):
                direction = "SHORT"
            else:
                direction = None

            # SL Hunt runs inline — reuses the df already in memory, zero extra fetches
            if direction and df is not None and len(df) >= 5:
                ts = _sl_ts_ist(df)
                sl_alerts += _check_wick(sym, direction, df, ts)
                sl_alerts += _check_vol_spike(sym, direction, df, ts)
                sl_alerts += _check_rsi_divergence(sym, direction, df, rsi_s, ts)

        except Exception:
            continue

    return signals, sl_alerts


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f7 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: #ffffff !important; border-bottom: 2px solid #b0b8d0; }
.title-bar {
    background: #ffffff; border: 1px solid #b0b8d0; border-radius: 10px;
    padding: 18px 28px 14px 28px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.title-bar h1 { margin: 0; font-size: 1.6rem; color: #1a1d2e; font-weight: 700; }
.title-bar p  { margin: 0; font-size: 0.85rem; color: #6b718e; }
.metric-row { display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.metric-card {
    background: #ffffff; border: 1px solid #b0b8d0; border-radius: 8px;
    padding: 12px 20px; min-width: 140px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card .label { font-size: 0.75rem; color: #6b718e; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1a1d2e; line-height: 1.2; }
.metric-card .sub   { font-size: 0.75rem; color: #9098b0; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; margin: 2px; }
.badge-long  { background: #d0f5e8; color: #065c38; border: 1px solid #0a7c4e; }
.badge-short { background: #fde0e5; color: #8f0d20; border: 1px solid #c0142e; }
.footer { text-align: center; color: #9098b0; font-size: 0.78rem; margin-top: 24px; padding: 12px; border-top: 1px solid #b0b8d0; }
#MainMenu, footer, [data-testid="collapsedControl"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

TABLE_CSS = """
.sig-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:4px; }
.sig-table th { background:#2c3150; color:#ffffff; font-weight:700; padding:7px 8px;
                text-align:center; border-bottom:2px solid #5a6080; font-size:0.78rem; }
.sig-table td { padding:6px 8px; text-align:center; border-bottom:1px solid #d0d5e8; color:#1a1d2e; }
.sig-table tr:nth-child(even) td { background:#e8ecf5; }
.sig-table tr:nth-child(odd)  td { background:#f5f7fc; }
.long-tbl  tr:hover td { background:#a8e6c8 !important; }
.short-tbl tr:hover td { background:#f5b0bc !important; }
.sl-tbl th { background:#5a2d00; color:#ffffff; }
.sl-tbl tr.sl-high td { background:#b00020 !important; color:#ffffff !important; font-weight:700; }
.sl-tbl tr.sl-med  td { background:#8a4e00 !important; color:#ffffff !important; font-weight:600; }
.section-hdr { border-radius:8px; padding:10px 16px; margin-bottom:6px; display:flex; align-items:center; justify-content:space-between; }
.stat-pill { display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:700; margin:0 4px; }
"""


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def signal_table_html(hits, mode_long):
    if not hits:
        return "", ""
    ema_k   = "emah" if mode_long else "emal"
    ema_hdr = "EMA(33)H" if mode_long else "EMA(33)L"
    c1k = "lc1" if mode_long else "sc1"
    c2k = "lc2" if mode_long else "sc2"
    c3k = "lc3" if mode_long else "sc3"
    color = "#065c38" if mode_long else "#8f0d20"
    rows  = ""
    for s in sorted(hits, key=lambda x: x["rsi"], reverse=mode_long):
        chg_c = "#0a7c4e" if s["chg"] >= 0 else "#c0142e"
        chg_s = f"{'+' if s['chg']>=0 else ''}{s['chg']:.2f}%"
        c1 = "OK" if s[c1k] else "."
        c2 = "OK" if s[c2k] else "."
        c3 = "OK" if s[c3k] else "."
        rows += f"""<tr>
          <td style="font-weight:700;color:{color}">{s['sym']}</td>
          <td>Rs{s['price']:,.2f}</td>
          <td style="color:{chg_c};font-weight:600">{chg_s}</td>
          <td style="font-weight:700">{s['rsi']:.1f}</td>
          <td>{s['vwap']:.2f}</td>
          <td>{s[ema_k]:.2f}</td>
          <td>{fmt_vol(s['vol'])}</td>
          <td style="color:#0a7c4e;font-weight:700">{c1}</td>
          <td style="color:#0a7c4e;font-weight:700">{c2}</td>
          <td style="color:#0a7c4e;font-weight:700">{c3}</td>
        </tr>"""
    return rows, ema_hdr


def build_sl_table(alerts):
    if not alerts:
        return "<p style='color:#9098b0;font-size:0.82rem;padding:6px 0'>No SL Hunt alerts for this group.</p>"
    rows = ""
    for a in reversed(alerts):
        css     = "sl-high" if a["severity"] == "HIGH" else "sl-med"
        dir_txt = "LONG"  if a["direction"] == "LONG" else "SHORT"
        rows += f"""<tr class="{css}">
          <td>{a['ts']}</td><td><b>{a['sym']}</b></td>
          <td>{dir_txt}</td><td>{a['severity']}</td>
          <td>{a['pattern']}</td>
          <td style="text-align:left">{a['detail']}</td></tr>"""
    return f"""<table class="sig-table sl-tbl">
      <thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Sev</th>
      <th>Pattern</th><th>Detail</th></tr></thead>
      <tbody>{rows}</tbody></table>"""


def sl_hunt_rows_for(hits_syms, all_alerts):
    return [a for a in all_alerts if a["sym"] in hits_syms]


# ═══════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="title-bar">
  <div style="font-size:2.2rem">X</div>
  <div>
    <h1>Nifty 50 Signal Scanner</h1>
    <p>9-min OHLCV &nbsp; RSI(14) &nbsp; Session VWAP &nbsp; EMA(33) of 9-min High/Low &nbsp; SL Hunt Detector &nbsp; NSE India via yfinance</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    st.markdown("---")
    st.markdown("**Long criteria**")
    st.markdown("- RSI(14) >= 55\n- Price > VWAP\n- Price >= EMA(33) of 9-min High")
    st.markdown("**Short criteria**")
    st.markdown("- RSI(14) <= 45\n- Price < VWAP\n- Price <= EMA(33) of 9-min Low")
    st.markdown("---")
    st.markdown("**SL Hunt thresholds**")
    st.markdown(f"- Wick ratio: {SL_WICK_RATIO:.0%}\n- Vol spike: {SL_VOL_SPIKE_MULT}x\n- RSI diverge: {SL_RSI_DIVERGE} pts")
    st.markdown("---")
    if st.button("Scan Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

if "is_long" not in st.session_state:
    st.session_state["is_long"] = True

# ── Run scan (signals + SL Hunt in one single cached pass) ──
with st.spinner("Scanning all 50 Nifty stocks..."):
    signals, sl_alerts = run_scan()

ist       = timezone(timedelta(hours=5, minutes=30))
scan_time = datetime.now(ist).strftime("%H:%M:%S IST")

long_hits  = [s for s in signals if s.get("long_pass")]
short_hits = [s for s in signals if s.get("short_pass")]
avg_rsi_long  = sum(s["rsi"] for s in long_hits)  / len(long_hits)  if long_hits  else 0.0
avg_rsi_short = sum(s["rsi"] for s in short_hits) / len(short_hits) if short_hits else 0.0
sl_highs = sum(1 for a in sl_alerts if a["severity"] == "HIGH")
sl_meds  = len(sl_alerts) - sl_highs

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Scanned</div><div class="value">50</div><div class="sub">Nifty 50 stocks</div>
  </div>
  <div class="metric-card">
    <div class="label">Long Signals</div>
    <div class="value" style="color:#0a7c4e">{len(long_hits)}</div><div class="sub">qualifying</div>
  </div>
  <div class="metric-card">
    <div class="label">Short Signals</div>
    <div class="value" style="color:#c0142e">{len(short_hits)}</div><div class="sub">qualifying</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg RSI Long</div>
    <div class="value" style="color:#0a7c4e">{f"{avg_rsi_long:.1f}" if long_hits else "--"}</div>
    <div class="sub">9-min bars</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg RSI Short</div>
    <div class="value" style="color:#c0142e">{f"{avg_rsi_short:.1f}" if short_hits else "--"}</div>
    <div class="sub">9-min bars</div>
  </div>
  <div class="metric-card">
    <div class="label">SL Hunt Alerts</div>
    <div class="value" style="color:#b07800">{len(sl_alerts)}</div>
    <div class="sub">{sl_highs} HIGH / {sl_meds} MED</div>
  </div>
  <div class="metric-card">
    <div class="label">Last Scan</div>
    <div class="value" style="font-size:1rem;padding-top:6px">{scan_time}</div>
    <div class="sub">&nbsp;</div>
  </div>
</div>
""", unsafe_allow_html=True)

rc1, rc2 = st.columns([1, 7])
with rc1:
    if st.button("Refresh Now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with rc2:
    st.markdown(f"""
    <div style="padding:8px 0;color:#6b718e;font-size:0.85rem">
      Last scan: <strong style="color:#1a1d2e">{scan_time}</strong>
      &nbsp; Auto-refresh: every 60s
      &nbsp; Long avg RSI: <strong style="color:#065c38">{avg_rsi_long:.1f}</strong>
      &nbsp; Short avg RSI: <strong style="color:#c0142e">{avg_rsi_short:.1f}</strong>
      &nbsp; SL Alerts: <strong style="color:#b07800">{len(sl_alerts)}</strong>
      ({sl_highs} HIGH / {sl_meds} MED)
    </div>""", unsafe_allow_html=True)

st.markdown(f"<style>{TABLE_CSS}</style>", unsafe_allow_html=True)

long_syms  = {s["sym"] for s in long_hits}
short_syms = {s["sym"] for s in short_hits}
sl_long    = sl_hunt_rows_for(long_syms,  sl_alerts)
sl_short   = sl_hunt_rows_for(short_syms, sl_alerts)

col_l, col_r = st.columns(2, gap="medium")

with col_l:
    st.markdown(f"""
    <div class="section-hdr" style="background:#d0f5e8;border:2px solid #0a7c4e">
      <span style="color:#065c38;font-weight:800;font-size:1rem">LONG SIGNALS</span>
      <div>
        <span class="stat-pill" style="background:#0a7c4e;color:#fff">{len(long_hits)} stocks</span>
        <span class="stat-pill" style="background:#c8f0e0;color:#065c38">Avg RSI {avg_rsi_long:.1f}</span>
      </div>
    </div>
    <div style="color:#3a3f5c;font-size:0.75rem;margin-bottom:6px">
      RSI(14) &gt;= 55 &nbsp; Price &gt; VWAP &nbsp; Price &gt;= EMA(33) of 9-min High
    </div>""", unsafe_allow_html=True)

    if long_hits:
        with st.expander(f"Show Long signals ({len(long_hits)} stocks)", expanded=True):
            long_rows, long_ema_hdr = signal_table_html(long_hits, True)
            st.markdown(f"""
            <table class="sig-table long-tbl">
              <thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th><th>RSI</th>
              <th>VWAP</th><th>{long_ema_hdr}</th><th>Vol</th>
              <th>C1</th><th>C2</th><th>C3</th></tr></thead>
              <tbody>{long_rows}</tbody>
            </table>""", unsafe_allow_html=True)
            rows_l = [
                {"Symbol": s["sym"], "Price": s["price"], "Chg%": s["chg"],
                 "RSI": s["rsi"], "VWAP": s["vwap"], "EMA_H": s["emah"],
                 "Volume": s["vol"], "C1_RSI": s["lc1"], "C2_VWAP": s["lc2"], "C3_EMA": s["lc3"]}
                for s in sorted(long_hits, key=lambda x: x["rsi"], reverse=True)
            ]
            st.download_button("Download Long CSV", pd.DataFrame(rows_l).to_csv(index=False),
                               file_name=f"long_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_long")
    else:
        st.markdown("""<div style="background:#fff3e0;border:1px solid #bf6000;border-radius:6px;
                       padding:10px 14px;color:#7a4000;font-size:0.85rem">
                       No Long signals this cycle.</div>""", unsafe_allow_html=True)

    with st.expander(f"SL Hunt - Long ({len(sl_long)} alerts)", expanded=len(sl_long) > 0):
        st.markdown(build_sl_table(sl_long), unsafe_allow_html=True)

with col_r:
    st.markdown(f"""
    <div class="section-hdr" style="background:#fde0e5;border:2px solid #c0142e">
      <span style="color:#8f0d20;font-weight:800;font-size:1rem">SHORT SIGNALS</span>
      <div>
        <span class="stat-pill" style="background:#c0142e;color:#fff">{len(short_hits)} stocks</span>
        <span class="stat-pill" style="background:#fcd4db;color:#8f0d20">Avg RSI {avg_rsi_short:.1f}</span>
      </div>
    </div>
    <div style="color:#3a3f5c;font-size:0.75rem;margin-bottom:6px">
      RSI(14) &lt;= 45 &nbsp; Price &lt; VWAP &nbsp; Price &lt;= EMA(33) of 9-min Low
    </div>""", unsafe_allow_html=True)

    if short_hits:
        with st.expander(f"Show Short signals ({len(short_hits)} stocks)", expanded=True):
            short_rows, short_ema_hdr = signal_table_html(short_hits, False)
            st.markdown(f"""
            <table class="sig-table short-tbl">
              <thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th><th>RSI</th>
              <th>VWAP</th><th>{short_ema_hdr}</th><th>Vol</th>
              <th>C1</th><th>C2</th><th>C3</th></tr></thead>
              <tbody>{short_rows}</tbody>
            </table>""", unsafe_allow_html=True)
            rows_s = [
                {"Symbol": s["sym"], "Price": s["price"], "Chg%": s["chg"],
                 "RSI": s["rsi"], "VWAP": s["vwap"], "EMA_L": s["emal"],
                 "Volume": s["vol"], "C1_RSI": s["sc1"], "C2_VWAP": s["sc2"], "C3_EMA": s["sc3"]}
                for s in sorted(short_hits, key=lambda x: x["rsi"])
            ]
            st.download_button("Download Short CSV", pd.DataFrame(rows_s).to_csv(index=False),
                               file_name=f"short_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_short")
    else:
        st.markdown("""<div style="background:#fff3e0;border:1px solid #bf6000;border-radius:6px;
                       padding:10px 14px;color:#7a4000;font-size:0.85rem">
                       No Short signals this cycle.</div>""", unsafe_allow_html=True)

    with st.expander(f"SL Hunt - Short ({len(sl_short)} alerts)", expanded=len(sl_short) > 0):
        st.markdown(build_sl_table(sl_short), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# AUTO-REFRESH  — browser-side only, NEVER time.sleep()
# ═══════════════════════════════════════════════════════════════
# BUG FIXED: The original used time.sleep(60) which BLOCKS the
# Streamlit server thread.  On Streamlit Cloud this makes the
# WebSocket time out and the URL becomes unreachable.
# Solution: inject an HTML meta-refresh tag.  The BROWSER reloads
# the page after 60 s; the server thread is never blocked.
if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True,
    )

st.markdown(f"""
<div class="footer">
  Nifty 50 Signal Scanner &nbsp; Data: yfinance NSE (.NS) &nbsp;
  RSI(14) VWAP EMA(33) 9-min OHLCV &nbsp; Last scan: {scan_time}
</div>
""", unsafe_allow_html=True)
