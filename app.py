"""
Nifty 50 Signal Scanner — Streamlit Web App
Long / Short signals · RSI · VWAP · EMA(33) · SL Hunt Detector

KEY FIX vs previous versions:
  All 50 tickers downloaded in ONE single yf.download() batch call.
  This avoids Yahoo Finance rate-limiting on Streamlit Cloud IPs,
  which was causing each individual request to fail/hang and making
  the page never finish loading (triggering infinite refresh loop).
"""

import warnings
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh   # pip install streamlit-autorefresh

warnings.filterwarnings("ignore")

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 Signal Scanner",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
NIFTY50_TICKERS: dict = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "INFY":       "INFY.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",
    "SBIN":       "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "LT":         "LT.NS",
    "AXISBANK":   "AXISBANK.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI":     "MARUTI.NS",
    "TITAN":      "TITAN.NS",
    "WIPRO":      "WIPRO.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ONGC":       "ONGC.NS",
    "NTPC":       "NTPC.NS",
    "POWERGRID":  "POWERGRID.NS",
    "M&M":        "M&M.NS",
    "NESTLEIND":  "NESTLEIND.NS",
    "TECHM":      "TECHM.NS",
    "JSWSTEEL":   "JSWSTEEL.NS",
    "HCLTECH":    "HCLTECH.NS",
    "TATAMOTORS": "TMPV.NS",
    "COALINDIA":  "COALINDIA.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "DRREDDY":    "DRREDDY.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "DIVISLAB":   "DIVISLAB.NS",
    "CIPLA":      "CIPLA.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "GRASIM":     "GRASIM.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "ADANIENT":   "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "HINDALCO":   "HINDALCO.NS",
    "TATASTEEL":  "TATASTEEL.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "LTIM":       "LTIM.NS",
    "BEL":        "BEL.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "TRENT":      "TRENT.NS",
    "ETERNAL":    "ETERNAL.NS",
    "BPCL":       "BPCL.NS",
    "EICHERMOT":  "EICHERMOT.NS",
}
NIFTY50      = list(NIFTY50_TICKERS.keys())
NS_TICKERS   = list(NIFTY50_TICKERS.values())          # list of .NS strings

INTERVAL_MIN = 9
RSI_PERIOD   = 14
EMA_PERIOD   = 33

SL_WICK_RATIO     = 0.55
SL_VOL_SPIKE_MULT = 2.0
SL_RSI_DIVERGE    = 8.0

IST = timezone(timedelta(hours=5, minutes=30))

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH DATA FETCH  ← THE CRITICAL FIX
#
# Instead of 50 individual yf.download() calls (which hit Yahoo Finance
# rate limits on Streamlit Cloud IPs), we make ONE single batch call for
# all 50 tickers at once.  Yahoo Finance handles batch requests differently
# and is far less likely to rate-limit them.
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min — enough for one trading session
def fetch_all_batch() -> dict:
    """
    Single yf.download() call for all 50 NS tickers, 1-min interval.
    Returns {sym: df_1min} dict.  Skips any ticker that returns empty data.

    Why batch?  On Streamlit Cloud, Yahoo Finance rate-limits requests from
    shared IP ranges.  50 individual calls = 50 opportunities to get blocked.
    1 batch call = 1 opportunity, and Yahoo handles batch downloads via a
    different (less restricted) endpoint.
    """
    raw = {}
    try:
        # Try today's data first (period="1d")
        data = yf.download(
            NS_TICKERS,
            interval="1m",
            period="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
        # If less than half succeeded, fall back to 5d period
        if data.empty:
            raise ValueError("empty")

        # Parse out per-ticker DataFrames
        # With group_by="ticker", columns are MultiIndex: (ticker, OHLCV)
        ns_to_sym = {v: k for k, v in NIFTY50_TICKERS.items()}
        REQUIRED  = (RSI_PERIOD + EMA_PERIOD) * INTERVAL_MIN

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in NS_TICKERS:
                try:
                    df = data[ticker].copy()
                    df.columns = [c.lower() for c in df.columns]
                    df = df.dropna(subset=["close", "volume"])
                    if len(df) >= REQUIRED:
                        sym = ns_to_sym.get(ticker, ticker.replace(".NS",""))
                        raw[sym] = df
                except Exception:
                    pass
        else:
            # Single ticker returned flat — shouldn't happen in batch, handle anyway
            pass

        # If fewer than 10 symbols succeeded, broaden to 5d
        if len(raw) < 10:
            raise ValueError("too few")

    except Exception:
        # Fallback: try 5d period
        try:
            data = yf.download(
                NS_TICKERS,
                interval="1m",
                period="5d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            ns_to_sym = {v: k for k, v in NIFTY50_TICKERS.items()}
            REQUIRED  = (RSI_PERIOD + EMA_PERIOD) * INTERVAL_MIN

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in NS_TICKERS:
                    if ticker in raw:
                        continue   # already have good data
                    try:
                        df = data[ticker].copy()
                        df.columns = [c.lower() for c in df.columns]
                        df = df.dropna(subset=["close", "volume"])
                        if len(df) >= REQUIRED:
                            sym = ns_to_sym.get(ticker, ticker.replace(".NS",""))
                            raw[sym] = df
                    except Exception:
                        pass
        except Exception:
            pass

    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
def resample(df: pd.DataFrame) -> pd.DataFrame:
    return (df.resample(f"{INTERVAL_MIN}min", label="left", closed="left")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),    close=("close","last"),
                   volume=("volume","sum"))
              .dropna())


def calc_rsi(s: pd.Series) -> pd.Series:
    d  = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def fmt_vol(v):
    if v >= 1_000_000: return f"{v/1e6:.1f}M"
    if v >= 1_000:     return f"{v/1e3:.0f}K"
    return str(v)


# ═══════════════════════════════════════════════════════════════════════════════
# SL HUNT
# ═══════════════════════════════════════════════════════════════════════════════
def _sl_ts(df):
    ts = df.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.astimezone(IST).strftime("%H:%M:%S")


def _check_wick(sym, direction, df, ts):
    alerts = []
    r = df.iloc[-1]
    o, h, l, c = r["open"], r["high"], r["low"], r["close"]
    rng = h - l
    if rng < 0.5:
        return alerts
    uw = h - max(o, c)
    lw = min(o, c) - l
    if direction == "LONG" and uw / rng >= SL_WICK_RATIO:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "UPPER WICK",
                       "detail": f"Wick {uw:.1f} / Range {rng:.1f} = {uw/rng:.0%} | H:{h:.1f} L:{l:.1f} C:{c:.1f}",
                       "severity": "HIGH" if uw/rng >= 0.70 else "MED"})
    if direction == "SHORT" and lw / rng >= SL_WICK_RATIO:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "LOWER WICK",
                       "detail": f"Wick {lw:.1f} / Range {rng:.1f} = {lw/rng:.0%} | H:{h:.1f} L:{l:.1f} C:{c:.1f}",
                       "severity": "HIGH" if lw/rng >= 0.70 else "MED"})
    return alerts


def _check_vol(sym, direction, df, ts):
    alerts = []
    if len(df) < 6:
        return alerts
    avg = df["volume"].iloc[-6:-1].mean()
    if avg <= 0:
        return alerts
    r = df.iloc[-1]
    vr = r["volume"] / avg
    pc = r["close"] - r["open"]
    if direction == "LONG" and vr >= SL_VOL_SPIKE_MULT and pc < 0:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "VOL SPIKE BEAR",
                       "detail": f"Vol {vr:.1f}x avg | C:{r['close']:.1f} O:{r['open']:.1f}",
                       "severity": "HIGH" if vr >= SL_VOL_SPIKE_MULT*1.5 else "MED"})
    if direction == "SHORT" and vr >= SL_VOL_SPIKE_MULT and pc > 0:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "VOL SPIKE BULL",
                       "detail": f"Vol {vr:.1f}x avg | C:{r['close']:.1f} O:{r['open']:.1f}",
                       "severity": "HIGH" if vr >= SL_VOL_SPIKE_MULT*1.5 else "MED"})
    return alerts


def _check_div(sym, direction, df, rsi_s, ts):
    alerts = []
    if rsi_s is None or len(rsi_s) < 4 or len(df) < 4:
        return alerts
    p = df["close"].iloc[-4:].values
    r = rsi_s.iloc[-4:].values
    if direction == "LONG" and p[-1] > p[-2] and r[-1] < r[-2] - SL_RSI_DIVERGE:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "BEARISH RSI DIV",
                       "detail": f"Price up {p[-2]:.1f}>{p[-1]:.1f} | RSI down {r[-2]:.1f}>{r[-1]:.1f}",
                       "severity": "MED"})
    if direction == "SHORT" and p[-1] < p[-2] and r[-1] > r[-2] + SL_RSI_DIVERGE:
        alerts.append({"ts": ts, "sym": sym, "direction": direction,
                       "pattern": "BULLISH RSI DIV",
                       "detail": f"Price down {p[-2]:.1f}>{p[-1]:.1f} | RSI up {r[-2]:.1f}>{r[-1]:.1f}",
                       "severity": "MED"})
    return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN  — runs on the already-fetched batch data (no more HTTP calls)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=295, show_spinner=False)
def run_scan() -> tuple:
    """
    Calls fetch_all_batch() once, then processes all 50 symbols purely in-memory.
    Zero additional network requests.  Runs in < 2 seconds after data is fetched.
    """
    raw_data  = fetch_all_batch()
    signals   = []
    sl_alerts = []

    for sym, df1 in raw_data.items():
        try:
            if df1.empty or len(df1) < RSI_PERIOD * INTERVAL_MIN:
                continue
            df = resample(df1)
            if len(df) < RSI_PERIOD + EMA_PERIOD:
                continue

            rsi_s  = calc_rsi(df["close"])
            vwap_s = calc_vwap(df)
            emah_s = df["high"].ewm(span=EMA_PERIOD, adjust=False).mean()
            emal_s = df["low"].ewm(span=EMA_PERIOD, adjust=False).mean()

            price  = float(df["close"].iloc[-1])
            rsi_v  = float(rsi_s.iloc[-1])
            vwap_v = float(vwap_s.iloc[-1])
            emah_v = float(emah_s.iloc[-1])
            emal_v = float(emal_s.iloc[-1])
            prev   = float(df["close"].iloc[-2]) if len(df) > 1 else price
            chg    = (price - prev) / prev * 100
            vol    = int(df["volume"].iloc[-1])

            lc1 = rsi_v >= 55; lc2 = price > vwap_v; lc3 = price >= emah_v
            sc1 = rsi_v <= 45; sc2 = price < vwap_v; sc3 = price <= emal_v

            sig = dict(
                sym=sym, price=round(price,2), chg=round(chg,2),
                rsi=round(rsi_v,1), vwap=round(vwap_v,2),
                emah=round(emah_v,2), emal=round(emal_v,2), vol=vol,
                lc1=lc1, lc2=lc2, lc3=lc3, long_pass=lc1 and lc2 and lc3,
                sc1=sc1, sc2=sc2, sc3=sc3, short_pass=sc1 and sc2 and sc3,
            )
            signals.append(sig)

            # SL Hunt inline — no extra data needed
            if (sig["long_pass"] or sig["short_pass"]) and len(df) >= 5:
                direction = "LONG" if sig["long_pass"] else "SHORT"
                ts = _sl_ts(df)
                sl_alerts += _check_wick(sym, direction, df, ts)
                sl_alerts += _check_vol(sym, direction, df, ts)
                sl_alerts += _check_div(sym, direction, df, rsi_s, ts)

        except Exception:
            continue

    return signals, sl_alerts


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#f0f2f7!important;font-family:'Inter','Segoe UI',sans-serif}
[data-testid="stHeader"]{background:#fff!important;border-bottom:2px solid #b0b8d0}
.title-bar{background:#fff;border:1px solid #b0b8d0;border-radius:10px;padding:16px 24px;margin-bottom:14px;display:flex;align-items:center;gap:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.title-bar h1{margin:0;font-size:1.5rem;color:#1a1d2e;font-weight:700}
.title-bar p{margin:0;font-size:.82rem;color:#6b718e}
.mrow{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.mcrd{background:#fff;border:1px solid #b0b8d0;border-radius:8px;padding:10px 18px;min-width:128px;box-shadow:0 1px 4px rgba(0,0,0,.04)}
.mcrd .lbl{font-size:.72rem;color:#6b718e;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.mcrd .val{font-size:1.6rem;font-weight:700;color:#1a1d2e;line-height:1.2}
.mcrd .sub{font-size:.72rem;color:#9098b0}
.sig-table{width:100%;border-collapse:collapse;font-size:.81rem;margin-top:4px}
.sig-table th{background:#2c3150;color:#fff;font-weight:700;padding:6px 8px;text-align:center;border-bottom:2px solid #5a6080;font-size:.76rem}
.sig-table td{padding:5px 8px;text-align:center;border-bottom:1px solid #d0d5e8;color:#1a1d2e}
.sig-table tr:nth-child(even) td{background:#e8ecf5}
.sig-table tr:nth-child(odd) td{background:#f5f7fc}
.long-tbl tr:hover td{background:#a8e6c8!important}
.short-tbl tr:hover td{background:#f5b0bc!important}
.sl-tbl th{background:#5a2d00;color:#fff}
.sl-tbl tr.sl-hi td{background:#b00020!important;color:#fff!important;font-weight:700}
.sl-tbl tr.sl-md td{background:#8a4e00!important;color:#fff!important;font-weight:600}
.shdr{border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between}
.pill{display:inline-block;border-radius:6px;padding:2px 9px;font-size:.73rem;font-weight:700;margin:0 3px}
.footer{text-align:center;color:#9098b0;font-size:.76rem;margin-top:20px;padding:10px;border-top:1px solid #b0b8d0}
#MainMenu,footer,[data-testid="collapsedControl"]{visibility:hidden}
</style>""", unsafe_allow_html=True)

TABLE_CSS = """
.sig-table{width:100%;border-collapse:collapse;font-size:.81rem;margin-top:4px}
.sig-table th{background:#2c3150;color:#fff;font-weight:700;padding:6px 8px;text-align:center;border-bottom:2px solid #5a6080;font-size:.76rem}
.sig-table td{padding:5px 8px;text-align:center;border-bottom:1px solid #d0d5e8;color:#1a1d2e}
.sig-table tr:nth-child(even) td{background:#e8ecf5}
.sig-table tr:nth-child(odd) td{background:#f5f7fc}
.long-tbl tr:hover td{background:#a8e6c8!important}
.short-tbl tr:hover td{background:#f5b0bc!important}
.sl-tbl th{background:#5a2d00;color:#fff}
.sl-tbl tr.sl-hi td{background:#b00020!important;color:#fff!important;font-weight:700}
.sl-tbl tr.sl-md td{background:#8a4e00!important;color:#fff!important;font-weight:600}
.shdr{border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between}
.pill{display:inline-block;border-radius:6px;padding:2px 9px;font-size:.73rem;font-weight:700;margin:0 3px}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def sig_table_html(hits, mode_long):
    if not hits:
        return "", ""
    ema_k = "emah" if mode_long else "emal"
    ema_h = "EMA33H" if mode_long else "EMA33L"
    ck    = ("lc1","lc2","lc3") if mode_long else ("sc1","sc2","sc3")
    col   = "#065c38" if mode_long else "#8f0d20"
    rows  = ""
    for s in sorted(hits, key=lambda x: x["rsi"], reverse=mode_long):
        cc = "#0a7c4e" if s["chg"] >= 0 else "#c0142e"
        cs = f"{'+'if s['chg']>=0 else''}{s['chg']:.2f}%"
        c1,c2,c3 = ("✓" if s[k] else "·" for k in ck)
        rows += (f"<tr>"
                 f"<td style='font-weight:700;color:{col}'>{s['sym']}</td>"
                 f"<td>&#8377;{s['price']:,.1f}</td>"
                 f"<td style='color:{cc};font-weight:600'>{cs}</td>"
                 f"<td style='font-weight:700'>{s['rsi']:.1f}</td>"
                 f"<td>{s['vwap']:.1f}</td>"
                 f"<td>{s[ema_k]:.1f}</td>"
                 f"<td>{fmt_vol(s['vol'])}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c1}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c2}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c3}</td>"
                 f"</tr>")
    return rows, ema_h


def sl_table_html(alerts):
    if not alerts:
        return "<p style='color:#9098b0;font-size:.81rem;padding:4px 0'>No SL Hunt alerts.</p>"
    rows = ""
    for a in reversed(alerts):
        css = "sl-hi" if a["severity"] == "HIGH" else "sl-md"
        rows += (f"<tr class='{css}'>"
                 f"<td>{a['ts']}</td><td><b>{a['sym']}</b></td>"
                 f"<td>{'&#9650;LONG' if a['direction']=='LONG' else '&#9660;SHORT'}</td>"
                 f"<td>{a['severity']}</td><td>{a['pattern']}</td>"
                 f"<td style='text-align:left'>{a['detail']}</td></tr>")
    return (f"<table class='sig-table sl-tbl'>"
            f"<thead><tr><th>Time</th><th>Symbol</th><th>Dir</th>"
            f"<th>Sev</th><th>Pattern</th><th>Detail</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Auto-refresh using streamlit-autorefresh (proper non-blocking refresh) ──
# This fires AFTER the page finishes rendering — unlike meta refresh which
# counts from when the HTML is first sent (before data loads).
refresh_count = st_autorefresh(interval=5 * 60 * 1000, limit=None, key="scan_refresh")

st.markdown("""
<div class="title-bar">
  <div style="font-size:2rem">&#11041;</div>
  <div>
    <h1>Nifty 50 Signal Scanner</h1>
    <p>9-min OHLCV &#183; RSI(14) &#183; Session VWAP &#183; EMA(33) High/Low
       &#183; SL Hunt &#183; NSE via yfinance batch</p>
  </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### &#9881; Settings")
    st.info("Auto-refreshes every **5 minutes**", icon="🔄")
    st.markdown("---")
    st.markdown("**Long:** RSI&#8805;55 · Price>VWAP · Price&#8805;EMA33H")
    st.markdown("**Short:** RSI&#8804;45 · Price<VWAP · Price&#8804;EMA33L")
    st.markdown("---")
    if st.button("&#8635; Force Rescan", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption(f"Refresh #{refresh_count}")

# ── Run scan ──
t0 = time.time()
with st.spinner("Fetching all 50 Nifty stocks (batch download)..."):
    signals, sl_alerts = run_scan()
elapsed = time.time() - t0

ist       = timezone(timedelta(hours=5, minutes=30))
scan_time = datetime.now(ist).strftime("%H:%M:%S IST")

long_hits  = [s for s in signals if s.get("long_pass")]
short_hits = [s for s in signals if s.get("short_pass")]
avg_l = sum(s["rsi"] for s in long_hits)  / len(long_hits)  if long_hits  else 0.0
avg_s = sum(s["rsi"] for s in short_hits) / len(short_hits) if short_hits else 0.0
sl_hi = sum(1 for a in sl_alerts if a["severity"] == "HIGH")
sl_md = len(sl_alerts) - sl_hi

# Data quality warning
if len(signals) < 20:
    st.warning(
        f"⚠️ Only {len(signals)}/50 stocks returned data. "
        "Yahoo Finance may be rate-limiting Streamlit Cloud. "
        "Click **Force Rescan** to retry, or wait a few minutes.",
        icon="⚠️"
    )

st.markdown(f"""
<div class="mrow">
  <div class="mcrd"><div class="lbl">Scanned</div>
    <div class="val" style="{'color:#c0142e' if len(signals)<20 else 'color:#1a1d2e'}">{len(signals)}</div>
    <div class="sub">of 50 stocks</div></div>
  <div class="mcrd"><div class="lbl">&#9650; Long</div>
    <div class="val" style="color:#0a7c4e">{len(long_hits)}</div>
    <div class="sub">signals</div></div>
  <div class="mcrd"><div class="lbl">&#9660; Short</div>
    <div class="val" style="color:#c0142e">{len(short_hits)}</div>
    <div class="sub">signals</div></div>
  <div class="mcrd"><div class="lbl">Avg RSI Long</div>
    <div class="val" style="color:#0a7c4e">{f"{avg_l:.1f}" if long_hits else "&#8212;"}</div>
    <div class="sub">9-min</div></div>
  <div class="mcrd"><div class="lbl">Avg RSI Short</div>
    <div class="val" style="color:#c0142e">{f"{avg_s:.1f}" if short_hits else "&#8212;"}</div>
    <div class="sub">9-min</div></div>
  <div class="mcrd"><div class="lbl">&#9889; SL Alerts</div>
    <div class="val" style="color:#b07800">{len(sl_alerts)}</div>
    <div class="sub">{sl_hi} HIGH / {sl_md} MED</div></div>
  <div class="mcrd"><div class="lbl">Last Scan</div>
    <div class="val" style="font-size:.9rem;padding-top:4px">{scan_time}</div>
    <div class="sub">in {elapsed:.1f}s</div></div>
</div>""", unsafe_allow_html=True)

# Refresh bar
c1, c2 = st.columns([1, 7])
with c1:
    if st.button("&#8635; Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with c2:
    st.markdown(f"""<div style="padding:8px 0;color:#6b718e;font-size:.83rem">
      <b style="color:#1a1d2e">{scan_time}</b> &#183;
      Auto-refresh every 5 min &#183;
      {len(signals)} stocks loaded in <b>{elapsed:.1f}s</b> (batch download)</div>""",
      unsafe_allow_html=True)

st.markdown(f"<style>{TABLE_CSS}</style>", unsafe_allow_html=True)

long_syms  = {s["sym"] for s in long_hits}
short_syms = {s["sym"] for s in short_hits}
sl_long    = [a for a in sl_alerts if a["sym"] in long_syms]
sl_short   = [a for a in sl_alerts if a["sym"] in short_syms]

col_l, col_r = st.columns(2, gap="medium")

# ── LONG ──
with col_l:
    st.markdown(f"""
    <div class="shdr" style="background:#d0f5e8;border:2px solid #0a7c4e">
      <span style="color:#065c38;font-weight:800">&#9650; LONG SIGNALS</span>
      <div>
        <span class="pill" style="background:#0a7c4e;color:#fff">{len(long_hits)} stocks</span>
        <span class="pill" style="background:#c8f0e0;color:#065c38">Avg RSI {avg_l:.1f}</span>
      </div>
    </div>
    <div style="color:#3a3f5c;font-size:.74rem;margin-bottom:4px">
      RSI&#8805;55 &#183; Price&gt;VWAP &#183; Price&#8805;EMA(33) of 9-min High
    </div>""", unsafe_allow_html=True)

    if long_hits:
        with st.expander(f"Show {len(long_hits)} Long signals", expanded=True):
            rows_html, ema_hdr = sig_table_html(long_hits, True)
            st.markdown(
                f"<table class='sig-table long-tbl'>"
                f"<thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th>"
                f"<th>RSI</th><th>VWAP</th><th>{ema_hdr}</th><th>Vol</th>"
                f"<th>C1</th><th>C2</th><th>C3</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True)
            csv_l = pd.DataFrame([
                {"Symbol":s["sym"],"Price":s["price"],"Chg%":s["chg"],"RSI":s["rsi"],
                 "VWAP":s["vwap"],"EMA_H":s["emah"],"Vol":s["vol"],
                 "C1":s["lc1"],"C2":s["lc2"],"C3":s["lc3"]}
                for s in sorted(long_hits, key=lambda x: x["rsi"], reverse=True)
            ]).to_csv(index=False)
            st.download_button("&#8659; Long CSV", csv_l,
                               file_name=f"long_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_l")
    else:
        st.info("No Long signals this cycle.")

    with st.expander(f"&#9889; SL Hunt — Long ({len(sl_long)} alerts)",
                     expanded=len(sl_long) > 0):
        st.markdown(sl_table_html(sl_long), unsafe_allow_html=True)

# ── SHORT ──
with col_r:
    st.markdown(f"""
    <div class="shdr" style="background:#fde0e5;border:2px solid #c0142e">
      <span style="color:#8f0d20;font-weight:800">&#9660; SHORT SIGNALS</span>
      <div>
        <span class="pill" style="background:#c0142e;color:#fff">{len(short_hits)} stocks</span>
        <span class="pill" style="background:#fcd4db;color:#8f0d20">Avg RSI {avg_s:.1f}</span>
      </div>
    </div>
    <div style="color:#3a3f5c;font-size:.74rem;margin-bottom:4px">
      RSI&#8804;45 &#183; Price&lt;VWAP &#183; Price&#8804;EMA(33) of 9-min Low
    </div>""", unsafe_allow_html=True)

    if short_hits:
        with st.expander(f"Show {len(short_hits)} Short signals", expanded=True):
            rows_html, ema_hdr = sig_table_html(short_hits, False)
            st.markdown(
                f"<table class='sig-table short-tbl'>"
                f"<thead><tr><th>Symbol</th><th>Price</th><th>Chg%</th>"
                f"<th>RSI</th><th>VWAP</th><th>{ema_hdr}</th><th>Vol</th>"
                f"<th>C1</th><th>C2</th><th>C3</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True)
            csv_s = pd.DataFrame([
                {"Symbol":s["sym"],"Price":s["price"],"Chg%":s["chg"],"RSI":s["rsi"],
                 "VWAP":s["vwap"],"EMA_L":s["emal"],"Vol":s["vol"],
                 "C1":s["sc1"],"C2":s["sc2"],"C3":s["sc3"]}
                for s in sorted(short_hits, key=lambda x: x["rsi"])
            ]).to_csv(index=False)
            st.download_button("&#8659; Short CSV", csv_s,
                               file_name=f"short_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_s")
    else:
        st.info("No Short signals this cycle.")

    with st.expander(f"&#9889; SL Hunt — Short ({len(sl_short)} alerts)",
                     expanded=len(sl_short) > 0):
        st.markdown(sl_table_html(sl_short), unsafe_allow_html=True)


st.markdown(f"""<div class="footer">
  &#11041; Nifty 50 Signal Scanner &#183;
  yfinance NSE batch &#183; RSI(14) VWAP EMA(33) 9-min OHLCV &#183;
  Last scan: {scan_time}
</div>""", unsafe_allow_html=True)
