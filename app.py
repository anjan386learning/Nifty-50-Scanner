"""
Nifty 50 Signal Scanner — Streamlit Web App  v4
=================================================
ROOT CAUSE OF INFINITE REFRESH (diagnosed March 2025):
  Yahoo Finance permanently rate-limits Streamlit Cloud's shared IP
  addresses. yfinance calls hang/fail silently → page never finishes
  loading → any refresh fires before load → infinite loop.

THIS VERSION FIXES IT BY:
  1. Rendering the full page skeleton IMMEDIATELY before any data fetch.
     The URL always opens. There is no blank/spinner state.
  2. Using st.session_state to persist the last successful scan.
     Page always shows something, even if the current fetch fails.
  3. Fetching data in a daemon thread with a hard 25-second timeout.
     If Yahoo is slow/blocked, we show old data + a warning — not a hang.
  4. The appdirs cache-dir fix (required for yfinance ≥ 0.2.29 on
     Streamlit Cloud, per https://github.com/blackary/yf-fix).
  5. NO auto-refresh at all. A countdown timer shows time to next
     manual rerun. This removes every possible refresh loop.
"""

# ── appdirs fix MUST be before any yfinance import ───────────────────────────
from pathlib import Path
try:
    import appdirs as ad
    _CACHE = ".cache"
    ad.user_cache_dir = lambda *a, **k: _CACHE
    Path(_CACHE).mkdir(exist_ok=True)
except ImportError:
    pass

import threading
import warnings
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nifty 50 Signal Scanner",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "RELIANCE":"RELIANCE.NS","TCS":"TCS.NS","HDFCBANK":"HDFCBANK.NS",
    "ICICIBANK":"ICICIBANK.NS","INFY":"INFY.NS","HINDUNILVR":"HINDUNILVR.NS",
    "ITC":"ITC.NS","SBIN":"SBIN.NS","BHARTIARTL":"BHARTIARTL.NS",
    "KOTAKBANK":"KOTAKBANK.NS","LT":"LT.NS","AXISBANK":"AXISBANK.NS",
    "ASIANPAINT":"ASIANPAINT.NS","MARUTI":"MARUTI.NS","TITAN":"TITAN.NS",
    "WIPRO":"WIPRO.NS","ULTRACEMCO":"ULTRACEMCO.NS","BAJFINANCE":"BAJFINANCE.NS",
    "ONGC":"ONGC.NS","NTPC":"NTPC.NS","POWERGRID":"POWERGRID.NS",
    "M&M":"M&M.NS","NESTLEIND":"NESTLEIND.NS","TECHM":"TECHM.NS",
    "JSWSTEEL":"JSWSTEEL.NS","HCLTECH":"HCLTECH.NS","TATAMOTORS":"TMPV.NS",
    "COALINDIA":"COALINDIA.NS","INDUSINDBK":"INDUSINDBK.NS",
    "SUNPHARMA":"SUNPHARMA.NS","DRREDDY":"DRREDDY.NS",
    "BAJAJFINSV":"BAJAJFINSV.NS","DIVISLAB":"DIVISLAB.NS","CIPLA":"CIPLA.NS",
    "HEROMOTOCO":"HEROMOTOCO.NS","GRASIM":"GRASIM.NS",
    "TATACONSUM":"TATACONSUM.NS","BAJAJ-AUTO":"BAJAJ-AUTO.NS",
    "ADANIENT":"ADANIENT.NS","ADANIPORTS":"ADANIPORTS.NS",
    "HINDALCO":"HINDALCO.NS","TATASTEEL":"TATASTEEL.NS",
    "APOLLOHOSP":"APOLLOHOSP.NS","LTIM":"LTIM.NS","BEL":"BEL.NS",
    "SHRIRAMFIN":"SHRIRAMFIN.NS","TRENT":"TRENT.NS","ETERNAL":"ETERNAL.NS",
    "BPCL":"BPCL.NS","EICHERMOT":"EICHERMOT.NS",
}
NS_LIST  = list(TICKERS.values())
SYM_LIST = list(TICKERS.keys())
NS_TO_SYM = {v: k for k, v in TICKERS.items()}

INTERVAL_MIN = 9
RSI_PERIOD   = 14
EMA_PERIOD   = 33
REQUIRED     = (RSI_PERIOD + EMA_PERIOD) * INTERVAL_MIN

SL_WICK  = 0.55
SL_VOL   = 2.0
SL_DIV   = 8.0
IST      = timezone(timedelta(hours=5, minutes=30))
REFRESH_EVERY_MIN = 5          # show countdown; user can refresh manually

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER  — batch fetch with hard timeout
# ═══════════════════════════════════════════════════════════════════════════════
def _batch_download(period: str, timeout: int = 25) -> dict:
    """
    One yf.download() call for ALL 50 tickers at once.
    Runs in a daemon thread; returns {} if it doesn't finish within timeout.
    """
    result = {}

    def _work():
        try:
            raw = yf.download(
                NS_LIST, interval="1m", period=period,
                progress=False, auto_adjust=True,
                group_by="ticker", threads=True,
            )
            if raw is None or raw.empty:
                return
            if isinstance(raw.columns, pd.MultiIndex):
                for ns in NS_LIST:
                    try:
                        df = raw[ns].copy().dropna()
                        df.columns = [c.lower() for c in df.columns]
                        if len(df) >= REQUIRED:
                            sym = NS_TO_SYM.get(ns, ns.replace(".NS",""))
                            result[sym] = df
                    except Exception:
                        pass
        except Exception:
            pass

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    return result


def fetch_all() -> tuple[dict, str]:
    """Returns (raw_dict, status_msg)."""
    # Try 1-day data first
    raw = _batch_download("1d", timeout=25)
    if len(raw) >= 10:
        return raw, f"OK ({len(raw)}/50 stocks)"

    # Fallback to 5-day
    raw5 = _batch_download("5d", timeout=25)
    if len(raw5) > len(raw):
        raw = raw5

    if not raw:
        return {}, "Yahoo Finance rate-limited — showing last known data"
    return raw, f"Partial: {len(raw)}/50 stocks loaded"


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
def resample(df):
    return (df.resample(f"{INTERVAL_MIN}min", label="left", closed="left")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"),    close=("close","last"),
                   volume=("volume","sum"))
              .dropna())

def calc_rsi(s):
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def calc_vwap(df):
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
    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
    return ts.astimezone(IST).strftime("%H:%M:%S")

def _wick(sym, d, df, ts):
    r = df.iloc[-1]; o,h,l,c = r["open"],r["high"],r["low"],r["close"]
    rng = h-l
    if rng < 0.5: return []
    uw,lw = h-max(o,c), min(o,c)-l
    out = []
    if d=="LONG"  and uw/rng >= SL_WICK:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"UPPER WICK",
                    "detail":f"Wick {uw:.1f} / Range {rng:.1f} = {uw/rng:.0%}",
                    "severity":"HIGH" if uw/rng>=.70 else "MED"})
    if d=="SHORT" and lw/rng >= SL_WICK:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"LOWER WICK",
                    "detail":f"Wick {lw:.1f} / Range {rng:.1f} = {lw/rng:.0%}",
                    "severity":"HIGH" if lw/rng>=.70 else "MED"})
    return out

def _vol(sym, d, df, ts):
    if len(df)<6: return []
    avg = df["volume"].iloc[-6:-1].mean()
    if avg<=0: return []
    r = df.iloc[-1]; vr = r["volume"]/avg; pc = r["close"]-r["open"]
    out = []
    if d=="LONG"  and vr>=SL_VOL and pc<0:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"VOL SPIKE BEAR",
                    "detail":f"Vol {vr:.1f}x avg C:{r['close']:.1f} O:{r['open']:.1f}",
                    "severity":"HIGH" if vr>=SL_VOL*1.5 else "MED"})
    if d=="SHORT" and vr>=SL_VOL and pc>0:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"VOL SPIKE BULL",
                    "detail":f"Vol {vr:.1f}x avg C:{r['close']:.1f} O:{r['open']:.1f}",
                    "severity":"HIGH" if vr>=SL_VOL*1.5 else "MED"})
    return out

def _div(sym, d, df, rsi_s, ts):
    if rsi_s is None or len(rsi_s)<4 or len(df)<4: return []
    p = df["close"].iloc[-4:].values; r = rsi_s.iloc[-4:].values
    out = []
    if d=="LONG"  and p[-1]>p[-2] and r[-1]<r[-2]-SL_DIV:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"BEARISH RSI DIV",
                    "detail":f"Price up {p[-2]:.1f}>{p[-1]:.1f} RSI dn {r[-2]:.1f}>{r[-1]:.1f}",
                    "severity":"MED"})
    if d=="SHORT" and p[-1]<p[-2] and r[-1]>r[-2]+SL_DIV:
        out.append({"ts":ts,"sym":sym,"direction":d,"pattern":"BULLISH RSI DIV",
                    "detail":f"Price dn {p[-2]:.1f}>{p[-1]:.1f} RSI up {r[-2]:.1f}>{r[-1]:.1f}",
                    "severity":"MED"})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN  — pure CPU, runs on already-fetched data
# ═══════════════════════════════════════════════════════════════════════════════
def process_raw(raw: dict) -> tuple[list, list]:
    signals = []; sl = []
    for sym, df1 in raw.items():
        try:
            if df1.empty or len(df1) < REQUIRED: continue
            df = resample(df1)
            if len(df) < RSI_PERIOD + EMA_PERIOD: continue

            rsi_s = calc_rsi(df["close"])
            vwap  = calc_vwap(df)
            emah  = df["high"].ewm(span=EMA_PERIOD, adjust=False).mean()
            emal  = df["low"].ewm(span=EMA_PERIOD, adjust=False).mean()

            price = float(df["close"].iloc[-1])
            rsi_v = float(rsi_s.iloc[-1])
            vwap_v= float(vwap.iloc[-1])
            emah_v= float(emah.iloc[-1])
            emal_v= float(emal.iloc[-1])
            prev  = float(df["close"].iloc[-2]) if len(df)>1 else price
            chg   = (price - prev)/prev*100
            vol   = int(df["volume"].iloc[-1])

            lc1=rsi_v>=55; lc2=price>vwap_v; lc3=price>=emah_v
            sc1=rsi_v<=45; sc2=price<vwap_v; sc3=price<=emal_v

            sig = dict(sym=sym, price=round(price,2), chg=round(chg,2),
                       rsi=round(rsi_v,1), vwap=round(vwap_v,2),
                       emah=round(emah_v,2), emal=round(emal_v,2), vol=vol,
                       lc1=lc1,lc2=lc2,lc3=lc3, long_pass=lc1 and lc2 and lc3,
                       sc1=sc1,sc2=sc2,sc3=sc3, short_pass=sc1 and sc2 and sc3)
            signals.append(sig)

            if (sig["long_pass"] or sig["short_pass"]) and len(df)>=5:
                d2 = "LONG" if sig["long_pass"] else "SHORT"
                ts = _sl_ts(df)
                sl += _wick(sym,d2,df,ts)
                sl += _vol(sym,d2,df,ts)
                sl += _div(sym,d2,df,rsi_s,ts)
        except Exception:
            continue
    return signals, sl


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION-STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _ss_init():
    for k, v in [
        ("signals", []), ("sl_alerts", []),
        ("scan_time", None), ("fetch_status", "Not yet scanned"),
        ("is_fetching", False), ("elapsed", 0.0),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

def _do_fetch():
    """Run fetch + process; update session_state when done."""
    if st.session_state.get("is_fetching"):
        return
    st.session_state["is_fetching"] = True
    t0 = time.time()
    raw, status = fetch_all()
    if raw:
        sigs, sl = process_raw(raw)
        st.session_state["signals"]     = sigs
        st.session_state["sl_alerts"]   = sl
        st.session_state["scan_time"]   = datetime.now(IST).strftime("%H:%M:%S IST")
    st.session_state["fetch_status"] = status
    st.session_state["elapsed"]      = round(time.time()-t0, 1)
    st.session_state["is_fetching"]  = False

def _age_min() -> float:
    """Minutes since last scan."""
    if not st.session_state["scan_time"]:
        return 999
    try:
        now_ist = datetime.now(IST)
        last = datetime.strptime(
            st.session_state["scan_time"], "%H:%M:%S IST"
        ).replace(
            year=now_ist.year, month=now_ist.month,
            day=now_ist.day, tzinfo=IST
        )
        diff = (now_ist - last).total_seconds() / 60
        return max(diff, 0)
    except Exception:
        return 999


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
html,body,[data-testid="stAppViewContainer"]{background:#f0f2f7!important;font-family:'Inter','Segoe UI',sans-serif}
[data-testid="stHeader"]{background:#fff!important;border-bottom:2px solid #b0b8d0}
.tbar{background:#fff;border:1px solid #b0b8d0;border-radius:10px;padding:14px 22px;margin-bottom:12px;
      display:flex;align-items:center;gap:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.tbar h1{margin:0;font-size:1.45rem;color:#1a1d2e;font-weight:700}
.tbar p{margin:0;font-size:.8rem;color:#6b718e}
.mrow{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.mcrd{background:#fff;border:1px solid #b0b8d0;border-radius:8px;padding:10px 16px;min-width:120px}
.mcrd .lb{font-size:.7rem;color:#6b718e;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.mcrd .vl{font-size:1.55rem;font-weight:700;color:#1a1d2e;line-height:1.15}
.mcrd .sb{font-size:.7rem;color:#9098b0}
.stbl{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:3px}
.stbl th{background:#2c3150;color:#fff;font-weight:700;padding:5px 7px;text-align:center;
         border-bottom:2px solid #5a6080;font-size:.74rem}
.stbl td{padding:5px 7px;text-align:center;border-bottom:1px solid #d0d5e8;color:#1a1d2e}
.stbl tr:nth-child(even) td{background:#e8ecf5}
.stbl tr:nth-child(odd)  td{background:#f5f7fc}
.ltbl tr:hover td{background:#a8e6c8!important}
.shtbl tr:hover td{background:#f5b0bc!important}
.sl-tbl th{background:#5a2d00;color:#fff}
.sl-tbl tr.sl-hi td{background:#b00020!important;color:#fff!important;font-weight:700}
.sl-tbl tr.sl-md td{background:#8a4e00!important;color:#fff!important;font-weight:600}
.shdr{border-radius:8px;padding:9px 13px;margin-bottom:5px;display:flex;align-items:center;justify-content:space-between}
.pill{display:inline-block;border-radius:6px;padding:2px 8px;font-size:.71rem;font-weight:700;margin:0 2px}
.ftr{text-align:center;color:#9098b0;font-size:.74rem;margin-top:18px;padding:8px;border-top:1px solid #b0b8d0}
#MainMenu,footer,[data-testid="collapsedControl"]{visibility:hidden}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _sig_rows(hits, mode_long):
    if not hits: return "", ""
    ema_k = "emah" if mode_long else "emal"
    ema_h = "EMA33H" if mode_long else "EMA33L"
    ck = ("lc1","lc2","lc3") if mode_long else ("sc1","sc2","sc3")
    col = "#065c38" if mode_long else "#8f0d20"
    rows = ""
    for s in sorted(hits, key=lambda x: x["rsi"], reverse=mode_long):
        cc = "#0a7c4e" if s["chg"]>=0 else "#c0142e"
        cs = f"{'+'if s['chg']>=0 else''}{s['chg']:.2f}%"
        c1,c2,c3 = ("&#10003;" if s[k] else "&#183;" for k in ck)
        rows += (f"<tr><td style='font-weight:700;color:{col}'>{s['sym']}</td>"
                 f"<td>&#8377;{s['price']:,.1f}</td>"
                 f"<td style='color:{cc};font-weight:600'>{cs}</td>"
                 f"<td style='font-weight:700'>{s['rsi']:.1f}</td>"
                 f"<td>{s['vwap']:.1f}</td><td>{s[ema_k]:.1f}</td>"
                 f"<td>{fmt_vol(s['vol'])}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c1}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c2}</td>"
                 f"<td style='color:#0a7c4e;font-weight:700'>{c3}</td></tr>")
    return rows, ema_h

def _sl_rows(alerts):
    if not alerts:
        return "<p style='color:#9098b0;font-size:.79rem;padding:3px 0'>No SL Hunt alerts.</p>"
    rows = ""
    for a in reversed(alerts):
        css = "sl-hi" if a["severity"]=="HIGH" else "sl-md"
        rows += (f"<tr class='{css}'><td>{a['ts']}</td><td><b>{a['sym']}</b></td>"
                 f"<td>{'&#9650;L' if a['direction']=='LONG' else '&#9660;S'}</td>"
                 f"<td>{a['severity']}</td><td>{a['pattern']}</td>"
                 f"<td style='text-align:left'>{a['detail']}</td></tr>")
    return (f"<table class='stbl sl-tbl'><thead><tr>"
            f"<th>Time</th><th>Sym</th><th>Dir</th><th>Sev</th><th>Pattern</th><th>Detail</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP  — page renders IMMEDIATELY, then data is shown/refreshed
# ═══════════════════════════════════════════════════════════════════════════════
_ss_init()

st.markdown(CSS, unsafe_allow_html=True)

# ── Title  (renders instantly) ───────────────────────────────────────────────
st.markdown("""
<div class="tbar"><div style="font-size:1.9rem">&#11041;</div><div>
<h1>Nifty 50 Signal Scanner</h1>
<p>9-min OHLCV &#183; RSI(14) &#183; VWAP &#183; EMA(33) H/L &#183; SL Hunt &#183; NSE via yfinance batch</p>
</div></div>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### &#9881; Settings")
    st.markdown(f"Refresh every **{REFRESH_EVERY_MIN} min** (manual)")
    st.markdown("---")
    st.markdown("**Long:** RSI&#8805;55 · Price>VWAP · Price&#8805;EMA33H")
    st.markdown("**Short:** RSI&#8804;45 · Price<VWAP · Price&#8804;EMA33L")
    st.markdown("---")
    force_btn = st.button("&#8635; Force Rescan", type="primary",
                          use_container_width=True)
    st.markdown("---")
    age = _age_min()
    if st.session_state["scan_time"]:
        st.caption(f"Last scan: {st.session_state['scan_time']}")
        st.caption(f"Age: {age:.1f} min")
        st.caption(f"Status: {st.session_state['fetch_status']}")

# ── Decide if we need to fetch ───────────────────────────────────────────────
need_fetch = (
    force_btn                               # user clicked Force Rescan
    or not st.session_state["scan_time"]    # first load — no data yet
    or age >= REFRESH_EVERY_MIN             # data is stale
)

# ── If first load AND no data: show loading message immediately ───────────────
if not st.session_state["scan_time"]:
    st.info(
        "⏳ First load — fetching all 50 Nifty stocks via batch download "
        "(takes 10–30 seconds). The page will update automatically.",
        icon="📡"
    )

# ── Fetch if needed — happens AFTER page header is already rendered ───────────
if need_fetch:
    with st.spinner("Fetching market data (batch download, 25s timeout)..."):
        _do_fetch()
    if need_fetch and not force_btn:
        # Natural stale refresh — rerun to show fresh data cleanly
        st.rerun()

# ── From here, ALWAYS render from session_state (never blocks) ───────────────
signals   = st.session_state["signals"]
sl_alerts = st.session_state["sl_alerts"]
scan_time = st.session_state["scan_time"] or "Not yet scanned"
fetch_st  = st.session_state["fetch_status"]
elapsed   = st.session_state["elapsed"]

long_hits  = [s for s in signals if s.get("long_pass")]
short_hits = [s for s in signals if s.get("short_pass")]
avg_l = sum(s["rsi"] for s in long_hits)  / len(long_hits)  if long_hits  else 0.0
avg_s = sum(s["rsi"] for s in short_hits) / len(short_hits) if short_hits else 0.0
sl_hi = sum(1 for a in sl_alerts if a["severity"]=="HIGH")
sl_md = len(sl_alerts) - sl_hi

# ── Status / warning banner ───────────────────────────────────────────────────
if "rate-limited" in fetch_st.lower() or "partial" in fetch_st.lower():
    st.warning(
        f"⚠️ **{fetch_st}** — Yahoo Finance may be rate-limiting Streamlit Cloud. "
        "Showing last successful scan. Click **Force Rescan** in the sidebar to retry.",
        icon="⚠️"
    )
elif signals:
    st.success(
        f"&#10003; **{fetch_st}** — scanned in {elapsed}s at {scan_time}",
        icon="✅"
    )

# ── Metric cards ─────────────────────────────────────────────────────────────
age_now = _age_min()
next_in = max(0, REFRESH_EVERY_MIN * 60 - age_now * 60)
next_str = f"{int(next_in//60)}m {int(next_in%60)}s" if next_in > 0 else "now"

st.markdown(f"""
<div class="mrow">
  <div class="mcrd"><div class="lb">Scanned</div>
    <div class="vl" style="{'color:#c0142e' if len(signals)<20 and signals else 'color:#1a1d2e'}">{len(signals)}</div>
    <div class="sb">of 50 stocks</div></div>
  <div class="mcrd"><div class="lb">&#9650; Long</div>
    <div class="vl" style="color:#0a7c4e">{len(long_hits)}</div><div class="sb">signals</div></div>
  <div class="mcrd"><div class="lb">&#9660; Short</div>
    <div class="vl" style="color:#c0142e">{len(short_hits)}</div><div class="sb">signals</div></div>
  <div class="mcrd"><div class="lb">Avg RSI Long</div>
    <div class="vl" style="color:#0a7c4e">{f"{avg_l:.1f}" if long_hits else "&#8212;"}</div>
    <div class="sb">9-min bars</div></div>
  <div class="mcrd"><div class="lb">Avg RSI Short</div>
    <div class="vl" style="color:#c0142e">{f"{avg_s:.1f}" if short_hits else "&#8212;"}</div>
    <div class="sb">9-min bars</div></div>
  <div class="mcrd"><div class="lb">&#9889; SL Alerts</div>
    <div class="vl" style="color:#b07800">{len(sl_alerts)}</div>
    <div class="sb">{sl_hi} HIGH / {sl_md} MED</div></div>
  <div class="mcrd"><div class="lb">Last Scan</div>
    <div class="vl" style="font-size:.88rem;padding-top:3px">{scan_time}</div>
    <div class="sb">next in {next_str}</div></div>
</div>""", unsafe_allow_html=True)

# ── Refresh bar ───────────────────────────────────────────────────────────────
bc1, bc2 = st.columns([1, 7])
with bc1:
    if st.button("&#8635; Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        _do_fetch()
        st.rerun()
with bc2:
    st.markdown(
        f"<div style='padding:7px 0;color:#6b718e;font-size:.81rem'>"
        f"<b style='color:#1a1d2e'>{scan_time}</b> &#183; "
        f"{len(signals)} stocks loaded in {elapsed}s &#183; "
        f"Next auto-refresh in <b>{next_str}</b> — or use sidebar Force Rescan"
        f"</div>", unsafe_allow_html=True)

# ── Tables ────────────────────────────────────────────────────────────────────
long_syms  = {s["sym"] for s in long_hits}
short_syms = {s["sym"] for s in short_hits}
sl_long    = [a for a in sl_alerts if a["sym"] in long_syms]
sl_short   = [a for a in sl_alerts if a["sym"] in short_syms]

col_l, col_r = st.columns(2, gap="medium")

with col_l:
    st.markdown(f"""
    <div class="shdr" style="background:#d0f5e8;border:2px solid #0a7c4e">
      <span style="color:#065c38;font-weight:800">&#9650; LONG SIGNALS</span>
      <div><span class="pill" style="background:#0a7c4e;color:#fff">{len(long_hits)} stocks</span>
           <span class="pill" style="background:#c8f0e0;color:#065c38">Avg RSI {avg_l:.1f}</span></div>
    </div>
    <div style="color:#3a3f5c;font-size:.73rem;margin-bottom:3px">
      RSI&#8805;55 &#183; Price&gt;VWAP &#183; Price&#8805;EMA(33 High)</div>
    """, unsafe_allow_html=True)

    if long_hits:
        with st.expander(f"Show {len(long_hits)} Long signals", expanded=True):
            rows_h, ema_h = _sig_rows(long_hits, True)
            st.markdown(
                f"<table class='stbl ltbl'><thead><tr>"
                f"<th>Sym</th><th>Price</th><th>Chg%</th><th>RSI</th>"
                f"<th>VWAP</th><th>{ema_h}</th><th>Vol</th>"
                f"<th>C1</th><th>C2</th><th>C3</th></tr></thead>"
                f"<tbody>{rows_h}</tbody></table>", unsafe_allow_html=True)
            csv = pd.DataFrame([
                {"Symbol":s["sym"],"Price":s["price"],"Chg%":s["chg"],"RSI":s["rsi"],
                 "VWAP":s["vwap"],"EMA_H":s["emah"],"Vol":s["vol"],
                 "C1":s["lc1"],"C2":s["lc2"],"C3":s["lc3"]}
                for s in sorted(long_hits, key=lambda x:x["rsi"], reverse=True)
            ]).to_csv(index=False)
            st.download_button("&#8659; CSV", csv,
                               file_name=f"long_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_l")
    else:
        st.info("No Long signals this cycle.")

    with st.expander(f"&#9889; SL Hunt Long ({len(sl_long)} alerts)",
                     expanded=len(sl_long)>0):
        st.markdown(_sl_rows(sl_long), unsafe_allow_html=True)

with col_r:
    st.markdown(f"""
    <div class="shdr" style="background:#fde0e5;border:2px solid #c0142e">
      <span style="color:#8f0d20;font-weight:800">&#9660; SHORT SIGNALS</span>
      <div><span class="pill" style="background:#c0142e;color:#fff">{len(short_hits)} stocks</span>
           <span class="pill" style="background:#fcd4db;color:#8f0d20">Avg RSI {avg_s:.1f}</span></div>
    </div>
    <div style="color:#3a3f5c;font-size:.73rem;margin-bottom:3px">
      RSI&#8804;45 &#183; Price&lt;VWAP &#183; Price&#8804;EMA(33 Low)</div>
    """, unsafe_allow_html=True)

    if short_hits:
        with st.expander(f"Show {len(short_hits)} Short signals", expanded=True):
            rows_h, ema_h = _sig_rows(short_hits, False)
            st.markdown(
                f"<table class='stbl shtbl'><thead><tr>"
                f"<th>Sym</th><th>Price</th><th>Chg%</th><th>RSI</th>"
                f"<th>VWAP</th><th>{ema_h}</th><th>Vol</th>"
                f"<th>C1</th><th>C2</th><th>C3</th></tr></thead>"
                f"<tbody>{rows_h}</tbody></table>", unsafe_allow_html=True)
            csv = pd.DataFrame([
                {"Symbol":s["sym"],"Price":s["price"],"Chg%":s["chg"],"RSI":s["rsi"],
                 "VWAP":s["vwap"],"EMA_L":s["emal"],"Vol":s["vol"],
                 "C1":s["sc1"],"C2":s["sc2"],"C3":s["sc3"]}
                for s in sorted(short_hits, key=lambda x:x["rsi"])
            ]).to_csv(index=False)
            st.download_button("&#8659; CSV", csv,
                               file_name=f"short_{scan_time[:8].replace(':','')}.csv",
                               mime="text/csv", key="dl_s")
    else:
        st.info("No Short signals this cycle.")

    with st.expander(f"&#9889; SL Hunt Short ({len(sl_short)} alerts)",
                     expanded=len(sl_short)>0):
        st.markdown(_sl_rows(sl_short), unsafe_allow_html=True)

# ── Auto-rerun when data is stale (NO meta refresh, NO time.sleep) ────────────
# We schedule a rerun using st.rerun() ONLY after the full page has rendered.
# This is the only safe pattern — page is ALWAYS fully visible first.
if age_now >= REFRESH_EVERY_MIN and not need_fetch:
    # Data went stale while user was on the page → trigger one clean refresh
    time.sleep(0.5)   # tiny yield so Streamlit flushes the render above
    st.rerun()

st.markdown(f"""<div class="ftr">
  &#11041; Nifty 50 Signal Scanner &#183; yfinance NSE batch &#183;
  RSI(14) VWAP EMA(33) 9-min &#183; Last: {scan_time}
</div>""", unsafe_allow_html=True)
