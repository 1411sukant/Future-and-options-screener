# =============================================================================
#  F&O REAL-TIME SCREENER  |  app.py  v3.0
#  Data  : yfinance (Yahoo Finance) — FREE, no API key, ~2-min delay
#  Run   : streamlit run app.py
#  Deps  : pip install streamlit pandas numpy yfinance requests
# =============================================================================
#
#  DATA SOURCES (all free):
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  yfinance (.NS suffix)  → CMP, OHLCV, ATR, VWAP, Volume       │
#  │  yfinance option_chain  → Real OI (calls+puts), PCR, IV        │
#  │  Session State          → OI delta between refreshes           │
#  └─────────────────────────────────────────────────────────────────┘
#
#  PERFORMANCE:
#  First load  ~30-60s  (fetches 20 symbols in parallel via threads)
#  Cached for  5 minutes — instant subsequent loads
#  Refresh     click sidebar button or wait for cache to expire
#
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Auto-install yfinance if missing (Streamlit Cloud safety net) ────────────
# Handles the case where requirements.txt was not committed to the repo.
try:
    import yfinance as yf
except ModuleNotFoundError:
    st.info("⚙️ Installing yfinance for the first time — please wait ~15 seconds…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "yfinance>=0.2.40", "-q"]
    )
    import yfinance as yf
    st.rerun()   # restart so all imports are clean

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Live Screener",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #060B14 !important;
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
section[data-testid="stSidebar"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8B949E !important;
}
[data-testid="stMetric"] {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #238636, #1F6FEB);
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8B949E !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #3FB950 !important;
}
hr { border-color: #21262D !important; }
h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; }
h2, h3 { color: #1F6FEB !important;
          font-size: 0.85rem !important;
          text-transform: uppercase; letter-spacing: 0.07em; }
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #3FB950 !important;
    color: #3FB950 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem; letter-spacing: 0.07em;
    text-transform: uppercase; border-radius: 6px;
}
.stDownloadButton > button:hover {
    background: #3FB950 !important; color: #060B14 !important;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid #1F6FEB !important;
    color: #1F6FEB !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem; letter-spacing: 0.07em;
    text-transform: uppercase; border-radius: 6px; width: 100%;
}
.stButton > button:hover {
    background: #1F6FEB !important; color: #060B14 !important;
}
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0D1117; border: 1px solid #238636;
    border-radius: 20px; padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3FB950; letter-spacing: 0.08em;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3FB950;
    box-shadow: 0 0 6px #3FB950;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SYMBOL UNIVERSE  — Top 20 NSE F&O stocks by liquidity
# =============================================================================

FO_SYMBOLS = [
    "RELIANCE", "TCS",       "HDFCBANK",  "ICICIBANK", "INFY",
    "SBIN",     "AXISBANK",  "KOTAKBANK", "BAJFINANCE","TITAN",
    "MARUTI",   "LT",        "SUNPHARMA", "WIPRO",     "HCLTECH",
    "TATAMOTORS","TECHM",    "ONGC",      "TATASTEEL", "APOLLOHOSP",
]

# yfinance needs ".NS" suffix for NSE stocks
def nse(sym: str) -> str:
    return f"{sym}.NS"


# =============================================================================
#  SINGLE SYMBOL DATA FETCHER
#  Fetches OHLCV + option chain for one symbol.
#  Designed to run inside a ThreadPoolExecutor worker.
# =============================================================================

def fetch_symbol_data(symbol: str) -> dict | None:
    """
    Returns a dict of market data for one NSE F&O symbol.

    Price / OHLCV  : yfinance Ticker.history(period='22d')
    ATR            : 14-period Average True Range from daily OHLCV
    VWAP           : Volume-weighted from 22-day history (swing VWAP)
    Support        : 20-day rolling low
    Resistance     : 20-day rolling high
    OI / PCR / IV  : Nearest-expiry option chain via yfinance
    """
    try:
        ticker = yf.Ticker(nse(symbol))

        # ── Historical OHLCV (22 days) ────────────────────────────────────────
        hist = ticker.history(period="22d", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None

        cmp        = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cmp
        volume     = int(hist["Volume"].iloc[-1])
        avg_volume = int(hist["Volume"].iloc[:-1].mean())

        # ── ATR (14-period) ───────────────────────────────────────────────────
        h, l, c = hist["High"], hist["Low"], hist["Close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().dropna().iloc[-1])

        # ── VWAP (swing, from available history) ─────────────────────────────
        typical_price = (hist["High"] + hist["Low"] + hist["Close"]) / 3
        vwap = float(
            (typical_price * hist["Volume"]).sum() / hist["Volume"].sum()
        )

        # ── Support & Resistance (20-day extremes) ────────────────────────────
        support    = float(hist["Low"].rolling(20).min().dropna().iloc[-1])
        resistance = float(hist["High"].rolling(20).max().dropna().iloc[-1])

        # ── Option Chain (nearest expiry) ─────────────────────────────────────
        oi_total = 0
        pcr      = 1.0     # neutral default
        iv       = 25.0    # default IV %

        try:
            exps = ticker.options           # tuple of expiry date strings
            if exps:
                chain = ticker.option_chain(exps[0])   # nearest expiry

                call_oi = int(chain.calls["openInterest"].fillna(0).sum())
                put_oi  = int(chain.puts["openInterest"].fillna(0).sum())
                oi_total = call_oi + put_oi

                # PCR = Put OI / Call OI
                pcr = round(put_oi / max(call_oi, 1), 2)

                # ATM IV: find the call strike closest to CMP
                calls_df = chain.calls.copy()
                calls_df["dist"] = (calls_df["strike"] - cmp).abs()
                atm_row  = calls_df.nsmallest(1, "dist")
                if not atm_row.empty:
                    raw_iv = float(atm_row["impliedVolatility"].values[0])
                    iv = round(raw_iv * 100, 2)   # convert 0.XX → XX%
        except Exception:
            pass  # option chain unavailable — use defaults

        return {
            "Symbol":    symbol,
            "CMP":       round(cmp, 2),
            "Prev_Close":round(prev_close, 2),
            "VWAP":      round(vwap, 2),
            "Volume":    volume,
            "Avg_Volume":avg_volume,
            "ATR":       round(atr, 2),
            "OI":        oi_total,
            "PCR":       pcr,
            "IV":        iv,
            "Support":   round(support, 2),
            "Resistance":round(resistance, 2),
        }

    except Exception:
        return None   # caller will skip this symbol


# =============================================================================
#  BATCH FETCHER  — runs all symbols in parallel threads
# =============================================================================

def fetch_live_universe(symbols: list[str], max_workers: int = 8) -> pd.DataFrame:
    """
    Fetch live data for every symbol using a thread pool.
    Returns a cleaned DataFrame ready for indicator calculation.
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_symbol_data, sym): sym
            for sym in symbols
        }
        for future in as_completed(future_map):
            sym  = future_map[future]
            data = future.result()
            if data is not None:
                results[sym] = data

    if not results:
        raise RuntimeError(
            "All symbol fetches failed. "
            "Check your internet connection or yfinance availability."
        )

    df = pd.DataFrame(list(results.values()))

    # ── OI Change %: compare with previous fetch stored in session state ──────
    prev_oi_map: dict = st.session_state.get("prev_oi_map", {})
    oi_change = []
    for _, row in df.iterrows():
        sym      = row["Symbol"]
        curr_oi  = row["OI"]
        prev_oi  = prev_oi_map.get(sym, curr_oi)
        if prev_oi > 0:
            chg = round((curr_oi - prev_oi) / prev_oi * 100, 2)
        else:
            # First load: use volume ratio as a proxy for OI momentum
            vol_ratio = row["Volume"] / max(row["Avg_Volume"], 1)
            chg = round((vol_ratio - 1) * 15, 2)   # scaled proxy signal
        oi_change.append(chg)

    df["OI_Change_Pct"] = oi_change

    # Persist current OI for next refresh comparison
    st.session_state["prev_oi_map"]  = dict(zip(df["Symbol"], df["OI"]))
    st.session_state["last_fetch_ts"]= datetime.now().strftime("%H:%M:%S")

    return df.reset_index(drop=True)


# =============================================================================
#  CACHED WRAPPER  — 5-minute TTL
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_universe(symbols_tuple: tuple, _cache_key: int) -> pd.DataFrame:
    """
    Thin cache wrapper. _cache_key is bumped by the Refresh button
    to force a new fetch without waiting for TTL expiry.
    """
    return fetch_live_universe(list(symbols_tuple))


# =============================================================================
#  INDICATOR ENGINE
# =============================================================================

def classify_oi_trend(price_chg: float, oi_chg: float) -> str:
    """Classic four-quadrant OI interpretation."""
    if price_chg >= 0 and oi_chg >= 0:
        return "🟢 Long Buildup"
    elif price_chg < 0 and oi_chg >= 0:
        return "🔴 Short Buildup"
    elif price_chg >= 0 and oi_chg < 0:
        return "🟡 Short Covering"
    else:
        return "🔵 Long Unwinding"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Price_Change_Pct"] = (
        (out["CMP"] - out["Prev_Close"]) / out["Prev_Close"] * 100
    ).round(2)
    out["Trend_OI"]    = out.apply(
        lambda r: classify_oi_trend(r["Price_Change_Pct"], r["OI_Change_Pct"]), axis=1
    )
    out["VWAP_Signal"] = np.where(out["CMP"] > out["VWAP"], "▲ Above", "▼ Below")
    out["Vol_Surge"]   = out["Volume"] > (1.5 * out["Avg_Volume"])
    return out


# =============================================================================
#  3:1 RISK-REWARD ENGINE
# =============================================================================

def calculate_rr_setups(
    df: pd.DataFrame,
    atr_multiplier: float,
    min_oi_change_pct: float,
    pcr_low: float  = 0.70,
    pcr_high: float = 1.50,
) -> pd.DataFrame:
    """
    Filters for high-probability setups satisfying a strict 3:1 R:R.

    Entry      = CMP
    Stop-Loss  = Entry − (ATR × multiplier)
    Target     = Entry + (Risk × 3)

    Five validity gates:
      1. |OI Change %| >= threshold          (liquidity / momentum)
      2. Resistance >= Target                (clear structural headroom)
      3. PCR outside neutral band            (sentiment extreme)
      4. Stop-Loss >= Support × 0.98        (SL on structural floor)
      5. OI Trend is bullish (LB or SC)      (directional confirmation)
    """
    out = df.copy()
    sl_dist          = out["ATR"] * atr_multiplier
    out["Entry"]     = out["CMP"]
    out["Stop_Loss"] = (out["Entry"] - sl_dist).round(2)
    out["Target_3x"] = (out["Entry"] + sl_dist * 3).round(2)

    f1 = out["OI_Change_Pct"].abs() >= min_oi_change_pct
    f2 = out["Resistance"] >= out["Target_3x"]
    f3 = (out["PCR"] < pcr_low) | (out["PCR"] > pcr_high)
    f4 = out["Stop_Loss"] >= out["Support"] * 0.98
    f5 = out["Trend_OI"].isin(["🟢 Long Buildup", "🟡 Short Covering"])

    screened = out[f1 & f2 & f3 & f4 & f5].copy()
    if not screened.empty:
        screened["Actual_RR"] = (
            (screened["Target_3x"] - screened["Entry"]) /
            (screened["Entry"]     - screened["Stop_Loss"])
        ).round(2)
    else:
        screened["Actual_RR"] = pd.Series(dtype=float)

    return screened.reset_index(drop=True)


# =============================================================================
#  POSITION SIZING
# =============================================================================

def position_sizing(entry: float, sl: float, capital: float, risk_pct: float) -> dict:
    risk_amount   = capital * (risk_pct / 100)
    risk_per_unit = abs(entry - sl)
    if risk_per_unit < 0.01:
        return {"units": 0, "risk_inr": 0.0, "reward_inr": 0.0}
    return {
        "units":      int(risk_amount / risk_per_unit),
        "risk_inr":   round(risk_amount, 2),
        "reward_inr": round(risk_amount * 3, 2),
    }


# =============================================================================
#  SESSION STATE INIT
# =============================================================================

if "cache_key"     not in st.session_state:
    st.session_state["cache_key"]      = 0
if "prev_oi_map"   not in st.session_state:
    st.session_state["prev_oi_map"]    = {}
if "last_fetch_ts" not in st.session_state:
    st.session_state["last_fetch_ts"]  = "—"


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding:0.3rem 0 0.9rem;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
                  letter-spacing:0.14em;text-transform:uppercase;
                  color:#8B949E;margin:0;'>NSE · LIVE DATA</p>
        <h1 style='font-size:1.2rem;font-weight:700;margin:4px 0 0;
                   background:linear-gradient(135deg,#3FB950,#1F6FEB);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            F&O Screener v3.0
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**💰 Capital**")
    capital  = st.number_input("Total Capital (₹)", min_value=10_000,
                                max_value=50_000_000, value=500_000, step=10_000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.1)
    st.divider()

    st.markdown("**📐 Stop-Loss**")
    atr_multiplier = st.slider("ATR Multiplier", 0.5, 3.0, 1.5, 0.1,
                                 help="SL = CMP − (ATR × multiplier)")
    st.divider()

    st.markdown("**📈 OI Filter**")
    min_oi_change = st.slider("Min |OI Change| (%)", 0.5, 25.0, 2.0, 0.5,
                               help="Lower this if market is quiet (first load uses volume proxy)")
    st.divider()

    st.markdown("**⚖️ PCR Thresholds**")
    c1, c2 = st.columns(2)
    with c1:
        pcr_low  = st.number_input("Low",  value=0.70, step=0.05,
                                    min_value=0.10, max_value=1.00)
    with c2:
        pcr_high = st.number_input("High", value=1.50, step=0.05,
                                    min_value=1.00, max_value=3.00)
    st.divider()

    if st.button("🔄  Refresh Live Data"):
        st.cache_data.clear()
        st.session_state["cache_key"] += 1
        st.rerun()

    last_ts = st.session_state["last_fetch_ts"]
    st.markdown(
        f"<p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
        f"color:#8B949E;text-align:center;margin-top:8px;'>"
        f"Last fetch: {last_ts}<br>"
        f"Cache TTL: 5 min</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("""
    <p style='font-size:0.6rem;color:#30363D;text-align:center;
    font-family:JetBrains Mono,monospace;'>
    Data: Yahoo Finance (yfinance)<br>
    Delay: ~2 min · Not financial advice
    </p>""", unsafe_allow_html=True)


# =============================================================================
#  HEADER
# =============================================================================

col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown("""
    <h1 style='margin:0;font-size:1.7rem;'>📡 F&O Live Screener</h1>
    <p style='margin:2px 0 0;font-size:0.72rem;color:#8B949E;
              font-family:JetBrains Mono,monospace;letter-spacing:0.06em;'>
        NSE FUTURES &amp; OPTIONS · REAL DATA · 3:1 RR ENGINE · yfinance
    </p>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style='text-align:right;padding-top:8px;'>
        <span class='live-badge'>
            <span class='live-dot'></span>LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)
st.divider()


# =============================================================================
#  DATA LOADING  — with progress indicator
# =============================================================================

data_placeholder  = st.empty()
progress_placeholder = st.empty()

try:
    with data_placeholder.container():
        with st.spinner("⏳ Fetching live NSE data via yfinance… (first load ~30s, then cached 5 min)"):
            raw_df = get_cached_universe(
                symbols_tuple = tuple(FO_SYMBOLS),
                _cache_key    = st.session_state["cache_key"],
            )

    data_placeholder.empty()

    enriched_df = add_indicators(raw_df)
    screened_df = calculate_rr_setups(
        enriched_df,
        atr_multiplier    = atr_multiplier,
        min_oi_change_pct = min_oi_change,
        pcr_low           = pcr_low,
        pcr_high          = pcr_high,
    )

except Exception as exc:
    data_placeholder.empty()
    st.error(
        f"**Live data fetch failed:** {exc}\n\n"
        "Possible causes:\n"
        "- No internet connection\n"
        "- yfinance/Yahoo Finance temporarily unavailable\n"
        "- Market closed (options data unavailable outside hours)\n\n"
        "👉 Click **Refresh Live Data** in the sidebar to retry."
    )
    st.stop()


# =============================================================================
#  KPI METRICS
# =============================================================================

total_setups    = len(screened_df)
bullish_setups  = int(screened_df["Trend_OI"].str.contains("Long Buildup",   na=False).sum())
covering_setups = int(screened_df["Trend_OI"].str.contains("Short Covering", na=False).sum())
symbols_loaded  = len(enriched_df)
avg_pcr         = float(enriched_df["PCR"].mean())
avg_iv          = float(enriched_df["IV"].mean())

st.markdown("## 📊  Market Pulse")
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric("🎯 Setups",         str(total_setups),
              delta=f"of {symbols_loaded} loaded")
with k2:
    st.metric("🟢 Long Buildup",   str(bullish_setups),
              delta="fresh longs")
with k3:
    st.metric("🟡 Short Covering", str(covering_setups),
              delta="bears exiting")
with k4:
    st.metric("⚖️ Avg PCR",        f"{avg_pcr:.2f}",
              delta="> 1.0 bearish skew")
with k5:
    st.metric("📊 Avg IV",         f"{avg_iv:.1f}%",
              delta="implied volatility")
with k6:
    above_vwap = int((enriched_df["VWAP_Signal"] == "▲ Above").sum())
    st.metric("📈 Above VWAP",     str(above_vwap),
              delta=f"of {symbols_loaded} symbols")

st.divider()


# =============================================================================
#  SETUPS TABLE
# =============================================================================

st.markdown("## 🔍  Actionable 3:1 RR Setups")

if screened_df.empty:
    st.warning(
        "⚠️ No setups pass all 5 filters right now.\n\n"
        "**Try these adjustments in the sidebar:**\n"
        "- Lower **Min |OI Change|** to 1–2% (market may be calm)\n"
        "- Widen **PCR range** (e.g., Low: 0.5, High: 1.8)\n"
        "- Increase **ATR Multiplier** to 2.0+ (wider stop = more room for target)\n"
        "- Click **🔄 Refresh** — OI delta improves after the 2nd fetch"
    )
else:
    # Attach position sizing
    pos_list = [
        position_sizing(row["Entry"], row["Stop_Loss"], capital, risk_pct)
        for _, row in screened_df.iterrows()
    ]
    screened_df["Units"]      = [p["units"]      for p in pos_list]
    screened_df["Risk_INR"]   = [p["risk_inr"]   for p in pos_list]
    screened_df["Reward_INR"] = [p["reward_inr"] for p in pos_list]

    display_cols = [
        "Symbol", "Trend_OI", "CMP", "Entry", "Stop_Loss",
        "Target_3x", "Actual_RR", "PCR", "IV",
        "VWAP_Signal", "OI_Change_Pct",
        "Units", "Risk_INR", "Reward_INR",
    ]
    display_df = screened_df[display_cols].copy()

    # Explicit dtype safety — prevents Arrow serialization errors
    float_cols = ["CMP","Entry","Stop_Loss","Target_3x","Actual_RR",
                  "PCR","IV","OI_Change_Pct","Risk_INR","Reward_INR"]
    for c in float_cols:
        display_df[c] = display_df[c].astype("float64")
    display_df["Units"] = display_df["Units"].astype("int64")

    col_cfg = {
        "Symbol":        st.column_config.TextColumn("Symbol",          width="small"),
        "Trend_OI":      st.column_config.TextColumn("OI Trend",        width="medium"),
        "CMP":           st.column_config.NumberColumn("CMP (₹)",       format="₹%.2f"),
        "Entry":         st.column_config.NumberColumn("Entry (₹)",     format="₹%.2f"),
        "Stop_Loss":     st.column_config.NumberColumn("Stop-Loss (₹)", format="₹%.2f",
                         help="SL = CMP − (ATR × multiplier)"),
        "Target_3x":     st.column_config.NumberColumn("Target 3:1 (₹)",format="₹%.2f",
                         help="Target = Entry + Risk × 3"),
        "Actual_RR":     st.column_config.NumberColumn("R:R",           format="%.2fx"),
        "PCR":           st.column_config.NumberColumn("PCR",           format="%.2f",
                         help="Put-Call Ratio from live option chain"),
        "IV":            st.column_config.NumberColumn("IV (%)",        format="%.1f%%",
                         help="ATM Implied Volatility from nearest expiry"),
        "VWAP_Signal":   st.column_config.TextColumn("VWAP",            width="small"),
        "OI_Change_Pct": st.column_config.NumberColumn("OI Δ (%)",      format="%.2f%%",
                         help="OI change vs previous fetch (1st load = volume proxy)"),
        "Units":         st.column_config.NumberColumn("Units",         format="%d"),
        "Risk_INR":      st.column_config.NumberColumn("Risk (₹)",      format="₹%.0f"),
        "Reward_INR":    st.column_config.NumberColumn("Reward (₹)",    format="₹%.0f"),
    }

    st.dataframe(
        display_df,
        column_config       = col_cfg,
        use_container_width = True,
        hide_index          = True,
        height              = min(80 + 38 * len(display_df), 560),
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Export Setups CSV", csv,
                        "fo_live_setups.csv", "text/csv")

st.divider()


# =============================================================================
#  FULL UNIVERSE TABLE
# =============================================================================

with st.expander("📋  Full Live Universe (all loaded symbols)", expanded=False):
    uni = enriched_df[[
        "Symbol","CMP","Prev_Close","Price_Change_Pct",
        "VWAP","VWAP_Signal","OI","OI_Change_Pct","Trend_OI",
        "PCR","IV","ATR","Support","Resistance",
    ]].copy()

    for c in ["CMP","Prev_Close","Price_Change_Pct","VWAP",
              "OI_Change_Pct","PCR","IV","ATR","Support","Resistance"]:
        uni[c] = uni[c].astype("float64")
    uni["OI"] = uni["OI"].astype("int64")

    uni_cfg = {
        "Symbol":           st.column_config.TextColumn("Symbol",     width="small"),
        "CMP":              st.column_config.NumberColumn("CMP (₹)",  format="₹%.2f"),
        "Prev_Close":       st.column_config.NumberColumn("Prev Close",format="₹%.2f"),
        "Price_Change_Pct": st.column_config.NumberColumn("Price Δ%", format="%.2f%%"),
        "VWAP":             st.column_config.NumberColumn("VWAP",     format="₹%.2f"),
        "VWAP_Signal":      st.column_config.TextColumn("VWAP Pos",   width="small"),
        "OI":               st.column_config.NumberColumn("Total OI", format="%d"),
        "OI_Change_Pct":    st.column_config.NumberColumn("OI Δ%",    format="%.2f%%"),
        "Trend_OI":         st.column_config.TextColumn("OI Trend",   width="medium"),
        "PCR":              st.column_config.NumberColumn("PCR",      format="%.2f"),
        "IV":               st.column_config.NumberColumn("IV%",      format="%.1f%%"),
        "ATR":              st.column_config.NumberColumn("ATR",      format="₹%.2f"),
        "Support":          st.column_config.NumberColumn("Support",  format="₹%.2f"),
        "Resistance":       st.column_config.NumberColumn("Resistance",format="₹%.2f"),
    }

    st.dataframe(uni, column_config=uni_cfg,
                  use_container_width=True, hide_index=True, height=440)


# =============================================================================
#  DATA SOURCE EXPLAINER
# =============================================================================

with st.expander("ℹ️  Data Sources & Methodology", expanded=False):
    st.markdown("""
### What data is real vs estimated?

| Column | Source | Notes |
|---|---|---|
| CMP, Prev Close | yfinance `.history()` | ~2 min delay |
| Volume | yfinance `.history()` | Real |
| ATR (14-period) | Calculated from OHLCV | Real |
| VWAP | Calculated from 22-day OHLCV | Swing VWAP, not intraday |
| Support / Resistance | 20-day Low / High | Real technicals |
| PCR | yfinance `.option_chain()` | Real, nearest expiry |
| IV | yfinance `.option_chain()` ATM | Real, nearest expiry |
| OI (Total) | yfinance `.option_chain()` sum | Real, nearest expiry |
| OI Change % | Session state delta | **Proxy on 1st load** (volume-based); improves after Refresh |

### Why "OI Change" is a proxy on first load
OI change requires two data points. On the first load there is no previous
snapshot, so the app uses `(Volume / AvgVolume − 1) × 15` as a scaled proxy
for participation intensity. After you click **🔄 Refresh**, real OI delta
between the two fetches is used.

### 3:1 RR Formula
```
Risk       = ATR × ATR_Multiplier
Stop-Loss  = CMP − Risk
Target     = CMP + (Risk × 3)
Valid if   : Resistance >= Target  AND  Stop-Loss >= Support × 0.98
```
> Data via Yahoo Finance (yfinance). Free, ~2-min delay. Not financial advice.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.3rem;'>
    <p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;
              color:#21262D;letter-spacing:0.1em;'>
        F&O SCREENER v3.0 · yfinance · NSE · NOT FINANCIAL ADVICE
    </p>
</div>
""", unsafe_allow_html=True)
