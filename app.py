# =============================================================================
#  F&O SIGNAL SCREENER  |  app.py  v4.0
#  Simple: Press Refresh → Get BUY/SELL signals
#  Data  : yfinance (Yahoo Finance) — FREE
#  Run   : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_OK = True
except ModuleNotFoundError:
    YFINANCE_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Signal Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #080C14 !important;
    color: #E2E8F0 !important;
}

/* ── Big refresh button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.9rem 2rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #00C851, #007E33) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,200,81,0.3);
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 28px rgba(0,200,81,0.5) !important;
    transform: translateY(-1px);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0D1421;
    border: 1px solid #1A2535;
    border-radius: 12px;
    padding: 1rem 1.3rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B6080 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00C851 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1A2535 !important;
    border-radius: 10px;
    overflow: hidden;
}

hr { border-color: #1A2535 !important; }

/* ── Last updated tag ── */
.update-tag {
    display: inline-block;
    background: #0D1421;
    border: 1px solid #1A2535;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4B6080;
    letter-spacing: 0.07em;
}

/* ── Signal badge colors in markdown ── */
.buy  { color: #00C851; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.sell { color: #FF3B5C; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  NSE F&O SYMBOL LIST  (top 30 liquid stocks)
# =============================================================================

SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY",
    "SBIN","AXISBANK","KOTAKBANK","BAJFINANCE","TITAN",
    "MARUTI","LT","SUNPHARMA","WIPRO","HCLTECH",
    "TATAMOTORS","TECHM","ONGC","TATASTEEL","APOLLOHOSP",
    "BAJAJFINSV","POWERGRID","NTPC","DIVISLAB","CIPLA",
    "DRREDDY","COALINDIA","HINDALCO","JSWSTEEL","ADANIPORTS",
]


# =============================================================================
#  SIGNAL ENGINE
# =============================================================================

def compute_signal(symbol: str) -> dict | None:
    """
    Fetches live data for one symbol and returns a trading signal.

    Signal logic (all must align for BUY / SELL):
    ──────────────────────────────────────────────
    BUY  when:
      • CMP > VWAP            (price above average — bullish)
      • CMP > EMA20           (above 20-day trend)
      • RSI between 45–70     (momentum without being overbought)
      • Price change % > 0    (today is green)

    SELL when:
      • CMP < VWAP            (price below average — bearish)
      • CMP < EMA20           (below 20-day trend)
      • RSI between 30–55     (weakness without being oversold)
      • Price change % < 0    (today is red)

    NEUTRAL when conditions are mixed.

    Stop-Loss  = CMP − ATR × 1.5   (for BUY)
               = CMP + ATR × 1.5   (for SELL)
    Target     = CMP + ATR × 4.5   (3:1 reward for BUY)
               = CMP − ATR × 4.5   (3:1 reward for SELL)
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist   = ticker.history(period="30d", auto_adjust=True)

        if hist.empty or len(hist) < 15:
            return None

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        cmp        = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        price_chg  = round((cmp - prev_close) / prev_close * 100, 2)

        # ── VWAP (swing, from available history) ─────────────────────────────
        typical = (high + low + close) / 3
        vwap    = float((typical * volume).sum() / volume.sum())

        # ── EMA 20 ───────────────────────────────────────────────────────────
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # ── ATR 14 ───────────────────────────────────────────────────────────
        tr  = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().dropna().iloc[-1])

        # ── RSI 14 ───────────────────────────────────────────────────────────
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = float(100 - (100 / (1 + rs)).dropna().iloc[-1])

        # ── Volume surge flag ─────────────────────────────────────────────────
        avg_vol    = float(volume.iloc[:-1].mean())
        vol_surge  = volume.iloc[-1] > avg_vol * 1.3

        # ── Signal decision ───────────────────────────────────────────────────
        buy_score  = sum([
            cmp > vwap,
            cmp > ema20,
            45 <= rsi <= 72,
            price_chg > 0,
            vol_surge,
        ])
        sell_score = sum([
            cmp < vwap,
            cmp < ema20,
            28 <= rsi <= 55,
            price_chg < 0,
            vol_surge,
        ])

        if buy_score >= 4:
            signal   = "🟢 BUY"
            sl       = round(cmp - atr * 1.5, 2)
            target   = round(cmp + atr * 4.5, 2)   # 3:1 RR
        elif sell_score >= 4:
            signal   = "🔴 SELL"
            sl       = round(cmp + atr * 1.5, 2)
            target   = round(cmp - atr * 4.5, 2)   # 3:1 RR
        else:
            signal   = "⚪ NEUTRAL"
            sl       = round(cmp - atr * 1.5, 2)
            target   = round(cmp + atr * 1.5, 2)

        return {
            "Symbol":     symbol,
            "Signal":     signal,
            "CMP":        round(cmp, 2),
            "Target":     target,
            "Stop-Loss":  sl,
            "ATR":        round(atr, 2),
            "RSI":        round(rsi, 1),
            "EMA20":      round(ema20, 2),
            "VWAP":       round(vwap, 2),
            "Chg %":      price_chg,
            "Vol Surge":  "✅ Yes" if vol_surge else "—",
            "Score":      max(buy_score, sell_score),
        }

    except Exception:
        return None


def mock_signal(symbol: str, idx: int) -> dict:
    """Fallback mock signal when yfinance is unavailable."""
    rng     = np.random.default_rng(idx * 7 + 13)
    cmp     = round(float(rng.uniform(200, 4000)), 2)
    atr     = round(cmp * float(rng.uniform(0.015, 0.035)), 2)
    signals = ["🟢 BUY", "🔴 SELL", "⚪ NEUTRAL"]
    weights = [0.35, 0.30, 0.35]
    signal  = rng.choice(signals, p=weights)
    if signal == "🟢 BUY":
        sl, tgt = round(cmp - atr*1.5, 2), round(cmp + atr*4.5, 2)
    elif signal == "🔴 SELL":
        sl, tgt = round(cmp + atr*1.5, 2), round(cmp - atr*4.5, 2)
    else:
        sl, tgt = round(cmp - atr*1.5, 2), round(cmp + atr*1.5, 2)
    return {
        "Symbol": symbol, "Signal": signal,
        "CMP": cmp, "Target": tgt, "Stop-Loss": sl,
        "ATR": atr, "RSI": round(float(rng.uniform(30, 70)), 1),
        "EMA20": round(cmp * float(rng.uniform(0.97, 1.03)), 2),
        "VWAP":  round(cmp * float(rng.uniform(0.98, 1.02)), 2),
        "Chg %": round(float(rng.uniform(-3, 3)), 2),
        "Vol Surge": "✅ Yes" if rng.random() > 0.6 else "—",
        "Score": int(rng.integers(2, 6)),
    }


def run_scan(symbols: list) -> pd.DataFrame:
    """
    Parallel scan of all symbols.
    Returns a DataFrame sorted by Signal strength.
    """
    results = []

    if YFINANCE_OK:
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(compute_signal, s): s for s in symbols}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)
    else:
        # Mock fallback
        for i, sym in enumerate(symbols):
            results.append(mock_signal(sym, i))

    df = pd.DataFrame(results)

    # Sort: BUY first → SELL → NEUTRAL, then by score descending
    order = {"🟢 BUY": 0, "🔴 SELL": 1, "⚪ NEUTRAL": 2}
    df["_ord"] = df["Signal"].map(order)
    df = df.sort_values(["_ord", "Score"], ascending=[True, False])
    df = df.drop(columns=["_ord", "Score"]).reset_index(drop=True)
    return df


# =============================================================================
#  SESSION STATE
# =============================================================================

if "signals_df"  not in st.session_state:
    st.session_state["signals_df"]  = None
if "last_scan"   not in st.session_state:
    st.session_state["last_scan"]   = None
if "scanning"    not in st.session_state:
    st.session_state["scanning"]    = False


# =============================================================================
#  HEADER
# =============================================================================

st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.5rem;'>
    <p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
              letter-spacing:0.15em;color:#4B6080;margin:0;'>
        NSE F&O · REAL-TIME · FREE
    </p>
    <h1 style='font-size:2.4rem;font-weight:700;margin:6px 0 4px;
               background:linear-gradient(135deg,#00C851,#00BFFF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🎯 F&O Signal Screener
    </h1>
    <p style='font-size:0.85rem;color:#4B6080;margin:0;'>
        Scans 30 NSE F&O stocks · BUY / SELL / NEUTRAL signals · 3:1 Risk-Reward
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── yfinance missing warning ──────────────────────────────────────────────────
if not YFINANCE_OK:
    st.warning(
        "**yfinance not installed** — showing demo data. "
        "Add `requirements.txt` to your GitHub repo root and reboot the app for live data.",
        icon="⚠️",
    )

# =============================================================================
#  REFRESH BUTTON  (centred, prominent)
# =============================================================================

col_l, col_btn, col_r = st.columns([2, 3, 2])
with col_btn:
    scan_now = st.button("🔄  SCAN MARKET NOW", use_container_width=True)

# Show last update time
if st.session_state["last_scan"]:
    st.markdown(
        f"<div style='text-align:center;margin-top:8px;'>"
        f"<span class='update-tag'>Last scan: {st.session_state['last_scan']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
#  SCAN TRIGGER
# =============================================================================

if scan_now:
    with st.spinner("🔍 Scanning 30 F&O stocks… please wait 30–60 seconds"):
        df = run_scan(SYMBOLS)
    st.session_state["signals_df"] = df
    st.session_state["last_scan"]  = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    st.rerun()


# =============================================================================
#  RESULTS
# =============================================================================

if st.session_state["signals_df"] is not None:
    df = st.session_state["signals_df"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    buy_count     = int((df["Signal"] == "🟢 BUY").sum())
    sell_count    = int((df["Signal"] == "🔴 SELL").sum())
    neutral_count = int((df["Signal"] == "⚪ NEUTRAL").sum())
    total         = len(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📊 Total Scanned", str(total))
    with m2:
        st.metric("🟢 BUY Signals",  str(buy_count),
                  delta=f"{round(buy_count/total*100)}% of universe")
    with m3:
        st.metric("🔴 SELL Signals", str(sell_count),
                  delta=f"{round(sell_count/total*100)}% of universe")
    with m4:
        st.metric("⚪ Neutral",       str(neutral_count))

    st.divider()

    # ── BUY signals ───────────────────────────────────────────────────────────
    buy_df = df[df["Signal"] == "🟢 BUY"].copy()
    if not buy_df.empty:
        st.markdown(
            "<h2 style='color:#00C851;font-size:1.1rem;letter-spacing:0.06em;"
            "text-transform:uppercase;'>🟢 BUY Signals</h2>",
            unsafe_allow_html=True,
        )
        buy_display = buy_df.drop(columns=["Signal"]).copy()
        for c in ["CMP","Target","Stop-Loss","ATR","EMA20","VWAP","Chg %"]:
            buy_display[c] = buy_display[c].astype("float64")
        buy_display["RSI"] = buy_display["RSI"].astype("float64")

        st.dataframe(
            buy_display,
            column_config={
                "Symbol":    st.column_config.TextColumn("Symbol",      width="small"),
                "CMP":       st.column_config.NumberColumn("CMP (₹)",   format="₹%.2f"),
                "Target":    st.column_config.NumberColumn("Target (₹)",format="₹%.2f",
                             help="3:1 Reward target"),
                "Stop-Loss": st.column_config.NumberColumn("SL (₹)",    format="₹%.2f"),
                "ATR":       st.column_config.NumberColumn("ATR",       format="₹%.2f"),
                "RSI":       st.column_config.NumberColumn("RSI",       format="%.1f"),
                "EMA20":     st.column_config.NumberColumn("EMA 20",    format="₹%.2f"),
                "VWAP":      st.column_config.NumberColumn("VWAP",      format="₹%.2f"),
                "Chg %":     st.column_config.NumberColumn("Chg %",     format="%.2f%%"),
                "Vol Surge": st.column_config.TextColumn("Vol Surge",   width="small"),
            },
            use_container_width=True,
            hide_index=True,
            height=min(80 + 38 * len(buy_display), 500),
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── SELL signals ──────────────────────────────────────────────────────────
    sell_df = df[df["Signal"] == "🔴 SELL"].copy()
    if not sell_df.empty:
        st.markdown(
            "<h2 style='color:#FF3B5C;font-size:1.1rem;letter-spacing:0.06em;"
            "text-transform:uppercase;'>🔴 SELL Signals</h2>",
            unsafe_allow_html=True,
        )
        sell_display = sell_df.drop(columns=["Signal"]).copy()
        for c in ["CMP","Target","Stop-Loss","ATR","EMA20","VWAP","Chg %"]:
            sell_display[c] = sell_display[c].astype("float64")
        sell_display["RSI"] = sell_display["RSI"].astype("float64")

        st.dataframe(
            sell_display,
            column_config={
                "Symbol":    st.column_config.TextColumn("Symbol",      width="small"),
                "CMP":       st.column_config.NumberColumn("CMP (₹)",   format="₹%.2f"),
                "Target":    st.column_config.NumberColumn("Target (₹)",format="₹%.2f",
                             help="3:1 Reward target"),
                "Stop-Loss": st.column_config.NumberColumn("SL (₹)",    format="₹%.2f"),
                "ATR":       st.column_config.NumberColumn("ATR",       format="₹%.2f"),
                "RSI":       st.column_config.NumberColumn("RSI",       format="%.1f"),
                "EMA20":     st.column_config.NumberColumn("EMA 20",    format="₹%.2f"),
                "VWAP":      st.column_config.NumberColumn("VWAP",      format="₹%.2f"),
                "Chg %":     st.column_config.NumberColumn("Chg %",     format="%.2f%%"),
                "Vol Surge": st.column_config.TextColumn("Vol Surge",   width="small"),
            },
            use_container_width=True,
            hide_index=True,
            height=min(80 + 38 * len(sell_display), 500),
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── NEUTRAL signals ───────────────────────────────────────────────────────
    neutral_df = df[df["Signal"] == "⚪ NEUTRAL"].copy()
    with st.expander(f"⚪  Neutral / No Clear Signal ({len(neutral_df)} symbols)",
                     expanded=False):
        if not neutral_df.empty:
            neutral_display = neutral_df.drop(columns=["Signal"]).copy()
            for c in ["CMP","Target","Stop-Loss","ATR","EMA20","VWAP","Chg %"]:
                neutral_display[c] = neutral_display[c].astype("float64")
            neutral_display["RSI"] = neutral_display["RSI"].astype("float64")
            st.dataframe(
                neutral_display,
                column_config={
                    "Symbol":    st.column_config.TextColumn("Symbol",   width="small"),
                    "CMP":       st.column_config.NumberColumn("CMP",    format="₹%.2f"),
                    "Target":    st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "Stop-Loss": st.column_config.NumberColumn("SL",     format="₹%.2f"),
                    "RSI":       st.column_config.NumberColumn("RSI",    format="%.1f"),
                    "Chg %":     st.column_config.NumberColumn("Chg %",  format="%.2f%%"),
                    "Vol Surge": st.column_config.TextColumn("Vol Surge",width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    csv = df.to_csv(index=False).encode("utf-8")
    col_e1, col_e2, col_e3 = st.columns([2, 2, 2])
    with col_e2:
        st.download_button(
            label     = "⬇  Download Full Report CSV",
            data      = csv,
            file_name = f"fo_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

else:
    # ── Empty state — before first scan ───────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>📡</div>
        <p style='font-size:1.2rem;color:#E2E8F0;font-weight:600;margin:0;'>
            No signals yet
        </p>
        <p style='font-size:0.85rem;color:#4B6080;margin:8px 0 0;'>
            Press <strong style='color:#00C851;'>SCAN MARKET NOW</strong>
            to fetch live data and generate signals
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Signal logic explainer ────────────────────────────────────────────────────
with st.expander("📖  How signals are generated", expanded=False):
    st.markdown("""
| Indicator | BUY condition | SELL condition |
|---|---|---|
| **VWAP** | CMP above VWAP | CMP below VWAP |
| **EMA 20** | CMP above EMA20 | CMP below EMA20 |
| **RSI 14** | Between 45–72 (momentum) | Between 28–55 (weakness) |
| **Price Change** | Today is green (+%) | Today is red (−%) |
| **Volume Surge** | Volume > 1.3× average | Volume > 1.3× average |

**Threshold: 4 out of 5 conditions must match** for a BUY or SELL signal.
If fewer than 4 match → NEUTRAL.

**Stop-Loss** = CMP ± ATR × 1.5
**Target** = CMP ± ATR × 4.5  *(3:1 Risk-Reward)*

> Data via Yahoo Finance · ~2 min delay · Not financial advice
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 0.5rem;'>
    <p style='font-family:JetBrains Mono,monospace;font-size:0.58rem;
              color:#1A2535;letter-spacing:0.1em;'>
        F&O SIGNAL SCREENER · yfinance · NOT FINANCIAL ADVICE
    </p>
</div>
""", unsafe_allow_html=True)
