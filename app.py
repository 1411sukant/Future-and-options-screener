# =============================================================================
#  NSE F&O SIGNAL SCREENER  |  app.py  v4.0
#  Signals : RSI · MACD · EMA Cross · Bollinger Bands · Volume Surge
#  Data    : yfinance (free, ~2 min delay)
#  Run     : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    LIVE = True
except ModuleNotFoundError:
    LIVE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE F&O Signal Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #05080F !important;
    color: #E2E8F0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E293B !important;
}

/* ── Divider ── */
hr { border-color: #1E293B !important; margin: 0.5rem 0 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0D1117 0%, #111827 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #F8FAFC !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Refresh button ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    border: none !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(99,102,241,0.6) !important;
}

/* ── Select box ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748B !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    color: #64748B !important;
}

/* ── Info box ── */
[data-testid="stInfo"] {
    background: #0F172A !important;
    border-left: 3px solid #6366F1 !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #6366F1 !important;
    color: #6366F1 !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    padding: 0.4rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  UNIVERSE
# =============================================================================

INDICES = {
    "^NSEI":      {"name": "NIFTY 50",    "emoji": "🔵"},
    "^NSEBANK":   {"name": "BANK NIFTY",  "emoji": "🏦"},
    "^CNXIT":     {"name": "NIFTY IT",    "emoji": "💻"},
    "^CNXPHARMA": {"name": "NIFTY PHARMA","emoji": "💊"},
    "^CNXAUTO":   {"name": "NIFTY AUTO",  "emoji": "🚗"},
    "^CNXMIDCAP": {"name": "MIDCAP 100",  "emoji": "📊"},
}

FO_STOCKS = [
    "RELIANCE", "TCS",       "HDFCBANK",  "ICICIBANK",  "INFY",
    "SBIN",     "AXISBANK",  "KOTAKBANK", "BAJFINANCE", "TITAN",
    "MARUTI",   "LT",        "TATAMOTORS","SUNPHARMA",  "WIPRO",
    "HCLTECH",  "ONGC",      "ADANIENT",  "TATASTEEL",  "APOLLOHOSP",
]


# =============================================================================
#  TECHNICAL INDICATOR ENGINE
# =============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def compute_macd(close: pd.Series):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def compute_bb(close: pd.Series, period: int = 20):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower


def compute_atr(high, low, close, period: int = 14) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().dropna().iloc[-1])


def generate_signal(hist: pd.DataFrame) -> dict:
    """
    Multi-indicator signal engine.
    Scores each indicator, sums to a final signal.

    Score range : -6 to +6
    ≥ 4  → STRONG BUY  🚀
    2–3  → BUY         🟢
    -1–1 → HOLD        🟡
    -2 to -3 → SELL    🔴
    ≤ -4 → STRONG SELL ⛔
    """
    if len(hist) < 30:
        return None

    close  = hist["Close"]
    high   = hist["High"]
    low    = hist["Low"]
    volume = hist["Volume"]

    # ── Indicators ────────────────────────────────────────────────────────────
    rsi               = compute_rsi(close)
    macd, sig, _hist  = compute_macd(close)
    ema20             = close.ewm(span=20, adjust=False).mean()
    ema50             = close.ewm(span=50, adjust=False).mean()
    bb_upper, bb_mid, bb_lower = compute_bb(close)
    atr               = compute_atr(high, low, close)

    # Latest values
    c      = float(close.iloc[-1])
    r      = float(rsi.iloc[-1])
    m      = float(macd.iloc[-1])
    ms     = float(sig.iloc[-1])
    e20    = float(ema20.iloc[-1])
    e50    = float(ema50.iloc[-1])
    bbu    = float(bb_upper.iloc[-1])
    bbl    = float(bb_lower.iloc[-1])
    prev_c = float(close.iloc[-2])
    avg_vol= float(volume.iloc[-20:].mean())
    cur_vol= float(volume.iloc[-1])

    score   = 0
    reasons = []

    # ── RSI Signal ────────────────────────────────────────────────────────────
    if r <= 30:
        score += 2; reasons.append(f"RSI Oversold ({r:.0f}) 🔥")
    elif r <= 45:
        score += 1; reasons.append(f"RSI Bullish Zone ({r:.0f})")
    elif r >= 70:
        score -= 2; reasons.append(f"RSI Overbought ({r:.0f}) ⚠️")
    elif r >= 55:
        score -= 1; reasons.append(f"RSI Bearish Zone ({r:.0f})")
    else:
        reasons.append(f"RSI Neutral ({r:.0f})")

    # ── MACD Signal ───────────────────────────────────────────────────────────
    if m > ms:
        # Check for fresh crossover in last 3 bars
        prev_macd = float(macd.iloc[-3])
        prev_sig  = float(sig.iloc[-3])
        if prev_macd < prev_sig:
            score += 2; reasons.append("MACD Fresh Bullish Cross ✅")
        else:
            score += 1; reasons.append("MACD Bullish")
    else:
        prev_macd = float(macd.iloc[-3])
        prev_sig  = float(sig.iloc[-3])
        if prev_macd > prev_sig:
            score -= 2; reasons.append("MACD Fresh Bearish Cross ❌")
        else:
            score -= 1; reasons.append("MACD Bearish")

    # ── EMA Trend ─────────────────────────────────────────────────────────────
    if c > e20 and e20 > e50:
        score += 2; reasons.append("Price > EMA20 > EMA50 (Uptrend)")
    elif c > e20:
        score += 1; reasons.append("Price Above EMA20")
    elif c < e20 and e20 < e50:
        score -= 2; reasons.append("Price < EMA20 < EMA50 (Downtrend)")
    elif c < e20:
        score -= 1; reasons.append("Price Below EMA20")

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if c <= bbl:
        score += 1; reasons.append("At Lower Bollinger (Reversal Zone)")
    elif c >= bbu:
        score -= 1; reasons.append("At Upper Bollinger (Sell Zone)")

    # ── Volume Surge ─────────────────────────────────────────────────────────
    if cur_vol > 1.8 * avg_vol and c > prev_c:
        score += 1; reasons.append("Volume Surge + Price Up 📈")
    elif cur_vol > 1.8 * avg_vol and c < prev_c:
        score -= 1; reasons.append("Volume Surge + Price Down 📉")

    # ── Final Signal ─────────────────────────────────────────────────────────
    if score >= 4:
        signal, color, emoji = "STRONG BUY",  "#00C853", "🚀"
    elif score >= 2:
        signal, color, emoji = "BUY",         "#4CAF50", "🟢"
    elif score <= -4:
        signal, color, emoji = "STRONG SELL", "#D50000", "⛔"
    elif score <= -2:
        signal, color, emoji = "SELL",        "#F44336", "🔴"
    else:
        signal, color, emoji = "HOLD",        "#FF9800", "🟡"

    # ── Entry / Target / Stop-Loss ────────────────────────────────────────────
    if score >= 0:   # bullish bias
        entry  = round(c, 2)
        sl     = round(c - atr * 1.5, 2)
        target = round(c + atr * 3.0, 2)
    else:            # bearish bias
        entry  = round(c, 2)
        sl     = round(c + atr * 1.5, 2)
        target = round(c - atr * 3.0, 2)

    price_change = round((c - prev_c) / prev_c * 100, 2)

    return {
        "signal":       signal,
        "color":        color,
        "emoji":        emoji,
        "score":        score,
        "cmp":          round(c, 2),
        "change_pct":   price_change,
        "entry":        entry,
        "target":       target,
        "sl":           sl,
        "atr":          round(atr, 2),
        "rsi":          round(r, 1),
        "reasons":      reasons[:3],
    }


# =============================================================================
#  DATA FETCHER
# =============================================================================

def fetch_signal(ticker_sym: str, display_name: str) -> dict | None:
    """Fetch OHLCV and generate signal for one symbol."""
    try:
        t    = yf.Ticker(ticker_sym)
        hist = t.history(period="60d", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return None
        result = generate_signal(hist)
        if result:
            result["symbol"]  = display_name
            result["ticker"]  = ticker_sym
        return result
    except Exception:
        return None


def mock_signal(display_name: str, ticker: str) -> dict:
    """Deterministic mock signal when yfinance is unavailable."""
    rng    = np.random.default_rng(abs(hash(ticker)) % 9999)
    score  = int(rng.integers(-5, 6))
    cmp    = round(float(rng.uniform(200, 4000)), 2)
    atr    = round(cmp * float(rng.uniform(0.015, 0.035)), 2)
    change = round(float(rng.uniform(-3, 3)), 2)

    if score >= 4:   signal, color, emoji = "STRONG BUY",  "#00C853", "🚀"
    elif score >= 2: signal, color, emoji = "BUY",         "#4CAF50", "🟢"
    elif score <= -4:signal, color, emoji = "STRONG SELL", "#D50000", "⛔"
    elif score <= -2:signal, color, emoji = "SELL",        "#F44336", "🔴"
    else:            signal, color, emoji = "HOLD",        "#FF9800", "🟡"

    return {
        "symbol": display_name, "ticker": ticker,
        "signal": signal, "color": color, "emoji": emoji,
        "score": score, "cmp": cmp, "change_pct": change,
        "entry": cmp,
        "target": round(cmp + atr*3, 2) if score >= 0 else round(cmp - atr*3, 2),
        "sl":     round(cmp - atr*1.5, 2) if score >= 0 else round(cmp + atr*1.5, 2),
        "atr": atr, "rsi": round(float(rng.uniform(25, 75)), 1),
        "reasons": ["Mock data — add requirements.txt to GitHub"],
    }


@st.cache_data(ttl=180, show_spinner=False)
def load_all_signals(_cache_key: int) -> tuple[list, list]:
    """
    Returns (index_signals, stock_signals).
    Cached for 3 minutes. Cache busted by refresh button.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    idx_results   = []
    stock_results = []

    if not LIVE:
        for sym, meta in INDICES.items():
            idx_results.append(mock_signal(meta["name"], sym))
        for sym in FO_STOCKS:
            stock_results.append(mock_signal(sym, f"{sym}.NS"))
        return idx_results, stock_results

    # ── Parallel fetch ────────────────────────────────────────────────────────
    all_tasks = (
        [(sym, meta["name"], "idx") for sym, meta in INDICES.items()] +
        [(f"{sym}.NS", sym, "stock") for sym in FO_STOCKS]
    )

    with ThreadPoolExecutor(max_workers=10) as ex:
        future_map = {
            ex.submit(fetch_signal, ticker, name): (ticker, name, kind)
            for ticker, name, kind in all_tasks
        }
        for future in as_completed(future_map):
            ticker, name, kind = future_map[future]
            res = future.result()
            if res is None:
                res = mock_signal(name, ticker)
            if kind == "idx":
                idx_results.append(res)
            else:
                stock_results.append(res)

    # Sort: STRONG BUY first, STRONG SELL last, HOLD in middle
    order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG SELL": 4}
    idx_results   = sorted(idx_results,   key=lambda x: order.get(x["signal"], 2))
    stock_results = sorted(stock_results, key=lambda x: order.get(x["signal"], 2))

    return idx_results, stock_results


# =============================================================================
#  SESSION STATE
# =============================================================================
if "cache_key" not in st.session_state:
    st.session_state["cache_key"] = 0
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = "Never"


# =============================================================================
#  CARD RENDERER
# =============================================================================

SIGNAL_BG = {
    "STRONG BUY":  "linear-gradient(135deg,#002B14 0%,#003D1C 100%)",
    "BUY":         "linear-gradient(135deg,#0A1F10 0%,#0F2D17 100%)",
    "HOLD":        "linear-gradient(135deg,#1A1400 0%,#261D00 100%)",
    "SELL":        "linear-gradient(135deg,#1F0A0A 0%,#2D0F0F 100%)",
    "STRONG SELL": "linear-gradient(135deg,#1A0000 0%,#2B0000 100%)",
}
SIGNAL_BORDER = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#4CAF50",
    "HOLD":        "#FF9800",
    "SELL":        "#F44336",
    "STRONG SELL": "#D50000",
}
BADGE_BG = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#4CAF50",
    "HOLD":        "#FF9800",
    "SELL":        "#F44336",
    "STRONG SELL": "#D50000",
}
BADGE_COLOR = {
    "STRONG BUY":  "#000",
    "BUY":         "#000",
    "HOLD":        "#000",
    "SELL":        "#fff",
    "STRONG SELL": "#fff",
}


def signal_card(d: dict, large: bool = False) -> str:
    """Generate HTML for a signal card."""
    sig     = d["signal"]
    bg      = SIGNAL_BG.get(sig, SIGNAL_BG["HOLD"])
    border  = SIGNAL_BORDER.get(sig, "#FF9800")
    bbg     = BADGE_BG.get(sig, "#FF9800")
    bclr    = BADGE_COLOR.get(sig, "#000")
    chg_col = "#4CAF50" if d["change_pct"] >= 0 else "#F44336"
    chg_sym = "▲" if d["change_pct"] >= 0 else "▼"
    tgt_col = "#4CAF50" if d["score"] >= 0 else "#F44336"
    sl_col  = "#F44336" if d["score"] >= 0 else "#4CAF50"
    reasons_html = "".join(
        f"<div style='font-size:0.68rem;color:#94A3B8;margin:2px 0;'>• {r}</div>"
        for r in d["reasons"]
    )
    score_w = int(((d["score"] + 6) / 12) * 100)
    score_col = border

    # Score bar dots
    dots = ""
    for i in range(-6, 7):
        if i == 0:
            dots += "<span style='color:#64748B;font-size:0.5rem;'>|</span>"
            continue
        filled = (d["score"] >= 0 and i > 0 and i <= d["score"]) or \
                 (d["score"] < 0 and i < 0 and i >= d["score"])
        col = "#4CAF50" if i > 0 else "#F44336"
        dots += f"<span style='color:{'"+col+"' if filled else '#1E293B'};font-size:0.9rem;'>{'●' if filled else '○'}</span>"

    font_size = "1.1rem" if large else "0.95rem"
    price_size = "1.6rem" if large else "1.3rem"
    pad = "1.4rem" if large else "1rem"

    return f"""
<div style='
    background:{bg};
    border:1px solid {border};
    border-radius:16px;
    padding:{pad};
    margin:6px 0;
    position:relative;
    overflow:hidden;
    transition:all 0.2s;
'>
  <div style='position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,{border}88,{border});'></div>

  <!-- Header -->
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;'>
    <div>
      <div style='font-size:{font_size};font-weight:800;color:#F8FAFC;
                  letter-spacing:-0.02em;'>{d['symbol']}</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:{price_size};
                  font-weight:700;color:#F8FAFC;margin:2px 0;'>
        ₹{d['cmp']:,.2f}
      </div>
      <div style='font-size:0.78rem;color:{chg_col};font-weight:600;'>
        {chg_sym} {abs(d['change_pct'])}%
      </div>
    </div>
    <div>
      <div style='background:{bbg};color:{bclr};font-size:0.72rem;font-weight:800;
                  padding:5px 12px;border-radius:20px;letter-spacing:0.06em;
                  text-align:center;'>{d['emoji']} {sig}</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                  color:#64748B;text-align:center;margin-top:4px;'>
        RSI {d['rsi']:.0f}
      </div>
    </div>
  </div>

  <!-- Levels -->
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:10px 0;'>
    <div style='background:#0D1117;border-radius:8px;padding:6px 8px;text-align:center;'>
      <div style='font-size:0.58rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;'>Entry</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                  color:#E2E8F0;font-weight:600;'>₹{d['entry']:,.0f}</div>
    </div>
    <div style='background:#0D1117;border-radius:8px;padding:6px 8px;text-align:center;'>
      <div style='font-size:0.58rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;'>Target</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                  color:{tgt_col};font-weight:700;'>₹{d['target']:,.0f}</div>
    </div>
    <div style='background:#0D1117;border-radius:8px;padding:6px 8px;text-align:center;'>
      <div style='font-size:0.58rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;'>Stop Loss</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                  color:{sl_col};font-weight:700;'>₹{d['sl']:,.0f}</div>
    </div>
  </div>

  <!-- Signal Score Dots -->
  <div style='margin:8px 0 6px;text-align:center;letter-spacing:2px;'>{dots}</div>

  <!-- Reasons -->
  <div style='border-top:1px solid #1E293B;padding-top:8px;margin-top:4px;'>
    {reasons_html}
  </div>
</div>
"""


def index_hero_card(d: dict, meta: dict) -> str:
    """Larger hero card for indices like Nifty / Bank Nifty."""
    sig     = d["signal"]
    border  = SIGNAL_BORDER.get(sig, "#FF9800")
    bbg     = BADGE_BG.get(sig, "#FF9800")
    bclr    = BADGE_COLOR.get(sig, "#000")
    chg_col = "#4CAF50" if d["change_pct"] >= 0 else "#F44336"
    chg_sym = "▲" if d["change_pct"] >= 0 else "▼"
    tgt_col = "#4CAF50" if d["score"] >= 0 else "#F44336"
    sl_col  = "#F44336" if d["score"] >= 0 else "#4CAF50"

    return f"""
<div style='
    background:linear-gradient(135deg,#0D1117 0%,#111827 100%);
    border:1.5px solid {border};
    border-radius:20px;
    padding:1.6rem;
    margin:4px 0;
    position:relative;
    overflow:hidden;
'>
  <div style='position:absolute;top:0;left:0;right:0;height:4px;
              background:linear-gradient(90deg,{border},{border}44);'></div>
  <div style='position:absolute;top:-30px;right:-30px;width:120px;height:120px;
              border-radius:50%;background:{border}08;'></div>

  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div style='font-size:0.68rem;font-weight:600;color:#64748B;
                  letter-spacing:0.14em;text-transform:uppercase;margin-bottom:4px;'>
        {meta["emoji"]}  {d["symbol"]}
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:2rem;
                  font-weight:800;color:#F8FAFC;line-height:1;'>
        {d['cmp']:,.2f}
      </div>
      <div style='font-size:0.9rem;color:{chg_col};font-weight:700;margin-top:4px;'>
        {chg_sym} {abs(d['change_pct'])}% today
      </div>
    </div>
    <div style='text-align:right;'>
      <div style='background:{bbg};color:{bclr};font-size:0.85rem;font-weight:800;
                  padding:8px 18px;border-radius:24px;letter-spacing:0.06em;
                  margin-bottom:8px;'>{d["emoji"]} {sig}</div>
      <div style='font-size:0.7rem;color:#64748B;'>RSI {d["rsi"]:.0f} · ATR {d["atr"]:.0f}</div>
    </div>
  </div>

  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:14px;'>
    <div style='background:#060B14;border-radius:10px;padding:8px 12px;text-align:center;'>
      <div style='font-size:0.6rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;margin-bottom:3px;'>Entry</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;
                  color:#E2E8F0;font-weight:700;'>₹{d['entry']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:10px;padding:8px 12px;text-align:center;'>
      <div style='font-size:0.6rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;margin-bottom:3px;'>Target</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;
                  color:{tgt_col};font-weight:700;'>₹{d['target']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:10px;padding:8px 12px;text-align:center;'>
      <div style='font-size:0.6rem;color:#64748B;text-transform:uppercase;
                  letter-spacing:0.1em;font-weight:600;margin-bottom:3px;'>Stop Loss</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;
                  color:{sl_col};font-weight:700;'>₹{d['sl']:,.0f}</div>
    </div>
  </div>
</div>
"""


# =============================================================================
#  HEADER
# =============================================================================
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.5rem;'>
  <div style='font-size:0.7rem;font-weight:600;letter-spacing:0.2em;
              text-transform:uppercase;color:#6366F1;margin-bottom:8px;'>
    NSE · FUTURES & OPTIONS
  </div>
  <h1 style='font-size:2.4rem;font-weight:800;margin:0;
             background:linear-gradient(135deg,#F8FAFC 0%,#94A3B8 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             letter-spacing:-0.03em;'>
    Signal Screener
  </h1>
  <p style='color:#475569;font-size:0.85rem;margin:8px 0 0;'>
    RSI · MACD · EMA · Bollinger Bands · Volume — all in one signal
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =============================================================================
#  CONTROLS ROW
# =============================================================================
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])

with ctrl1:
    if st.button("🔄  REFRESH SIGNALS", use_container_width=True):
        st.cache_data.clear()
        st.session_state["cache_key"] += 1
        from datetime import datetime
        st.session_state["last_refresh"] = datetime.now().strftime("%I:%M:%S %p")
        st.rerun()

with ctrl2:
    filter_signal = st.selectbox(
        "Filter by Signal",
        ["All Signals", "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
        label_visibility="collapsed",
    )

with ctrl3:
    filter_section = st.selectbox(
        "Section",
        ["All", "Indices Only", "F&O Stocks Only"],
        label_visibility="collapsed",
    )

with ctrl4:
    st.markdown(
        f"<div style='text-align:right;padding:0.5rem 0;'>"
        f"<span style='font-size:0.65rem;color:#64748B;font-family:JetBrains Mono,monospace;"
        f"letter-spacing:0.08em;text-transform:uppercase;'>Last Refresh</span><br>"
        f"<span style='font-size:0.85rem;color:#6366F1;font-weight:600;'>"
        f"{st.session_state['last_refresh']}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# =============================================================================
#  LOAD DATA
# =============================================================================
if not LIVE:
    st.warning("⚠️ yfinance not found — showing demo data. Add `requirements.txt` to your GitHub repo and reboot.", icon="📦")

with st.spinner("⏳ Loading live signals… (cached for 3 min after first load)"):
    idx_signals, stock_signals = load_all_signals(
        _cache_key=st.session_state["cache_key"]
    )

# Apply filters
def apply_filter(signals):
    out = signals
    if filter_signal != "All Signals":
        out = [s for s in out if s["signal"] == filter_signal]
    return out

filtered_idx    = apply_filter(idx_signals)   if filter_section != "F&O Stocks Only" else []
filtered_stocks = apply_filter(stock_signals) if filter_section != "Indices Only"    else []

# =============================================================================
#  SUMMARY METRICS
# =============================================================================
all_sigs = idx_signals + stock_signals
counts = {s: sum(1 for x in all_sigs if x["signal"] == s)
          for s in ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]}

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("🚀 Strong Buy",  counts["STRONG BUY"],
              delta=f"{counts['STRONG BUY']} setups")
with m2:
    st.metric("🟢 Buy",         counts["BUY"],
              delta=f"{counts['BUY']} setups")
with m3:
    st.metric("🟡 Hold",        counts["HOLD"],
              delta=f"{counts['HOLD']} neutral")
with m4:
    st.metric("🔴 Sell",        counts["SELL"],
              delta=f"{counts['SELL']} setups")
with m5:
    st.metric("⛔ Strong Sell", counts["STRONG SELL"],
              delta=f"{counts['STRONG SELL']} setups")

st.divider()

# =============================================================================
#  INDEX HERO CARDS
# =============================================================================
if filtered_idx:
    st.markdown("""
    <div style='font-size:0.7rem;font-weight:700;letter-spacing:0.14em;
                text-transform:uppercase;color:#6366F1;margin-bottom:12px;'>
        📊  Market Indices
    </div>
    """, unsafe_allow_html=True)

    # Build ticker→meta mapping for rendering
    idx_meta_map = {v["name"]: {"emoji": v["emoji"]} for v in INDICES.values()}

    # 3 per row
    for row_start in range(0, len(filtered_idx), 3):
        row = filtered_idx[row_start:row_start+3]
        cols = st.columns(len(row))
        for col, d in zip(cols, row):
            meta = idx_meta_map.get(d["symbol"], {"emoji": "📈"})
            col.markdown(index_hero_card(d, meta), unsafe_allow_html=True)

    st.divider()

# =============================================================================
#  F&O STOCK SIGNAL CARDS
# =============================================================================
if filtered_stocks:
    st.markdown("""
    <div style='font-size:0.7rem;font-weight:700;letter-spacing:0.14em;
                text-transform:uppercase;color:#6366F1;margin-bottom:12px;'>
        🎯  F&O Stock Signals
    </div>
    """, unsafe_allow_html=True)

    # 4 cards per row
    for row_start in range(0, len(filtered_stocks), 4):
        row = filtered_stocks[row_start:row_start+4]
        cols = st.columns(4)
        for i, col in enumerate(cols):
            if i < len(row):
                col.markdown(signal_card(row[i]), unsafe_allow_html=True)

elif not filtered_idx:
    st.info("No signals match the selected filter. Try **'All Signals'** from the dropdown.")

st.divider()

# =============================================================================
#  SUMMARY TABLE  (collapsible)
# =============================================================================
with st.expander("📋  All Signals — Summary Table", expanded=False):
    all_data = []
    for d in (idx_signals + stock_signals):
        all_data.append({
            "Symbol":   d["symbol"],
            "Signal":   f"{d['emoji']} {d['signal']}",
            "CMP (₹)":  d["cmp"],
            "Change %": d["change_pct"],
            "RSI":      d["rsi"],
            "Entry (₹)":d["entry"],
            "Target (₹)":d["target"],
            "SL (₹)":   d["sl"],
            "Score":    d["score"],
            "Top Reason": d["reasons"][0] if d["reasons"] else "—",
        })

    tbl = pd.DataFrame(all_data)
    for c in ["CMP (₹)","Change %","Entry (₹)","Target (₹)","SL (₹)"]:
        tbl[c] = tbl[c].astype(float)
    tbl["RSI"]   = tbl["RSI"].astype(float)
    tbl["Score"] = tbl["Score"].astype(int)

    st.dataframe(
        tbl,
        column_config={
            "CMP (₹)":    st.column_config.NumberColumn(format="₹%.2f"),
            "Change %":   st.column_config.NumberColumn(format="%.2f%%"),
            "Entry (₹)":  st.column_config.NumberColumn(format="₹%.2f"),
            "Target (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "SL (₹)":     st.column_config.NumberColumn(format="₹%.2f"),
            "RSI":        st.column_config.NumberColumn(format="%.1f"),
            "Score":      st.column_config.NumberColumn(format="%d"),
        },
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV", csv, "fo_signals.csv", "text/csv")

# =============================================================================
#  DISCLAIMER
# =============================================================================
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;'>
  <p style='font-size:0.65rem;color:#1E293B;font-family:JetBrains Mono,monospace;
            letter-spacing:0.08em;line-height:1.8;'>
    ⚠️ SIGNALS ARE FOR EDUCATIONAL PURPOSES ONLY · NOT SEBI-REGISTERED ADVICE<br>
    DATA VIA YAHOO FINANCE (~2 MIN DELAY) · ALWAYS USE YOUR OWN JUDGEMENT
  </p>
</div>
""", unsafe_allow_html=True)
