# =============================================================================
#  F&O HIGH-PROBABILITY SCREENER  |  app.py
#  Stack : Streamlit · Pandas · NumPy
#  Run   : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="F&O Screener | High-Probability Setups",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark-terminal aesthetic ─────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0A0E17;
    --bg-secondary:  #111827;
    --bg-card:       #141C2E;
    --accent-green:  #00FF88;
    --accent-red:    #FF3B5C;
    --accent-yellow: #FFD700;
    --accent-blue:   #3B82F6;
    --accent-purple: #A855F7;
    --text-primary:  #E2E8F0;
    --text-muted:    #64748B;
    --border:        #1E293B;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--accent-green) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Section headers ── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h1 { color: var(--text-primary) !important; }
h2 { color: var(--accent-blue) !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted) !important;
}

/* ── Info / warning boxes ── */
[data-testid="stInfo"]    { background: #0F2027 !important; border-left: 3px solid var(--accent-blue) !important; }
[data-testid="stWarning"] { background: #1A1400 !important; border-left: 3px solid var(--accent-yellow) !important; }

/* ── Button ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 6px;
    padding: 0.4rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent-green) !important;
    color: var(--bg-primary) !important;
}

/* ── Tag badges (rendered inside markdown) ── */
.badge-green  { color: #0A0E17; background: #00FF88; border-radius: 4px; padding: 1px 6px; font-size: 0.72rem; font-weight: 700; }
.badge-red    { color: #0A0E17; background: #FF3B5C; border-radius: 4px; padding: 1px 6px; font-size: 0.72rem; font-weight: 700; }
.badge-yellow { color: #0A0E17; background: #FFD700; border-radius: 4px; padding: 1px 6px; font-size: 0.72rem; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
#  1. MOCK DATA GENERATOR
#  Produces a realistic NSE F&O universe dataframe.
#  Every call with a fixed seed is deterministic (reproducible).
# =============================================================================

NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "TITAN",
    "WIPRO", "ULTRACEMCO", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA",
    "LT", "NESTLEIND", "HCLTECH", "BAJAJFINSV", "POWERGRID", "NTPC",
    "TATAMOTORS", "TECHM", "ONGC", "DIVISLAB", "CIPLA", "DRREDDY",
    "TATACONSUM", "COALINDIA", "ADANIENT", "ADANIPORTS", "BRITANNIA",
    "EICHERMOT", "HEROMOTOCO", "JSWSTEEL", "TATASTEEL", "HINDALCO",
    "APOLLOHOSP", "BPCL", "GRASIM", "SHREECEM", "MM",
]

@st.cache_data(ttl=300)   # cache 5 min; in production replace with live feed
def generate_mock_fo_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulate an F&O market-data snapshot for the NSE universe.

    Columns generated
    -----------------
    Symbol, CMP, Prev_Close, VWAP
    OI, Prev_OI, OI_Change_Pct
    Volume, Avg_Volume
    IV (Implied Volatility %)
    PCR (Put-Call Ratio)
    ATR (Average True Range)
    Support, Resistance     ← key levels for RR validation
    """
    rng = np.random.default_rng(seed)
    n   = len(NIFTY50_SYMBOLS)

    # ── Price universe ──────────────────────────────────────────────────────
    cmp        = rng.uniform(200, 4000, n).round(2)
    prev_close = (cmp * rng.uniform(0.95, 1.05, n)).round(2)

    # VWAP: within ±2 % of CMP with a slight mean-reversion bias
    vwap_noise = rng.uniform(-0.02, 0.02, n)
    vwap       = (cmp * (1 + vwap_noise)).round(2)

    # ── Open Interest ────────────────────────────────────────────────────────
    oi          = rng.integers(500_000, 50_000_000, n)
    # OI change: fat-tailed distribution to mimic real activity bursts
    oi_change   = rng.normal(0.0, 8.0, n).round(2)          # % change
    prev_oi     = (oi / (1 + oi_change / 100)).astype(int)

    # ── Volume ───────────────────────────────────────────────────────────────
    avg_volume  = rng.integers(200_000, 5_000_000, n)
    volume      = (avg_volume * rng.uniform(0.5, 3.0, n)).astype(int)

    # ── Derivatives metrics ───────────────────────────────────────────────────
    # IV: realistic range 10–80 %
    iv  = rng.uniform(10, 80, n).round(2)
    # PCR: realistic range 0.4–2.5
    pcr = rng.uniform(0.4, 2.5, n).round(2)

    # ── ATR (Average True Range) ─────────────────────────────────────────────
    # ATR as a % of CMP, roughly 1–4 % for large-caps
    atr_pct = rng.uniform(0.01, 0.04, n)
    atr     = (cmp * atr_pct).round(2)

    # ── Support / Resistance ─────────────────────────────────────────────────
    # Support: 4–10 % below CMP  |  Resistance: 8–22 % above CMP
    # Wider resistance band ensures realistic headroom for 3× ATR targets.
    support    = (cmp * rng.uniform(0.90, 0.96, n)).round(2)
    resistance = (cmp * rng.uniform(1.08, 1.22, n)).round(2)

    df = pd.DataFrame({
        "Symbol":          NIFTY50_SYMBOLS,
        "CMP":             cmp,
        "Prev_Close":      prev_close,
        "VWAP":            vwap,
        "OI":              oi,
        "Prev_OI":         prev_oi,
        "OI_Change_Pct":   oi_change,
        "Volume":          volume,
        "Avg_Volume":      avg_volume,
        "IV":              iv,
        "PCR":             pcr,
        "ATR":             atr,
        "Support":         support,
        "Resistance":      resistance,
    })
    return df


# =============================================================================
#  2. INDICATOR CALCULATION ENGINE
# =============================================================================

def classify_oi_trend(price_chg: float, oi_chg: float) -> str:
    """
    Classic four-quadrant OI interpretation.

    Price↑ + OI↑  → Long Buildup   (bulls entering)
    Price↓ + OI↑  → Short Buildup  (bears entering)
    Price↑ + OI↓  → Short Covering (bears exiting)
    Price↓ + OI↓  → Long Unwinding (bulls exiting)
    """
    if price_chg >= 0 and oi_chg >= 0:
        return "🟢 Long Buildup"
    elif price_chg < 0 and oi_chg >= 0:
        return "🔴 Short Buildup"
    elif price_chg >= 0 and oi_chg < 0:
        return "🟡 Short Covering"
    else:
        return "🔵 Long Unwinding"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the raw data with all computed indicators."""
    raw = df.copy()

    # ── Price change % ─────────────────────────────────────────────────────
    raw["Price_Change_Pct"] = (
        (raw["CMP"] - raw["Prev_Close"]) / raw["Prev_Close"] * 100
    ).round(2)

    # ── OI Trend label ──────────────────────────────────────────────────────
    raw["Trend_OI"] = raw.apply(
        lambda r: classify_oi_trend(r["Price_Change_Pct"], r["OI_Change_Pct"]),
        axis=1,
    )

    # ── VWAP Signal ─────────────────────────────────────────────────────────
    raw["VWAP_Signal"] = np.where(raw["CMP"] > raw["VWAP"], "▲ Above VWAP", "▼ Below VWAP")

    # ── Volume Surge flag (current vol > 1.5× average) ──────────────────────
    raw["Vol_Surge"] = raw["Volume"] > (1.5 * raw["Avg_Volume"])

    return raw


# =============================================================================
#  3.  3:1 RISK-REWARD ENGINE
# =============================================================================

def calculate_rr_setups(
    df: pd.DataFrame,
    atr_multiplier: float,
    min_oi_change_pct: float,
    pcr_low: float  = 0.7,
    pcr_high: float = 1.5,
) -> pd.DataFrame:
    """
    Core screening logic — enforces a strict 3:1 Reward-to-Risk ratio.

    Parameters
    ----------
    df               : Enriched dataframe from add_indicators()
    atr_multiplier   : How many ATRs define the Stop-Loss distance.
    min_oi_change_pct: Minimum absolute OI change % to ensure liquidity.
    pcr_low / high   : PCR band outside which we flag reversal potential.

    Logic
    -----
    Entry = CMP  (market entry on signal confirmation)
    SL    = Entry  -  (ATR × atr_multiplier)           for LONG setups
    Target= Entry  +  (Risk × 3)
    RR_OK = Target must not exceed Resistance level
            (ensures price has room to reach target)
    """
    out = df.copy()

    # ── SL & Target calculation ──────────────────────────────────────────────
    out["Entry"]       = out["CMP"]
    sl_distance        = out["ATR"] * atr_multiplier          # risk quantum
    out["Stop_Loss"]   = (out["Entry"] - sl_distance).round(2)
    out["Target_3x"]   = (out["Entry"] + sl_distance * 3).round(2)  # 3:1 RR
    out["Risk_Per_Lot"] = sl_distance.round(2)

    # ── Screening filters ────────────────────────────────────────────────────

    # 1. OI momentum filter — discard low-activity symbols
    f_oi = out["OI_Change_Pct"].abs() >= min_oi_change_pct

    # 2. RR geometry filter — Resistance must sit BEYOND the Target.
    #    If Resistance > Target, price has structural headroom to reach the
    #    3× reward level without hitting a supply wall first.
    f_rr = out["Resistance"] >= out["Target_3x"]

    # 3. PCR extremes filter — only trade when derivatives sentiment aligns
    f_pcr = (out["PCR"] < pcr_low) | (out["PCR"] > pcr_high)

    # 4. SL must sit above Support (structural floor provides additional cushion)
    f_sl_support = out["Stop_Loss"] >= out["Support"] * 0.98

    # 5. Bullish OI bias — prefer Long Buildup or Short Covering
    f_bullish = out["Trend_OI"].isin(["🟢 Long Buildup", "🟡 Short Covering"])

    combined_filter = f_oi & f_rr & f_pcr & f_sl_support & f_bullish
    screened        = out[combined_filter].copy()

    # ── Final column curation ────────────────────────────────────────────────
    screened["Actual_RR"] = (
        (screened["Target_3x"] - screened["Entry"]) /
        (screened["Entry"]     - screened["Stop_Loss"])
    ).round(2)

    return screened.reset_index(drop=True)


# =============================================================================
#  4. POSITION SIZING
# =============================================================================

def position_sizing(entry: float, sl: float, capital: float, risk_pct: float = 1.0) -> dict:
    """
    Kelly-adjacent fixed-fractional position sizing.

    risk_pct : % of capital to risk per trade (default 1 %)
    Returns lot count estimate (assumes lot size 1 for simplicity;
    multiply by actual lot size in production).
    """
    risk_amount = capital * (risk_pct / 100)
    risk_per_unit = abs(entry - sl)
    if risk_per_unit == 0:
        return {"units": 0, "risk_amount": 0}
    units = int(risk_amount / risk_per_unit)
    return {
        "units":       units,
        "risk_amount": round(risk_amount, 2),
        "rr_amount":   round(risk_amount * 3, 2),   # potential reward
    }


# =============================================================================
#  5. STREAMLIT SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style='padding:0.6rem 0 1.2rem;'>
            <p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                      letter-spacing:0.12em;text-transform:uppercase;
                      color:#64748B;margin-bottom:4px;'>NSE F&O ENGINE</p>
            <h1 style='font-size:1.3rem;font-weight:700;margin:0;
                       background:linear-gradient(135deg,#00FF88,#3B82F6);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                Screener v2.0
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Capital ──────────────────────────────────────────────────────────────
    st.markdown("**💰 Capital Allocation**")
    capital = st.number_input(
        "Total Capital (₹)",
        min_value=10_000,
        max_value=50_000_000,
        value=500_000,
        step=10_000,
        help="Total trading capital available. Position sizing is based on 1% risk per trade.",
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Percentage of capital to risk on a single trade.",
    )

    st.divider()

    # ── ATR Multiplier ────────────────────────────────────────────────────────
    st.markdown("**📐 Stop-Loss Configuration**")
    atr_multiplier = st.slider(
        "ATR Multiplier (SL Distance)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="SL = Entry − (ATR × multiplier). Higher = wider stop, fewer false exits.",
    )

    st.divider()

    # ── OI Filter ────────────────────────────────────────────────────────────
    st.markdown("**📈 Open Interest Filters**")
    min_oi_change = st.slider(
        "Min |OI Change| (%)",
        min_value=1.0,
        max_value=25.0,
        value=5.0,
        step=0.5,
        help="Filter out symbols with insufficient OI momentum.",
    )

    st.divider()

    # ── PCR Thresholds ────────────────────────────────────────────────────────
    st.markdown("**⚖️ PCR Thresholds**")
    pcr_col1, pcr_col2 = st.columns(2)
    with pcr_col1:
        pcr_low = st.number_input("PCR Low", value=0.70, step=0.05, min_value=0.1, max_value=1.0)
    with pcr_col2:
        pcr_high = st.number_input("PCR High", value=1.50, step=0.05, min_value=1.0, max_value=3.0)

    st.divider()

    # ── Data refresh ─────────────────────────────────────────────────────────
    refresh_seed = st.number_input(
        "Data Seed (mock variation)",
        min_value=1,
        max_value=9999,
        value=42,
        help="Change seed to generate a different mock dataset.",
    )
    if st.button("🔄  Refresh Data"):
        st.cache_data.clear()

    st.divider()
    st.markdown(
        "<p style='font-size:0.65rem;color:#374151;text-align:center;"
        "font-family:JetBrains Mono,monospace;'>"
        "⚠ Mock data only. Not financial advice.</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  6. MAIN DASHBOARD
# =============================================================================

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:0.2rem;'>
        <span style='font-size:2rem;'>📊</span>
        <div>
            <h1 style='margin:0;font-size:1.8rem;'>F&O High-Probability Screener</h1>
            <p style='margin:0;font-size:0.8rem;color:#64748B;font-family:JetBrains Mono,monospace;
                      letter-spacing:0.06em;'>NSE FUTURES & OPTIONS  ·  3:1 RISK-REWARD ENGINE  ·  LIVE MOCK DATA</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Run pipeline ──────────────────────────────────────────────────────────────
try:
    raw_df      = generate_mock_fo_data(seed=int(refresh_seed))
    enriched_df = add_indicators(raw_df)
    screened_df = calculate_rr_setups(
        enriched_df,
        atr_multiplier   = atr_multiplier,
        min_oi_change_pct= min_oi_change,
        pcr_low          = pcr_low,
        pcr_high         = pcr_high,
    )
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# ── Derived stats ─────────────────────────────────────────────────────────────
total_universe = len(enriched_df)
total_setups   = len(screened_df)
bullish_setups = len(screened_df[screened_df["Trend_OI"].str.contains("Long Buildup")])
covering_setups= len(screened_df[screened_df["Trend_OI"].str.contains("Short Covering")])

avg_iv  = enriched_df["IV"].mean()
avg_pcr = enriched_df["PCR"].mean()

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("## 📡  Market Pulse")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "🎯  Setups Found",
        f"{total_setups}",
        delta=f"of {total_universe} scanned",
    )
with k2:
    st.metric(
        "🟢  Long Buildup",
        f"{bullish_setups}",
        delta="bullish OI accumulation",
    )
with k3:
    st.metric(
        "🟡  Short Covering",
        f"{covering_setups}",
        delta="bear exit momentum",
    )
with k4:
    st.metric(
        "📊  Avg IV",
        f"{avg_iv:.1f}%",
        delta="universe implied vol",
    )
with k5:
    st.metric(
        "⚖️  Avg PCR",
        f"{avg_pcr:.2f}",
        delta="put-call ratio",
    )

st.divider()

# =============================================================================
#  7. ACTIONABLE SETUPS TABLE
# =============================================================================

st.markdown("## 🔍  Actionable 3:1 RR Setups")

if screened_df.empty:
    st.warning(
        "⚠️  No setups pass the current filters. "
        "Try reducing Min |OI Change| or adjusting PCR thresholds in the sidebar."
    )
else:
    # ── Build display dataframe ───────────────────────────────────────────────
    pos_sizes = screened_df.apply(
        lambda r: position_sizing(r["Entry"], r["Stop_Loss"], capital, risk_pct),
        axis=1,
    )
    screened_df["Units"]       = [p["units"]       for p in pos_sizes]
    screened_df["Risk_₹"]      = [p["risk_amount"]  for p in pos_sizes]
    screened_df["Reward_₹"]    = [p["rr_amount"]    for p in pos_sizes]

    display_cols = {
        "Symbol":        "Symbol",
        "Trend_OI":      "OI Trend",
        "CMP":           "CMP (₹)",
        "Entry":         "Entry (₹)",
        "Stop_Loss":     "Stop-Loss (₹)",
        "Target_3x":     "Target 3:1 (₹)",
        "Actual_RR":     "Actual R:R",
        "PCR":           "PCR",
        "IV":            "IV (%)",
        "VWAP_Signal":   "VWAP Signal",
        "OI_Change_Pct": "OI Δ (%)",
        "Units":         "Units",
        "Risk_₹":        "Risk (₹)",
        "Reward_₹":      "Reward (₹)",
    }
    display_df = screened_df[list(display_cols.keys())].rename(columns=display_cols)

    # ── Pandas Styler ─────────────────────────────────────────────────────────
    def style_table(df: pd.DataFrame):
        """
        Apply conditional colour-coding to the display dataframe.
        Green   → Target column, Long Buildup rows
        Red     → Stop-Loss column
        Yellow  → PCR extremes
        """
        styled = df.style

        # Row-level background for OI trend
        def row_bg(row):
            styles = [""] * len(row)
            trend  = row["OI Trend"]
            if "Long Buildup"   in trend:
                styles = ["background-color: rgba(0,255,136,0.05)"] * len(row)
            elif "Short Covering" in trend:
                styles = ["background-color: rgba(255,215,0,0.05)"]  * len(row)
            return styles

        styled = styled.apply(row_bg, axis=1)

        # Column-specific formatting
        styled = styled.format({
            "CMP (₹)":        "₹{:,.2f}",
            "Entry (₹)":      "₹{:,.2f}",
            "Stop-Loss (₹)":  "₹{:,.2f}",
            "Target 3:1 (₹)": "₹{:,.2f}",
            "Actual R:R":     "{:.2f}x",
            "PCR":            "{:.2f}",
            "IV (%)":         "{:.1f}%",
            "OI Δ (%)":       "{:+.2f}%",
            "Risk (₹)":       "₹{:,.0f}",
            "Reward (₹)":     "₹{:,.0f}",
        })

        # Target column → green text
        styled = styled.map(
            lambda v: "color: #00FF88; font-weight:600;",
            subset=["Target 3:1 (₹)"],
        )
        # Stop-Loss column → red text
        styled = styled.map(
            lambda v: "color: #FF3B5C; font-weight:600;",
            subset=["Stop-Loss (₹)"],
        )
        # Actual R:R → bold
        styled = styled.map(
            lambda v: "color: #3B82F6; font-weight:700; font-family:JetBrains Mono,monospace;",
            subset=["Actual R:R"],
        )
        # PCR extremes → yellow highlight
        styled = styled.map(
            lambda v: "color: #FFD700; font-weight:600;" if v < 0.7 or v > 1.5 else "",
            subset=["PCR"],
        )

        styled = styled.set_table_styles([
            {"selector": "thead th",
             "props": [
                 ("background-color", "#111827"),
                 ("color", "#64748B"),
                 ("font-family", "JetBrains Mono, monospace"),
                 ("font-size", "0.7rem"),
                 ("letter-spacing", "0.06em"),
                 ("text-transform", "uppercase"),
                 ("border-bottom", "1px solid #1E293B"),
             ]},
            {"selector": "tbody td",
             "props": [
                 ("background-color", "#141C2E"),
                 ("color", "#E2E8F0"),
                 ("font-size", "0.82rem"),
                 ("border-bottom", "1px solid #1E293B"),
             ]},
            {"selector": "tbody tr:hover td",
             "props": [("background-color", "#1E293B")]},
        ])
        return styled

    st.dataframe(
        style_table(display_df),
        use_container_width=True,
        height=min(40 + 38 * len(display_df), 600),
    )

    # ── Download button ───────────────────────────────────────────────────────
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Export Setups CSV",
        data=csv_data,
        file_name="fo_screener_setups.csv",
        mime="text/csv",
    )

st.divider()

# =============================================================================
#  8. FULL UNIVERSE TABLE (collapsed)
# =============================================================================

with st.expander("📋  Full Universe Snapshot  (all symbols, pre-filter)", expanded=False):
    universe_display = enriched_df[[
        "Symbol", "CMP", "Prev_Close", "Price_Change_Pct",
        "VWAP", "VWAP_Signal", "OI_Change_Pct", "Trend_OI",
        "PCR", "IV", "ATR",
    ]].rename(columns={
        "Price_Change_Pct": "Price Δ (%)",
        "OI_Change_Pct":    "OI Δ (%)",
        "Trend_OI":         "OI Trend",
        "VWAP_Signal":      "VWAP",
    }).style.format({
        "CMP":         "₹{:,.2f}",
        "Prev_Close":  "₹{:,.2f}",
        "Price Δ (%)": "{:+.2f}%",
        "VWAP":        "₹{:,.2f}",
        "OI Δ (%)":    "{:+.2f}%",
        "PCR":         "{:.2f}",
        "IV":          "{:.1f}%",
        "ATR":         "{:.2f}",
    }).map(
        lambda v: "color: #00FF88;" if isinstance(v, str) and "+" in str(v) else
                  "color: #FF3B5C;" if isinstance(v, str) and "-" in str(v) else "",
    )
    st.dataframe(universe_display, use_container_width=True, height=420)

# =============================================================================
#  9. METHODOLOGY EXPLAINER
# =============================================================================

with st.expander("📚  Methodology & Signal Logic", expanded=False):
    st.markdown(
        """
        ### OI Trend Classification
        | Price Δ | OI Δ | Signal          | Interpretation                      |
        |---------|------|-----------------|-------------------------------------|
        | ↑       | ↑    | Long Buildup    | Fresh longs being added — bullish    |
        | ↓       | ↑    | Short Buildup   | Fresh shorts being added — bearish   |
        | ↑       | ↓    | Short Covering  | Shorts exiting — bullish momentum    |
        | ↓       | ↓    | Long Unwinding  | Longs exiting — bearish momentum     |

        ### 3:1 RR Calculation
        ```
        Entry     = CMP  (entry on breakout/signal candle close)
        Risk      = ATR × ATR_Multiplier
        Stop-Loss = Entry − Risk
        Target    = Entry + (Risk × 3)
        Valid if  : Target ≤ Resistance × 1.01  AND  Stop-Loss ≥ Support × 0.98
        ```

        ### PCR Interpretation
        - **PCR > 1.5** → Excessive Put buying → bearish sentiment → contrarian LONG opportunity
        - **PCR < 0.7** → Excessive Call buying → complacency → contrarian SHORT or protective hedge
        - **PCR 0.7–1.5** → Neutral zone → filtered out by screener

        ### Position Sizing
        ```
        Risk per Trade = Capital × risk_pct %
        Units          = Risk_per_Trade / |Entry − Stop-Loss|
        Max Reward     = Risk_per_Trade × 3
        ```
        > *All data is simulated. This tool is for educational purposes only.*
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center;padding:2rem 0 1rem;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                  color:#1E293B;letter-spacing:0.1em;'>
            F&O SCREENER  ·  MOCK DATA  ·  NOT FINANCIAL ADVICE
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
