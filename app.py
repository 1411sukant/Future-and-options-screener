# =============================================================================
#  F&O HIGH-PROBABILITY SCREENER  |  app.py  v2.1
#  Stack : Streamlit · Pandas · NumPy
#  Run   : streamlit run app.py
#  Fix   : All Pandas Styler removed → replaced with st.column_config
#          (Styler causes KeyError on Streamlit Cloud with newer Pandas)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Screener | High-Probability Setups",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #0A0E17 !important;
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B !important;
}
[data-testid="stMetric"] {
    background: #141C2E;
    border: 1px solid #1E293B;
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
    background: linear-gradient(90deg, #00FF88, #3B82F6);
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #00FF88 !important;
}
hr { border-color: #1E293B !important; }
h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; }
h2 { color: #3B82F6 !important; font-size: 1.0rem !important;
     text-transform: uppercase; letter-spacing: 0.06em; }
.streamlit-expanderHeader {
    background: #141C2E !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #00FF88 !important;
    color: #00FF88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 6px;
}
.stDownloadButton > button:hover {
    background: #00FF88 !important;
    color: #0A0E17 !important;
}
.stButton > button {
    background: transparent !important;
    border: 1px solid #3B82F6 !important;
    color: #3B82F6 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 6px;
    width: 100%;
}
.stButton > button:hover {
    background: #3B82F6 !important;
    color: #0A0E17 !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  1. MOCK DATA GENERATOR
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

@st.cache_data(ttl=300)
def generate_mock_fo_data(seed: int = 42) -> pd.DataFrame:
    """
    Deterministic mock NSE F&O snapshot.
    Change seed in sidebar to simulate different market sessions.
    """
    rng = np.random.default_rng(seed)
    n   = len(NIFTY50_SYMBOLS)

    cmp        = rng.uniform(200, 4000, n).round(2)
    prev_close = (cmp * rng.uniform(0.95, 1.05, n)).round(2)
    vwap       = (cmp * rng.uniform(0.98, 1.02, n)).round(2)

    oi        = rng.integers(500_000, 50_000_000, n)
    oi_change = rng.normal(0.0, 8.0, n).round(2)   # % — fat-tailed
    prev_oi   = (oi / (1 + oi_change / 100)).astype(int)

    avg_volume = rng.integers(200_000, 5_000_000, n)
    volume     = (avg_volume * rng.uniform(0.5, 3.0, n)).astype(int)

    iv  = rng.uniform(10, 80, n).round(2)
    pcr = rng.uniform(0.4, 2.5, n).round(2)

    # ATR = 1–4 % of CMP (realistic NSE large-cap range)
    atr = (cmp * rng.uniform(0.01, 0.04, n)).round(2)

    # Support 4–10 % below | Resistance 8–22 % above (wide enough for 3× targets)
    support    = (cmp * rng.uniform(0.90, 0.96, n)).round(2)
    resistance = (cmp * rng.uniform(1.08, 1.22, n)).round(2)

    return pd.DataFrame({
        "Symbol":        NIFTY50_SYMBOLS,
        "CMP":           cmp,
        "Prev_Close":    prev_close,
        "VWAP":          vwap,
        "OI":            oi,
        "Prev_OI":       prev_oi,
        "OI_Change_Pct": oi_change,
        "Volume":        volume,
        "Avg_Volume":    avg_volume,
        "IV":            iv,
        "PCR":           pcr,
        "ATR":           atr,
        "Support":       support,
        "Resistance":    resistance,
    })


# =============================================================================
#  2. INDICATOR ENGINE
# =============================================================================

def classify_oi_trend(price_chg: float, oi_chg: float) -> str:
    """
    Four-quadrant OI interpretation:
      Price↑ OI↑ → Long Buildup   (bulls entering)
      Price↓ OI↑ → Short Buildup  (bears entering)
      Price↑ OI↓ → Short Covering (bears exiting)
      Price↓ OI↓ → Long Unwinding (bulls exiting)
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
    """Enrich raw data with all computed F&O indicators."""
    out = df.copy()

    out["Price_Change_Pct"] = (
        (out["CMP"] - out["Prev_Close"]) / out["Prev_Close"] * 100
    ).round(2)

    out["Trend_OI"] = out.apply(
        lambda r: classify_oi_trend(r["Price_Change_Pct"], r["OI_Change_Pct"]),
        axis=1,
    )

    out["VWAP_Signal"] = np.where(out["CMP"] > out["VWAP"], "▲ Above", "▼ Below")
    out["Vol_Surge"]   = out["Volume"] > (1.5 * out["Avg_Volume"])

    return out


# =============================================================================
#  3. 3:1 RISK-REWARD ENGINE
# =============================================================================

def calculate_rr_setups(
    df: pd.DataFrame,
    atr_multiplier: float,
    min_oi_change_pct: float,
    pcr_low: float  = 0.70,
    pcr_high: float = 1.50,
) -> pd.DataFrame:
    """
    Screens for strict 3:1 Reward-to-Risk setups.

    SL Distance = ATR × atr_multiplier
    Entry       = CMP
    Stop-Loss   = Entry − SL Distance
    Target      = Entry + (SL Distance × 3)

    Five-layer validity gate:
      1. |OI Change %| >= threshold      — liquidity confirmation
      2. Resistance >= Target            — structural headroom to target
      3. PCR outside neutral band        — sentiment extreme signal
      4. Stop-Loss >= Support × 0.98    — SL on structural floor
      5. OI Trend = Long Buildup / Short Covering — bullish bias
    """
    out = df.copy()

    out["Entry"]     = out["CMP"]
    sl_dist          = out["ATR"] * atr_multiplier
    out["Stop_Loss"] = (out["Entry"] - sl_dist).round(2)
    out["Target_3x"] = (out["Entry"] + sl_dist * 3).round(2)

    f1 = out["OI_Change_Pct"].abs() >= min_oi_change_pct
    f2 = out["Resistance"] >= out["Target_3x"]
    f3 = (out["PCR"] < pcr_low) | (out["PCR"] > pcr_high)
    f4 = out["Stop_Loss"] >= out["Support"] * 0.98
    f5 = out["Trend_OI"].isin(["🟢 Long Buildup", "🟡 Short Covering"])

    screened = out[f1 & f2 & f3 & f4 & f5].copy()

    screened["Actual_RR"] = (
        (screened["Target_3x"] - screened["Entry"]) /
        (screened["Entry"]     - screened["Stop_Loss"])
    ).round(2)

    return screened.reset_index(drop=True)


# =============================================================================
#  4. POSITION SIZING
# =============================================================================

def position_sizing(entry: float, sl: float, capital: float, risk_pct: float) -> dict:
    """Fixed-fractional: Units = (Capital × risk%) / |Entry − SL|"""
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
#  5. SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding:0.4rem 0 1rem;'>
        <p style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;
                  color:#64748B;margin:0;'>NSE F&O ENGINE</p>
        <h1 style='font-size:1.25rem;font-weight:700;margin:4px 0 0;
                   background:linear-gradient(135deg,#00FF88,#3B82F6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Screener v2.1
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**💰 Capital**")
    capital  = st.number_input("Total Capital (₹)",
                                min_value=10_000, max_value=50_000_000,
                                value=500_000, step=10_000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.1,
                          help="% of capital risked on each setup.")
    st.divider()

    st.markdown("**📐 Stop-Loss**")
    atr_multiplier = st.slider("ATR Multiplier", 0.5, 3.0, 1.5, 0.1,
                                 help="SL = Entry − (ATR × multiplier)")
    st.divider()

    st.markdown("**📈 OI Filter**")
    min_oi_change = st.slider("Min |OI Change| (%)", 1.0, 25.0, 5.0, 0.5)
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

    seed = st.number_input("Data Seed", min_value=1, max_value=9999, value=42,
                            help="Change to simulate a different market session.")
    if st.button("🔄  Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.62rem;color:#374151;text-align:center;"
        "font-family:JetBrains Mono,monospace;'>"
        "⚠ Mock data · Not financial advice</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  6. PIPELINE — run after sidebar inputs are bound
# =============================================================================

try:
    raw_df      = generate_mock_fo_data(seed=int(seed))
    enriched_df = add_indicators(raw_df)
    screened_df = calculate_rr_setups(
        enriched_df,
        atr_multiplier    = atr_multiplier,
        min_oi_change_pct = min_oi_change,
        pcr_low           = pcr_low,
        pcr_high          = pcr_high,
    )
except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.stop()

total_setups    = len(screened_df)
bullish_setups  = int(screened_df["Trend_OI"].str.contains("Long Buildup",   na=False).sum())
covering_setups = int(screened_df["Trend_OI"].str.contains("Short Covering", na=False).sum())
avg_iv  = float(enriched_df["IV"].mean())
avg_pcr = float(enriched_df["PCR"].mean())


# =============================================================================
#  7. HEADER
# =============================================================================

st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:0.2rem;'>
    <span style='font-size:2rem;'>📊</span>
    <div>
        <h1 style='margin:0;font-size:1.75rem;'>F&O High-Probability Screener</h1>
        <p style='margin:0;font-size:0.75rem;color:#64748B;
                  font-family:JetBrains Mono,monospace;letter-spacing:0.06em;'>
            NSE FUTURES &amp; OPTIONS · 3:1 RISK-REWARD ENGINE · MOCK DATA
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# =============================================================================
#  8. KPI METRICS
# =============================================================================

st.markdown("## 📡  Market Pulse")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("🎯 Setups Found",   str(total_setups),
              delta=f"of {len(enriched_df)} scanned")
with k2:
    st.metric("🟢 Long Buildup",   str(bullish_setups),
              delta="bullish OI accumulation")
with k3:
    st.metric("🟡 Short Covering", str(covering_setups),
              delta="bear exit momentum")
with k4:
    st.metric("📊 Avg IV",         f"{avg_iv:.1f}%",
              delta="universe implied vol")
with k5:
    st.metric("⚖️ Avg PCR",        f"{avg_pcr:.2f}",
              delta="put-call ratio")

st.divider()


# =============================================================================
#  9. ACTIONABLE SETUPS TABLE  — st.column_config only, zero Styler
# =============================================================================

st.markdown("## 🔍  Actionable 3:1 RR Setups")

if screened_df.empty:
    st.warning(
        "⚠️ No setups pass the current filters. "
        "Try lowering Min |OI Change| or widening the PCR range in the sidebar."
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

    # Build display dataframe with only plain Python-native types
    display_cols = ["Symbol","Trend_OI","CMP","Entry","Stop_Loss",
                    "Target_3x","Actual_RR","PCR","IV",
                    "VWAP_Signal","OI_Change_Pct",
                    "Units","Risk_INR","Reward_INR"]
    display_df = screened_df[display_cols].copy()

    # Explicit dtype coercion — prevents Arrow serialization KeyError
    float_cols = ["CMP","Entry","Stop_Loss","Target_3x","Actual_RR",
                  "PCR","IV","OI_Change_Pct","Risk_INR","Reward_INR"]
    for c in float_cols:
        display_df[c] = display_df[c].astype("float64")
    display_df["Units"] = display_df["Units"].astype("int64")

    # Streamlit-native column config (no Styler, no Arrow issues)
    col_cfg = {
        "Symbol":       st.column_config.TextColumn("Symbol",          width="small"),
        "Trend_OI":     st.column_config.TextColumn("OI Trend",        width="medium"),
        "CMP":          st.column_config.NumberColumn("CMP (₹)",       format="₹%.2f",  width="small"),
        "Entry":        st.column_config.NumberColumn("Entry (₹)",     format="₹%.2f",  width="small"),
        "Stop_Loss":    st.column_config.NumberColumn("Stop-Loss (₹)", format="₹%.2f",  width="small"),
        "Target_3x":    st.column_config.NumberColumn("Target 3:1 (₹)",format="₹%.2f", width="small"),
        "Actual_RR":    st.column_config.NumberColumn("R:R",           format="%.2fx",  width="small"),
        "PCR":          st.column_config.NumberColumn("PCR",           format="%.2f",   width="small"),
        "IV":           st.column_config.NumberColumn("IV (%)",        format="%.1f%%", width="small"),
        "VWAP_Signal":  st.column_config.TextColumn("VWAP",            width="small"),
        "OI_Change_Pct":st.column_config.NumberColumn("OI Δ (%)",      format="%.2f%%", width="small"),
        "Units":        st.column_config.NumberColumn("Units",         format="%d",     width="small"),
        "Risk_INR":     st.column_config.NumberColumn("Risk (₹)",      format="₹%.0f",  width="small"),
        "Reward_INR":   st.column_config.NumberColumn("Reward (₹)",    format="₹%.0f",  width="small"),
    }

    st.dataframe(
        display_df,
        column_config       = col_cfg,
        use_container_width = True,
        hide_index          = True,
        height              = min(80 + 38 * len(display_df), 600),
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇  Export Setups CSV",
        data      = csv,
        file_name = "fo_screener_setups.csv",
        mime      = "text/csv",
    )

st.divider()


# =============================================================================
#  10. FULL UNIVERSE TABLE
# =============================================================================

with st.expander("📋  Full Universe Snapshot (all symbols, pre-filter)", expanded=False):

    uni = enriched_df[[
        "Symbol","CMP","Prev_Close","Price_Change_Pct",
        "VWAP","VWAP_Signal","OI_Change_Pct","Trend_OI",
        "PCR","IV","ATR","Support","Resistance",
    ]].copy()

    # Explicit dtype coercion — same fix applied here
    for c in ["CMP","Prev_Close","Price_Change_Pct","VWAP",
              "OI_Change_Pct","PCR","IV","ATR","Support","Resistance"]:
        uni[c] = uni[c].astype("float64")

    uni_cfg = {
        "Symbol":           st.column_config.TextColumn("Symbol",     width="small"),
        "CMP":              st.column_config.NumberColumn("CMP (₹)",  format="₹%.2f"),
        "Prev_Close":       st.column_config.NumberColumn("Prev Close",format="₹%.2f"),
        "Price_Change_Pct": st.column_config.NumberColumn("Price Δ%", format="%.2f%%"),
        "VWAP":             st.column_config.NumberColumn("VWAP",     format="₹%.2f"),
        "VWAP_Signal":      st.column_config.TextColumn("VWAP Pos",   width="small"),
        "OI_Change_Pct":    st.column_config.NumberColumn("OI Δ%",    format="%.2f%%"),
        "Trend_OI":         st.column_config.TextColumn("OI Trend",   width="medium"),
        "PCR":              st.column_config.NumberColumn("PCR",      format="%.2f"),
        "IV":               st.column_config.NumberColumn("IV%",      format="%.1f%%"),
        "ATR":              st.column_config.NumberColumn("ATR",      format="₹%.2f"),
        "Support":          st.column_config.NumberColumn("Support",  format="₹%.2f"),
        "Resistance":       st.column_config.NumberColumn("Resistance",format="₹%.2f"),
    }

    st.dataframe(
        uni,
        column_config       = uni_cfg,
        use_container_width = True,
        hide_index          = True,
        height              = 420,
    )

st.divider()


# =============================================================================
#  11. METHODOLOGY
# =============================================================================

with st.expander("📚  Methodology & Signal Logic", expanded=False):
    st.markdown("""
### OI Trend Classification
| Price Δ | OI Δ | Signal | Interpretation |
|---|---|---|---|
| ↑ | ↑ | 🟢 Long Buildup | Fresh longs added — bullish |
| ↓ | ↑ | 🔴 Short Buildup | Fresh shorts added — bearish |
| ↑ | ↓ | 🟡 Short Covering | Shorts exiting — bullish momentum |
| ↓ | ↓ | 🔵 Long Unwinding | Longs exiting — bearish momentum |

### 3:1 RR Calculation
```
Entry      = CMP
Risk       = ATR × ATR_Multiplier
Stop-Loss  = Entry − Risk
Target     = Entry + (Risk × 3)

Valid if:
  Resistance >= Target           → structural headroom confirmed
  Stop-Loss  >= Support × 0.98  → SL anchored on support floor
```

### PCR Interpretation
- **PCR > 1.5** → Excessive put buying → contrarian LONG signal
- **PCR < 0.7** → Excessive call buying → hedge or contrarian SHORT
- **PCR 0.7–1.5** → Neutral zone → filtered out by screener

### Position Sizing
```
Risk Amount = Capital × risk_pct%
Units       = Risk Amount / |Entry − Stop-Loss|
Max Reward  = Risk Amount × 3
```
> ⚠ All data is simulated for educational purposes only. Not financial advice.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 0.5rem;'>
    <p style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
              color:#1E293B;letter-spacing:0.1em;'>
        F&O SCREENER v2.1 · MOCK DATA · NOT FINANCIAL ADVICE
    </p>
</div>
""", unsafe_allow_html=True)
