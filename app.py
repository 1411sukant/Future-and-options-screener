# =============================================================================
#  NSE F&O SIGNAL SCREENER  v5.0
#  • Search any NSE stock → Full Technical + Fundamental Analysis
#  • Real-time IST clock  • Intraday price (5-min delay)
#  • Nifty / Bank Nifty index signals
#  • BUY / SELL / HOLD with Entry · Target · Stop-Loss
#  Run : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    LIVE = True
except ModuleNotFoundError:
    LIVE = False

# ── IST helper (no external dependency) ──────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> str:
    return datetime.now(IST).strftime("%d %b %Y  %I:%M:%S %p  IST")
def time_ist() -> str:
    return datetime.now(IST).strftime("%I:%M %p IST")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Signal Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #05080F !important;
    color: #E2E8F0 !important;
}
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-track{background:#0D1117;}
::-webkit-scrollbar-thumb{background:#1E293B;border-radius:4px;}

section[data-testid="stSidebar"]{background:#0D1117 !important;border-right:1px solid #1E293B !important;}
hr{border-color:#1E293B !important;margin:0.4rem 0 !important;}

/* Metric */
[data-testid="stMetric"]{
    background:linear-gradient(135deg,#0D1117,#111827);
    border:1px solid #1E293B;border-radius:14px;padding:1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]{font-size:.62rem !important;font-weight:600 !important;
    letter-spacing:.12em !important;text-transform:uppercase !important;color:#64748B !important;}
[data-testid="stMetricValue"]{font-size:1.6rem !important;font-weight:800 !important;
    font-family:'JetBrains Mono',monospace !important;color:#F8FAFC !important;}
[data-testid="stMetricDelta"]{font-size:.72rem !important;}

/* Search input */
div[data-testid="stTextInput"] input {
    background:#0D1117 !important;border:1.5px solid #6366F1 !important;
    border-radius:14px !important;color:#F8FAFC !important;
    font-size:1rem !important;font-family:'JetBrains Mono',monospace !important;
    padding:0.75rem 1.2rem !important;
}
div[data-testid="stTextInput"] input:focus{border-color:#818CF8 !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.2) !important;}
div[data-testid="stTextInput"] label{font-size:.65rem !important;font-weight:600 !important;
    letter-spacing:.12em !important;text-transform:uppercase !important;color:#6366F1 !important;}

/* Buttons */
div[data-testid="stButton"] > button{
    background:linear-gradient(135deg,#6366F1,#8B5CF6) !important;
    border:none !important;color:#fff !important;font-weight:700 !important;
    font-size:.9rem !important;padding:.65rem 1.5rem !important;
    border-radius:12px !important;box-shadow:0 4px 16px rgba(99,102,241,.35) !important;
    transition:all .2s !important;
}
div[data-testid="stButton"] > button:hover{transform:translateY(-1px) !important;
    box-shadow:0 6px 22px rgba(99,102,241,.55) !important;}

/* Expander */
.streamlit-expanderHeader{background:#0D1117 !important;border:1px solid #1E293B !important;
    border-radius:10px !important;font-size:.78rem !important;color:#64748B !important;}

/* Selectbox */
div[data-testid="stSelectbox"] label{font-size:.62rem !important;text-transform:uppercase !important;
    letter-spacing:.1em !important;color:#64748B !important;}

/* Download */
.stDownloadButton>button{background:transparent !important;border:1px solid #6366F1 !important;
    color:#6366F1 !important;font-size:.72rem !important;border-radius:8px !important;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  UNIVERSE
# =============================================================================
INDICES = {
    "^NSEI":      {"name":"NIFTY 50",      "emoji":"🔵","short":"NIFTY"},
    "^NSEBANK":   {"name":"BANK NIFTY",    "emoji":"🏦","short":"BANKNIFTY"},
    "^CNXIT":     {"name":"NIFTY IT",      "emoji":"💻","short":"NIFTYIT"},
    "^CNXPHARMA": {"name":"NIFTY PHARMA",  "emoji":"💊","short":"NIFTYPHARMA"},
    "^CNXAUTO":   {"name":"NIFTY AUTO",    "emoji":"🚗","short":"NIFTYAUTO"},
    "^CNXMIDCAP": {"name":"MIDCAP 100",    "emoji":"📊","short":"MIDCAP"},
}

FO_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY",
    "SBIN","AXISBANK","KOTAKBANK","BAJFINANCE","TITAN",
    "MARUTI","LT","TATAMOTORS","SUNPHARMA","WIPRO",
    "HCLTECH","ONGC","ADANIENT","TATASTEEL","APOLLOHOSP",
]

# Search aliases → yfinance ticker
SEARCH_ALIASES = {
    "NIFTY":"^NSEI","NIFTY50":"^NSEI","NIFTY 50":"^NSEI",
    "BANKNIFTY":"^NSEBANK","BANK NIFTY":"^NSEBANK","BANKEX":"^NSEBANK",
    "NIFTYIT":"^CNXIT","NIFTY IT":"^CNXIT",
    "NIFTYPHARMA":"^CNXPHARMA","NIFTYAUTO":"^CNXAUTO",
    "MIDCAP":"^CNXMIDCAP",
}


# =============================================================================
#  TECHNICAL INDICATORS
# =============================================================================
def rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return (100 - 100/(1+g/l.replace(0,np.nan))).fillna(50)

def macd(close):
    e12 = close.ewm(span=12,adjust=False).mean()
    e26 = close.ewm(span=26,adjust=False).mean()
    m   = e12-e26; s = m.ewm(span=9,adjust=False).mean()
    return m, s, m-s

def bollinger(close, p=20):
    sma = close.rolling(p).mean()
    std = close.rolling(p).std()
    return sma+2*std, sma, sma-2*std

def atr(high, low, close, p=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return float(tr.rolling(p).mean().dropna().iloc[-1])

def stochastic(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k   = 100*(close-lowest)/(highest-lowest+1e-9)
    pct_d   = pct_k.rolling(d).mean()
    return pct_k, pct_d


# =============================================================================
#  SIGNAL ENGINE
# =============================================================================
def generate_signal(hist: pd.DataFrame) -> dict | None:
    """
    Six-indicator signal engine → score -8 to +8
    ≥5 STRONG BUY | 2-4 BUY | -1–1 HOLD | -2–-4 SELL | ≤-5 STRONG SELL
    """
    if len(hist) < 35:
        return None

    c, h, l, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]

    _rsi              = rsi(c)
    _macd, _sig, _hist= macd(c)
    e20               = c.ewm(span=20,adjust=False).mean()
    e50               = c.ewm(span=50,adjust=False).mean()
    e200              = c.ewm(span=200,adjust=False).mean() if len(c)>=200 else e50
    bbu,bbm,bbl       = bollinger(c)
    _atr              = atr(h,l,c)
    _k, _d            = stochastic(h,l,c)

    # Current snapshot
    cv  = float(c.iloc[-1]);   pv = float(c.iloc[-2])
    r   = float(_rsi.iloc[-1])
    m   = float(_macd.iloc[-1]); ms = float(_sig.iloc[-1])
    pm  = float(_macd.iloc[-3]); ps = float(_sig.iloc[-3])
    e2  = float(e20.iloc[-1]);   e5 = float(e50.iloc[-1])
    e200v=float(e200.iloc[-1])
    bu  = float(bbu.iloc[-1]);   bl = float(bbl.iloc[-1])
    sk  = float(_k.iloc[-1]);    sd = float(_d.iloc[-1])
    vol_avg = float(v.iloc[-20:].mean()); vol_cur = float(v.iloc[-1])

    score   = 0
    reasons = []
    tech_details = {}

    # 1. RSI
    tech_details["RSI"] = round(r,1)
    if r<=30:   score+=2; reasons.append(f"RSI Oversold ({r:.0f}) — Strong reversal zone 🔥")
    elif r<=42: score+=1; reasons.append(f"RSI Bullish zone ({r:.0f})")
    elif r>=70: score-=2; reasons.append(f"RSI Overbought ({r:.0f}) — Caution ⚠️")
    elif r>=58: score-=1; reasons.append(f"RSI Bearish zone ({r:.0f})")
    else:       reasons.append(f"RSI Neutral ({r:.0f})")

    # 2. MACD
    tech_details["MACD"] = "Bullish" if m>ms else "Bearish"
    if m>ms and pm<ps:   score+=2; reasons.append("MACD Fresh Bullish Crossover ✅")
    elif m>ms:           score+=1; reasons.append("MACD Above Signal (Bullish)")
    elif m<ms and pm>ps: score-=2; reasons.append("MACD Fresh Bearish Crossover ❌")
    else:                score-=1; reasons.append("MACD Below Signal (Bearish)")

    # 3. EMA Trend
    if cv>e2>e5:     score+=2; reasons.append("Price > EMA20 > EMA50 (Strong Uptrend) 📈")
    elif cv>e2:      score+=1; reasons.append("Price Above EMA20 (Uptrend)")
    elif cv<e2<e5:   score-=2; reasons.append("Price < EMA20 < EMA50 (Strong Downtrend) 📉")
    elif cv<e2:      score-=1; reasons.append("Price Below EMA20 (Downtrend)")
    tech_details["EMA Trend"] = "Uptrend" if cv>e2 else "Downtrend"

    # 4. Bollinger Band
    bb_pos = round((cv-bl)/(bu-bl)*100,1) if bu!=bl else 50
    tech_details["BB Position"] = f"{bb_pos:.0f}%"
    if cv<=bl:       score+=1; reasons.append("At/Below Lower BB — Oversold reversal zone")
    elif cv>=bu:     score-=1; reasons.append("At/Above Upper BB — Overbought zone")

    # 5. Stochastic
    tech_details["Stoch %K"] = round(sk,1)
    if sk<20 and sk>sd:  score+=1; reasons.append(f"Stochastic Oversold Crossup ({sk:.0f})")
    elif sk>80 and sk<sd:score-=1; reasons.append(f"Stochastic Overbought ({sk:.0f})")

    # 6. Volume
    if vol_cur>1.8*vol_avg and cv>pv:   score+=1; reasons.append("Volume Surge + Price Rising 📊")
    elif vol_cur>1.8*vol_avg and cv<pv: score-=1; reasons.append("Volume Surge + Price Falling 📊")
    tech_details["Volume"] = f"{vol_cur/vol_avg:.1f}x avg"

    # Final signal
    if score>=5:    sig,col,emj = "STRONG BUY",  "#00C853","🚀"
    elif score>=2:  sig,col,emj = "BUY",         "#4CAF50","🟢"
    elif score<=-5: sig,col,emj = "STRONG SELL", "#D50000","⛔"
    elif score<=-2: sig,col,emj = "SELL",        "#F44336","🔴"
    else:           sig,col,emj = "HOLD",        "#FF9800","🟡"

    chg = round((cv-pv)/pv*100,2)
    # Bias-aware levels
    if score>=0:
        entry=round(cv,2); sl=round(cv-_atr*1.5,2); tgt=round(cv+_atr*3,2)
    else:
        entry=round(cv,2); sl=round(cv+_atr*1.5,2); tgt=round(cv-_atr*3,2)

    return {
        "signal":sig,"color":col,"emoji":emj,"score":score,
        "cmp":round(cv,2),"change_pct":chg,
        "entry":entry,"target":tgt,"sl":sl,"atr":round(_atr,2),
        "rsi":round(r,1),"reasons":reasons[:4],
        "tech_details":tech_details,
        "e20":round(e2,2),"e50":round(e5,2),
        "bb_upper":round(bu,2),"bb_lower":round(bl,2),
    }


# =============================================================================
#  FUNDAMENTAL FETCHER
# =============================================================================
def get_fundamentals(ticker_obj) -> dict:
    """Extract fundamental data from yfinance .info dict."""
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    def fmt_cr(v):
        if v and v != "N/A":
            try:
                n = float(v)
                if n >= 1e12: return f"₹{n/1e12:.2f}T"
                if n >= 1e9:  return f"₹{n/1e9:.2f}B"
                if n >= 1e7:  return f"₹{n/1e7:.2f}Cr"
                return f"₹{n:,.0f}"
            except: return str(v)
        return "N/A"

    def pct(v):
        try: return f"{float(v)*100:.2f}%" if v and v!="N/A" else "N/A"
        except: return "N/A"

    def val(k, decimals=2):
        v = info.get(k)
        if v is None or v=="N/A": return "N/A"
        try: return round(float(v), decimals)
        except: return str(v)

    # Fundamental score
    pe   = val("trailingPE",1)
    roe  = info.get("returnOnEquity")
    de   = val("debtToEquity",2)
    eps  = val("trailingEps",2)

    f_score = 0
    if pe != "N/A":
        if   float(pe)<15: f_score+=2
        elif float(pe)<25: f_score+=1
        elif float(pe)>50: f_score-=2
        elif float(pe)>35: f_score-=1
    if roe:
        if   float(roe)>0.20: f_score+=2
        elif float(roe)>0.12: f_score+=1
        elif float(roe)<0:    f_score-=1
    if de != "N/A":
        if   float(de)<50:  f_score+=1
        elif float(de)>200: f_score-=1

    if   f_score>=3: fund_view="💪 Fundamentally Strong"
    elif f_score>=1: fund_view="✅ Fundamentally Decent"
    elif f_score<=-2:fund_view="⚠️ Fundamentally Weak"
    else:            fund_view="🔄 Mixed Fundamentals"

    return {
        "company":      info.get("longName","N/A"),
        "sector":       info.get("sector","N/A"),
        "industry":     info.get("industry","N/A"),
        "market_cap":   fmt_cr(info.get("marketCap")),
        "pe":           pe,
        "pb":           val("priceToBook",2),
        "eps":          eps,
        "roe":          pct(roe),
        "de":           de,
        "div_yield":    pct(info.get("dividendYield")),
        "revenue":      fmt_cr(info.get("totalRevenue")),
        "net_income":   fmt_cr(info.get("netIncomeToCommon")),
        "w52_high":     val("fiftyTwoWeekHigh",2),
        "w52_low":      val("fiftyTwoWeekLow",2),
        "avg_vol":      fmt_cr(info.get("averageVolume")),
        "book_value":   val("bookValue",2),
        "fund_view":    fund_view,
        "f_score":      f_score,
        "description":  info.get("longBusinessSummary","N/A")[:300]+"…"
                        if info.get("longBusinessSummary") else "N/A",
    }


# =============================================================================
#  FULL STOCK ANALYSIS (for search)
# =============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def full_stock_analysis(ticker_str: str) -> dict | None:
    """
    Fetches OHLCV + fundamentals + intraday CMP for a single stock.
    Used for the search result panel.
    """
    if not LIVE:
        return None
    try:
        t    = yf.Ticker(ticker_str)
        hist = t.history(period="6mo", auto_adjust=True)
        if hist.empty or len(hist) < 35:
            return None

        # Near-real-time CMP via intraday
        try:
            intra = t.history(period="1d", interval="5m", auto_adjust=True)
            cmp_live = float(intra["Close"].iloc[-1]) if not intra.empty else float(hist["Close"].iloc[-1])
        except Exception:
            cmp_live = float(hist["Close"].iloc[-1])

        sig  = generate_signal(hist)
        if sig is None:
            return None
        sig["cmp"] = round(cmp_live, 2)

        fund = get_fundamentals(t)

        # Price history for chart (last 90 days)
        chart_data = hist["Close"].tail(90).reset_index()
        chart_data.columns = ["Date","Price"]
        chart_data["Date"] = pd.to_datetime(chart_data["Date"]).dt.date

        # Combined verdict
        total = sig["score"] + sig.get("score",0)//2 + fund["f_score"]
        if total>=5:   combined="🚀 STRONG BULLISH"
        elif total>=2: combined="🟢 BULLISH"
        elif total<=-5:combined="⛔ STRONG BEARISH"
        elif total<=-2:combined="🔴 BEARISH"
        else:          combined="🟡 NEUTRAL"

        return {**sig, "fund":fund, "chart":chart_data,
                "combined":combined, "ticker":ticker_str}
    except Exception:
        return None


# =============================================================================
#  SCREENER BATCH FETCH
# =============================================================================
def fetch_one(ticker_sym: str, display_name: str) -> dict:
    if not LIVE:
        return _mock(display_name, ticker_sym)
    try:
        t    = yf.Ticker(ticker_sym)
        hist = t.history(period="60d", auto_adjust=True)
        # Try to get near-real-time CMP
        try:
            intra = t.history(period="1d", interval="5m", auto_adjust=True)
            cmp_live = float(intra["Close"].iloc[-1]) if not intra.empty else None
        except Exception:
            cmp_live = None
        if hist.empty or len(hist)<35:
            return _mock(display_name, ticker_sym)
        sig = generate_signal(hist)
        if sig is None:
            return _mock(display_name, ticker_sym)
        if cmp_live:
            sig["cmp"] = round(cmp_live, 2)
        sig["symbol"] = display_name
        sig["ticker"] = ticker_sym
        return sig
    except Exception:
        return _mock(display_name, ticker_sym)


def _mock(name: str, ticker: str) -> dict:
    rng = np.random.default_rng(abs(hash(ticker)) % 9999)
    sc  = int(rng.integers(-5,6))
    c   = round(float(rng.uniform(200,4000)),2)
    a   = round(c*float(rng.uniform(.015,.035)),2)
    chg = round(float(rng.uniform(-3,3)),2)
    if sc>=5:  s,col,e="STRONG BUY","#00C853","🚀"
    elif sc>=2:s,col,e="BUY","#4CAF50","🟢"
    elif sc<=-5:s,col,e="STRONG SELL","#D50000","⛔"
    elif sc<=-2:s,col,e="SELL","#F44336","🔴"
    else:      s,col,e="HOLD","#FF9800","🟡"
    entry=c; tgt=round(c+a*3,2) if sc>=0 else round(c-a*3,2)
    sl=round(c-a*1.5,2) if sc>=0 else round(c+a*1.5,2)
    return {"symbol":name,"ticker":ticker,"signal":s,"color":col,"emoji":e,
            "score":sc,"cmp":c,"change_pct":chg,"entry":entry,"target":tgt,"sl":sl,
            "atr":a,"rsi":round(float(rng.uniform(25,75)),1),
            "reasons":["Demo data — add requirements.txt to GitHub repo"],
            "tech_details":{}}


@st.cache_data(ttl=180, show_spinner=False)
def load_screener(_key: int) -> tuple:
    all_tasks = (
        [(sym, meta["name"], "idx") for sym,meta in INDICES.items()] +
        [(f"{s}.NS", s, "stk") for s in FO_STOCKS]
    )
    idx_res=[]; stk_res=[]
    with ThreadPoolExecutor(max_workers=10) as ex:
        fm = {ex.submit(fetch_one, t, n):(t,n,k) for t,n,k in all_tasks}
        for f in as_completed(fm):
            t,n,k = fm[f]; r=f.result()
            (idx_res if k=="idx" else stk_res).append(r)
    ord_ = {"STRONG BUY":0,"BUY":1,"HOLD":2,"SELL":3,"STRONG SELL":4}
    return (sorted(idx_res,key=lambda x:ord_.get(x["signal"],2)),
            sorted(stk_res,key=lambda x:ord_.get(x["signal"],2)))


# =============================================================================
#  SESSION STATE
# =============================================================================
for k,v in [("cache_key",0),("search_ticker",""),("last_ref",time_ist())]:
    if k not in st.session_state:
        st.session_state[k]=v


# =============================================================================
#  CARD HTML HELPERS
# =============================================================================
SIG_BG    = {"STRONG BUY":"#002B14","BUY":"#0A1F10","HOLD":"#1A1400","SELL":"#1F0A0A","STRONG SELL":"#1A0000"}
SIG_BORDER= {"STRONG BUY":"#00C853","BUY":"#4CAF50","HOLD":"#FF9800","SELL":"#F44336","STRONG SELL":"#D50000"}
BADGE_BG  = SIG_BORDER
BADGE_FG  = {"STRONG BUY":"#000","BUY":"#000","HOLD":"#000","SELL":"#fff","STRONG SELL":"#fff"}

def mini_card(d: dict) -> str:
    sig=d["signal"]; bd=SIG_BORDER.get(sig,"#FF9800"); bg=SIG_BG.get(sig,"#1A1400")
    bb=BADGE_BG.get(sig,"#FF9800"); bf=BADGE_FG.get(sig,"#000")
    cc="#4CAF50" if d["change_pct"]>=0 else "#F44336"
    cs="▲" if d["change_pct"]>=0 else "▼"
    tc="#4CAF50" if d["score"]>=0 else "#F44336"
    sc_="#F44336" if d["score"]>=0 else "#4CAF50"
    dots="".join(
        f"<span style='color:{'#4CAF50' if i>0 and i<=d['score'] else '#F44336' if i<0 and i>=d['score'] else '#1E293B'};font-size:.85rem;'>{'●' if (i>0 and i<=d['score']) or (i<0 and i>=d['score']) else '○'}</span>"
        for i in range(-5,6) if i!=0
    )
    reas="".join(f"<div style='font-size:.65rem;color:#94A3B8;margin:2px 0;'>• {r}</div>" for r in d["reasons"][:2])
    return f"""
<div style='background:{bg};border:1px solid {bd};border-radius:14px;padding:1rem;
            margin:5px 0;position:relative;overflow:hidden;'>
  <div style='position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,{bd},{bd}44);'></div>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>
    <div>
      <div style='font-size:.9rem;font-weight:800;color:#F8FAFC;'>{d['symbol']}</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;font-weight:700;color:#F8FAFC;'> ₹{d['cmp']:,.2f}</div>
      <div style='font-size:.75rem;color:{cc};font-weight:600;'>{cs} {abs(d['change_pct'])}%</div>
    </div>
    <div>
      <div style='background:{bb};color:{bf};font-size:.68rem;font-weight:800;
                  padding:4px 10px;border-radius:20px;text-align:center;'>{d['emoji']} {sig}</div>
      <div style='font-size:.6rem;color:#64748B;text-align:center;margin-top:3px;'>RSI {d['rsi']}</div>
    </div>
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin:8px 0;'>
    <div style='background:#060B14;border-radius:8px;padding:5px 6px;text-align:center;'>
      <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em;'>Entry</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:#E2E8F0;font-weight:600;'>₹{d['entry']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:8px;padding:5px 6px;text-align:center;'>
      <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em;'>Target</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{tc};font-weight:700;'>₹{d['target']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:8px;padding:5px 6px;text-align:center;'>
      <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em;'>SL</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{sc_};font-weight:700;'>₹{d['sl']:,.0f}</div>
    </div>
  </div>
  <div style='text-align:center;letter-spacing:1px;margin:5px 0;'>{dots}</div>
  <div style='border-top:1px solid #1E293B;padding-top:6px;margin-top:4px;'>{reas}</div>
</div>"""


def index_card(d: dict, meta: dict) -> str:
    sig=d["signal"]; bd=SIG_BORDER.get(sig,"#FF9800")
    bb=BADGE_BG.get(sig,"#FF9800"); bf=BADGE_FG.get(sig,"#000")
    cc="#4CAF50" if d["change_pct"]>=0 else "#F44336"
    cs="▲" if d["change_pct"]>=0 else "▼"
    tc="#4CAF50" if d["score"]>=0 else "#F44336"
    sc_="#F44336" if d["score"]>=0 else "#4CAF50"
    return f"""
<div style='background:linear-gradient(135deg,#0D1117,#111827);
            border:1.5px solid {bd};border-radius:18px;padding:1.4rem;
            margin:4px 0;position:relative;overflow:hidden;'>
  <div style='position:absolute;top:0;left:0;right:0;height:4px;
              background:linear-gradient(90deg,{bd},{bd}33);'></div>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div style='font-size:.62rem;font-weight:700;color:#6366F1;
                  letter-spacing:.14em;text-transform:uppercase;margin-bottom:3px;'>
        {meta['emoji']} {d['symbol']}</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:1.9rem;
                  font-weight:800;color:#F8FAFC;line-height:1;'>{d['cmp']:,.2f}</div>
      <div style='font-size:.85rem;color:{cc};font-weight:700;margin-top:3px;'>
        {cs} {abs(d['change_pct'])}% today</div>
    </div>
    <div style='text-align:right;'>
      <div style='background:{bb};color:{bf};font-size:.8rem;font-weight:800;
                  padding:7px 16px;border-radius:22px;margin-bottom:6px;'>
        {d['emoji']} {sig}</div>
      <div style='font-size:.65rem;color:#64748B;'>RSI {d['rsi']} · ATR {d['atr']:,.0f}</div>
    </div>
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;margin-top:12px;'>
    <div style='background:#060B14;border-radius:10px;padding:7px 10px;text-align:center;'>
      <div style='font-size:.58rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>Entry</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:#E2E8F0;font-weight:700;'>₹{d['entry']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:10px;padding:7px 10px;text-align:center;'>
      <div style='font-size:.58rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>Target</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:{tc};font-weight:700;'>₹{d['target']:,.0f}</div>
    </div>
    <div style='background:#060B14;border-radius:10px;padding:7px 10px;text-align:center;'>
      <div style='font-size:.58rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>Stop Loss</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:{sc_};font-weight:700;'>₹{d['sl']:,.0f}</div>
    </div>
  </div>
</div>"""


# =============================================================================
#  HEADER
# =============================================================================
h1, h2 = st.columns([5,2])
with h1:
    st.markdown("""
    <div style='padding:.5rem 0;'>
      <div style='font-size:.65rem;font-weight:700;letter-spacing:.2em;
                  text-transform:uppercase;color:#6366F1;margin-bottom:6px;'>
        NSE · FUTURES & OPTIONS · LIVE SIGNALS
      </div>
      <h1 style='font-size:2.2rem;font-weight:900;margin:0;letter-spacing:-.03em;
                 background:linear-gradient(135deg,#F8FAFC 0%,#94A3B8 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        Signal Screener
      </h1>
    </div>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:1rem;'>
      <div style='font-size:.6rem;font-weight:600;letter-spacing:.12em;
                  text-transform:uppercase;color:#64748B;'>🕐 Indian Standard Time</div>
      <div id='clock' style='font-family:JetBrains Mono,monospace;font-size:1rem;
                              color:#6366F1;font-weight:700;margin-top:3px;'>
        {now_ist()}
      </div>
      <div style='font-size:.62rem;color:#475569;margin-top:2px;'>
        Last Refresh: {st.session_state['last_ref']}
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()


# =============================================================================
#  SEARCH BAR
# =============================================================================
st.markdown("""
<div style='font-size:.65rem;font-weight:700;letter-spacing:.14em;
            text-transform:uppercase;color:#6366F1;margin-bottom:6px;'>
  🔍 Deep Analysis — Search Any NSE Stock or Index
</div>""", unsafe_allow_html=True)

sc1, sc2 = st.columns([5,1])
with sc1:
    search_input = st.text_input(
        "Search",
        placeholder="e.g.  RELIANCE  ·  TCS  ·  NIFTY  ·  BANKNIFTY  ·  HDFCBANK",
        label_visibility="collapsed",
        key="search_box",
    )
with sc2:
    if st.button("🔍 Analyse", use_container_width=True):
        st.session_state["search_ticker"] = search_input.strip().upper()

# Resolve ticker
raw_query = st.session_state.get("search_ticker","") or search_input.strip().upper()

if raw_query:
    if raw_query in SEARCH_ALIASES:
        resolved = SEARCH_ALIASES[raw_query]
    elif raw_query.startswith("^"):
        resolved = raw_query
    else:
        resolved = raw_query + ".NS"

    # ── SEARCH RESULT PANEL ───────────────────────────────────────────────────
    with st.spinner(f"⏳ Analysing {raw_query}…"):
        analysis = full_stock_analysis(resolved)

    if analysis:
        f   = analysis["fund"]
        sig = analysis["signal"]
        bd  = SIG_BORDER.get(sig,"#FF9800")
        bb  = BADGE_BG.get(sig,"#FF9800")
        bfc = BADGE_FG.get(sig,"#000")
        tc  = "#4CAF50" if analysis["score"]>=0 else "#F44336"
        sc_ = "#F44336" if analysis["score"]>=0 else "#4CAF50"
        cc  = "#4CAF50" if analysis["change_pct"]>=0 else "#F44336"

        # Company name banner
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0D1117,#111827);
                    border:1.5px solid {bd};border-radius:18px;
                    padding:1.2rem 1.6rem;margin:10px 0 4px;
                    position:relative;overflow:hidden;'>
          <div style='position:absolute;top:0;left:0;right:0;height:4px;
                      background:linear-gradient(90deg,{bd},{bd}33);'></div>
          <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;'>
            <div>
              <div style='font-size:.62rem;color:#6366F1;font-weight:700;
                          letter-spacing:.12em;text-transform:uppercase;'>{f['sector']} · {f['industry']}</div>
              <div style='font-size:1.4rem;font-weight:900;color:#F8FAFC;margin:3px 0;'>{f['company']}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:2rem;
                          font-weight:800;color:#F8FAFC;'>₹{analysis['cmp']:,.2f}
                <span style='font-size:.9rem;color:{cc};margin-left:10px;'>
                  {"▲" if analysis["change_pct"]>=0 else "▼"} {abs(analysis["change_pct"])}%
                </span>
              </div>
            </div>
            <div style='text-align:center;'>
              <div style='background:{bb};color:{bfc};font-size:1.1rem;font-weight:800;
                          padding:10px 24px;border-radius:28px;letter-spacing:.05em;'>
                {analysis['emoji']} {sig}
              </div>
              <div style='font-size:.7rem;color:#64748B;margin-top:6px;'>{analysis["combined"]}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Three main columns
        tl, tm, tr_ = st.columns([2, 2, 1.8])

        with tl:
            # Price chart
            st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:12px 0 6px;'>📈 90-Day Price Chart</div>", unsafe_allow_html=True)
            chart_df = analysis["chart"].set_index("Date")
            st.line_chart(chart_df["Price"], height=180, use_container_width=True)

            # Entry / Target / SL
            st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:12px 0 6px;'>🎯 Trade Levels</div>", unsafe_allow_html=True)
            l1,l2,l3 = st.columns(3)
            l1.metric("Entry",   f"₹{analysis['entry']:,.2f}")
            l2.metric("Target",  f"₹{analysis['target']:,.2f}")
            l3.metric("Stop Loss",f"₹{analysis['sl']:,.2f}")

        with tm:
            st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:12px 0 6px;'>🔬 Technical Analysis</div>", unsafe_allow_html=True)
            td = analysis.get("tech_details",{})

            def trow(label, value, bull_cond=None):
                if bull_cond is True:   col="#4CAF50"
                elif bull_cond is False: col="#F44336"
                else:                   col="#94A3B8"
                return f"""<div style='display:flex;justify-content:space-between;
                    padding:6px 10px;border-bottom:1px solid #1E293B;'>
                  <span style='font-size:.75rem;color:#64748B;'>{label}</span>
                  <span style='font-family:JetBrains Mono,monospace;font-size:.75rem;
                               color:{col};font-weight:600;'>{value}</span></div>"""

            r_val = analysis["rsi"]
            rows  = [
                trow("RSI (14)",        f"{r_val}",     r_val<50),
                trow("MACD",            td.get("MACD","—"), td.get("MACD")=="Bullish"),
                trow("EMA Trend",       td.get("EMA Trend","—"), td.get("EMA Trend")=="Uptrend"),
                trow("BB Position",     td.get("BB Position","—")),
                trow("Stoch %K",        str(td.get("Stoch %K","—"))),
                trow("Volume",          td.get("Volume","—")),
                trow("EMA 20",          f"₹{analysis['e20']:,.2f}"),
                trow("EMA 50",          f"₹{analysis['e50']:,.2f}"),
                trow("BB Upper",        f"₹{analysis['bb_upper']:,.2f}"),
                trow("BB Lower",        f"₹{analysis['bb_lower']:,.2f}"),
                trow("ATR",             f"₹{analysis['atr']:,.2f}"),
                trow("Signal Score",    f"{analysis['score']} / 8"),
            ]
            st.markdown(f"""
            <div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>
              {"".join(rows)}
            </div>""", unsafe_allow_html=True)

            # Signal reasons
            st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:12px 0 6px;'>💡 Why This Signal?</div>", unsafe_allow_html=True)
            for reason in analysis["reasons"]:
                st.markdown(f"<div style='font-size:.75rem;color:#94A3B8;padding:3px 0;'>• {reason}</div>", unsafe_allow_html=True)

        with tr_:
            st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:12px 0 6px;'>📊 Fundamentals</div>", unsafe_allow_html=True)

            def frow(label, value):
                return f"""<div style='display:flex;justify-content:space-between;
                    padding:6px 10px;border-bottom:1px solid #1E293B;'>
                  <span style='font-size:.72rem;color:#64748B;'>{label}</span>
                  <span style='font-family:JetBrains Mono,monospace;font-size:.72rem;
                               color:#E2E8F0;font-weight:600;'>{value}</span></div>"""

            fund_rows = [
                frow("Market Cap",   f["market_cap"]),
                frow("P/E Ratio",    str(f["pe"])),
                frow("P/B Ratio",    str(f["pb"])),
                frow("EPS",          str(f["eps"])),
                frow("ROE",          f["roe"]),
                frow("Debt/Equity",  str(f["de"])),
                frow("Revenue",      f["revenue"]),
                frow("Net Income",   f["net_income"]),
                frow("Div Yield",    f["div_yield"]),
                frow("52W High",     f"₹{f['w52_high']}"),
                frow("52W Low",      f"₹{f['w52_low']}"),
                frow("Book Value",   str(f["book_value"])),
            ]
            st.markdown(f"""
            <div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>
              {"".join(fund_rows)}
            </div>""", unsafe_allow_html=True)

            # Fundamental view badge
            st.markdown(f"""
            <div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;
                        padding:10px 12px;margin-top:10px;text-align:center;'>
              <div style='font-size:.62rem;color:#64748B;text-transform:uppercase;
                          letter-spacing:.1em;font-weight:600;margin-bottom:4px;'>
                Fundamental View</div>
              <div style='font-size:.85rem;font-weight:700;color:#6366F1;'>{f["fund_view"]}</div>
            </div>""", unsafe_allow_html=True)

            # Company description
            if f["description"] != "N/A":
                with st.expander("ℹ️ About Company"):
                    st.markdown(f"<div style='font-size:.72rem;color:#94A3B8;line-height:1.6;'>{f['description']}</div>", unsafe_allow_html=True)

    elif LIVE:
        st.warning(f"⚠️ Could not find data for **'{raw_query}'**. Try: RELIANCE, TCS, HDFCBANK, BANKNIFTY, NIFTY")
    else:
        st.info("📦 Add `requirements.txt` to GitHub and reboot to enable live search.")

    st.divider()


# =============================================================================
#  CONTROLS ROW
# =============================================================================
ct1, ct2, ct3, ct4 = st.columns([2,2,2,2])
with ct1:
    if st.button("🔄  REFRESH ALL SIGNALS", use_container_width=True):
        st.cache_data.clear()
        st.session_state["cache_key"] += 1
        st.session_state["last_ref"] = time_ist()
        st.rerun()
with ct2:
    filter_sig = st.selectbox("Signal Filter",
        ["All Signals","STRONG BUY","BUY","HOLD","SELL","STRONG SELL"],
        label_visibility="collapsed")
with ct3:
    filter_sec = st.selectbox("Section",
        ["All","Indices Only","F&O Stocks Only"],
        label_visibility="collapsed")
with ct4:
    st.markdown(f"<div style='text-align:right;padding:.4rem 0;'>"
                f"<div style='font-size:.6rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>Data Delay</div>"
                f"<div style='font-size:.85rem;color:#6366F1;font-weight:700;'>~5 min (intraday)</div></div>",
                unsafe_allow_html=True)

st.divider()

# ── Load screener data ────────────────────────────────────────────────────────
if not LIVE:
    st.warning("⚠️ yfinance not found — showing demo data. Commit `requirements.txt` to your GitHub repo.", icon="📦")

with st.spinner("⏳ Fetching signals…"):
    idx_sigs, stk_sigs = load_screener(_key=st.session_state["cache_key"])

def filt(lst):
    if filter_sig != "All Signals":
        lst = [x for x in lst if x["signal"]==filter_sig]
    return lst

fi = filt(idx_sigs) if filter_sec!="F&O Stocks Only" else []
fs = filt(stk_sigs) if filter_sec!="Indices Only"    else []

# =============================================================================
#  SUMMARY STRIP
# =============================================================================
all_s = idx_sigs + stk_sigs
cnts  = {s:sum(1 for x in all_s if x["signal"]==s)
         for s in ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]}
m1,m2,m3,m4,m5 = st.columns(5)
with m1: st.metric("🚀 Strong Buy",  cnts["STRONG BUY"])
with m2: st.metric("🟢 Buy",         cnts["BUY"])
with m3: st.metric("🟡 Hold",        cnts["HOLD"])
with m4: st.metric("🔴 Sell",        cnts["SELL"])
with m5: st.metric("⛔ Strong Sell", cnts["STRONG SELL"])

st.divider()

# =============================================================================
#  INDEX CARDS
# =============================================================================
if fi:
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:10px;'>📊 Market Indices</div>", unsafe_allow_html=True)
    idx_meta = {v["name"]:v for v in INDICES.values()}
    for i in range(0,len(fi),3):
        row=fi[i:i+3]; cols=st.columns(len(row))
        for col,d in zip(cols,row):
            meta=idx_meta.get(d["symbol"],{"emoji":"📈","name":d["symbol"],"short":""})
            col.markdown(index_card(d,meta), unsafe_allow_html=True)
    st.divider()

# =============================================================================
#  F&O STOCK CARDS
# =============================================================================
if fs:
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:10px;'>🎯 F&O Stock Signals</div>", unsafe_allow_html=True)
    for i in range(0,len(fs),4):
        row=fs[i:i+4]; cols=st.columns(4)
        for j,col in enumerate(cols):
            if j<len(row):
                col.markdown(mini_card(row[j]), unsafe_allow_html=True)
elif not fi:
    st.info("No signals match the selected filter.")

st.divider()

# =============================================================================
#  FULL TABLE
# =============================================================================
with st.expander("📋  All Signals — Full Table", expanded=False):
    rows=[]
    for d in (idx_sigs+stk_sigs):
        rows.append({"Symbol":d["symbol"],"Signal":f"{d['emoji']} {d['signal']}",
                     "CMP (₹)":d["cmp"],"Change %":d["change_pct"],"RSI":d["rsi"],
                     "Entry (₹)":d["entry"],"Target (₹)":d["target"],"SL (₹)":d["sl"],
                     "Score":d["score"]})
    tbl=pd.DataFrame(rows)
    for c in ["CMP (₹)","Change %","Entry (₹)","Target (₹)","SL (₹)","RSI"]:
        tbl[c]=tbl[c].astype(float)
    tbl["Score"]=tbl["Score"].astype(int)
    st.dataframe(tbl,column_config={
        "CMP (₹)":    st.column_config.NumberColumn(format="₹%.2f"),
        "Change %":   st.column_config.NumberColumn(format="%.2f%%"),
        "Entry (₹)":  st.column_config.NumberColumn(format="₹%.2f"),
        "Target (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        "SL (₹)":     st.column_config.NumberColumn(format="₹%.2f"),
        "RSI":        st.column_config.NumberColumn(format="%.1f"),
        "Score":      st.column_config.NumberColumn(format="%d"),
    },use_container_width=True,hide_index=True,height=460)
    csv=tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV",csv,"fo_signals.csv","text/csv")

st.markdown("""
<div style='text-align:center;padding:1.5rem 0 .5rem;'>
  <p style='font-size:.6rem;color:#1E293B;font-family:JetBrains Mono,monospace;letter-spacing:.08em;line-height:1.9;'>
    ⚠️ FOR EDUCATIONAL PURPOSES ONLY · NOT SEBI-REGISTERED ADVICE<br>
    DATA VIA YAHOO FINANCE · ALWAYS DO YOUR OWN RESEARCH BEFORE TRADING
  </p>
</div>""", unsafe_allow_html=True)
