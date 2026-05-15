# =============================================================================
#  NSE F&O SIGNAL SCREENER  v7.0
#  ✅ TradingView-style interactive charts (Plotly)
#     Candlestick · EMA 20/50 · Bollinger Bands · Volume · RSI · MACD
#  ✅ 127 stocks via batch yf.download
#  ✅ CALL / PUT option recommendations
#  ✅ Tabs: Chart · All · BUY · SELL · Options · Sector
#  ✅ Real-time IST clock · 5-min intraday price
#  Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    LIVE = True
except ModuleNotFoundError:
    LIVE = False

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():  return datetime.now(IST).strftime("%d %b %Y  %I:%M:%S %p IST")
def time_ist(): return datetime.now(IST).strftime("%I:%M %p IST")

st.set_page_config(page_title="NSE F&O Screener", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background-color:#0E1117 !important;color:#E2E8F0 !important;}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#0D1117}::-webkit-scrollbar-thumb{background:#1E293B;border-radius:4px}
section[data-testid="stSidebar"]{background:#0D1117 !important;border-right:1px solid #1E293B !important}
hr{border-color:#1E293B !important;margin:.3rem 0 !important}
[data-testid="stMetric"]{background:linear-gradient(135deg,#0D1117,#111827);border:1px solid #1E293B;border-radius:12px;padding:.9rem 1.1rem !important}
[data-testid="stMetricLabel"]{font-size:.6rem !important;font-weight:600 !important;letter-spacing:.12em !important;text-transform:uppercase !important;color:#64748B !important}
[data-testid="stMetricValue"]{font-size:1.5rem !important;font-weight:800 !important;font-family:'JetBrains Mono',monospace !important;color:#F8FAFC !important}
div[data-testid="stTextInput"] input{background:#0D1117 !important;border:1.5px solid #6366F1 !important;border-radius:12px !important;color:#F8FAFC !important;font-size:.95rem !important;font-family:'JetBrains Mono',monospace !important;padding:.7rem 1.1rem !important}
div[data-testid="stButton"]>button{background:linear-gradient(135deg,#6366F1,#8B5CF6) !important;border:none !important;color:#fff !important;font-weight:700 !important;font-size:.85rem !important;padding:.6rem 1.2rem !important;border-radius:10px !important;box-shadow:0 4px 14px rgba(99,102,241,.35) !important;transition:all .2s !important}
div[data-testid="stButton"]>button:hover{transform:translateY(-1px) !important;box-shadow:0 6px 20px rgba(99,102,241,.55) !important}
.stTabs [data-baseweb="tab-list"]{background:#0D1117 !important;border-bottom:1px solid #1E293B !important;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent !important;border:none !important;color:#64748B !important;font-weight:600 !important;font-size:.78rem !important;letter-spacing:.04em !important;padding:.55rem 1rem !important;border-radius:8px 8px 0 0 !important}
.stTabs [aria-selected="true"]{background:#1E293B !important;color:#6366F1 !important;border-bottom:2px solid #6366F1 !important}
.streamlit-expanderHeader{background:#0D1117 !important;border:1px solid #1E293B !important;border-radius:10px !important;font-size:.75rem !important}
.stDownloadButton>button{background:transparent !important;border:1px solid #6366F1 !important;color:#6366F1 !important;font-size:.7rem !important;border-radius:8px !important}
div[data-testid="stSelectbox"] label{font-size:.6rem !important;text-transform:uppercase !important;letter-spacing:.1em !important;color:#64748B !important}
</style>""", unsafe_allow_html=True)

# =============================================================================
#  UNIVERSE
# =============================================================================
INDICES = {
    "^NSEI":      {"name":"NIFTY 50",      "emoji":"🔵"},
    "^NSEBANK":   {"name":"BANK NIFTY",    "emoji":"🏦"},
    "^CNXIT":     {"name":"NIFTY IT",      "emoji":"💻"},
    "^CNXPHARMA": {"name":"NIFTY PHARMA",  "emoji":"💊"},
    "^CNXAUTO":   {"name":"NIFTY AUTO",    "emoji":"🚗"},
    "^CNXMIDCAP": {"name":"MIDCAP 100",    "emoji":"📊"},
}
SECTOR_MAP = {
    "Banking":    ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
                   "FEDERALBNK","BANDHANBNK","PNB","CANBK","BANKBARODA","IDFCFIRSTB","AUBANK"],
    "Finance":    ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SBICARD",
                   "HDFCLIFE","SBILIFE","ICICIGI","ICICIPRULI"],
    "IT & Tech":  ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS",
                   "PERSISTENT","COFORGE","OFSS","KPITTECH","TATAELXSI","NAUKRI"],
    "Consumer":   ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO",
                   "COLPAL","GODREJCP","EMAMILTD","TATACONSUM","TITAN","DMART","TRENT"],
    "Auto":       ["MARUTI","TATAMOTORS","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
                   "TVSMOTOR","MOTHERSON","BALKRISIND","MRF","BOSCHLTD"],
    "Pharma":     ["SUNPHARMA","DIVISLAB","CIPLA","DRREDDY","APOLLOHOSP","AUROPHARMA",
                   "LUPIN","BIOCON","ALKEM","TORNTPHARM","IPCALAB"],
    "Energy":     ["RELIANCE","ONGC","BPCL","ADANIGREEN","TATAPOWER","IGL",
                   "MGL","PETRONET","TORNTPOWER","CESC","ADANITRANS"],
    "Metals":     ["TATASTEEL","JSWSTEEL","HINDALCO","NATIONALUM","NMDC","SAIL","HINDCOPPER"],
    "Infra & RE": ["LT","ADANIENT","ADANIPORTS","DLF","GODREJPROP","PRESTIGE","OBEROIRLTY"],
    "Chemicals":  ["PIDILITIND","SRF","DEEPAKNTR","AARTIIND","NAVINFLUOR"],
    "Others":     ["ULTRACEMCO","ASIANPAINT","GRASIM","SHREECEM","POWERGRID","NTPC",
                   "COALINDIA","MM","BHARTIARTL","ZOMATO","PAYTM","NYKAA"],
}
FO_STOCKS = [s for stocks in SECTOR_MAP.values() for s in stocks]

SEARCH_ALIASES = {
    "NIFTY":"^NSEI","NIFTY50":"^NSEI","NIFTY 50":"^NSEI",
    "BANKNIFTY":"^NSEBANK","BANK NIFTY":"^NSEBANK",
    "NIFTYIT":"^CNXIT","NIFTYPHARMA":"^CNXPHARMA","NIFTYAUTO":"^CNXAUTO",
    "MIDCAP":"^CNXMIDCAP",
}
def get_sector(sym):
    for sec,stocks in SECTOR_MAP.items():
        if sym in stocks: return sec
    return "Others"

# =============================================================================
#  TECHNICAL INDICATORS
# =============================================================================
def calc_rsi(c, p=14):
    d=c.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return (100-100/(1+g/l.replace(0,np.nan))).fillna(50)

def calc_macd(c):
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    m=e12-e26; s=m.ewm(span=9,adjust=False).mean(); return m,s,m-s

def calc_bb(c, p=20):
    sma=c.rolling(p).mean(); std=c.rolling(p).std()
    return sma+2*std, sma, sma-2*std

def calc_atr(h, l, c, p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return float(tr.rolling(p).mean().dropna().iloc[-1])

def calc_stoch(h, l, c, k=14, d=3):
    lo=l.rolling(k).min(); hi=h.rolling(k).max()
    pct_k=100*(c-lo)/(hi-lo+1e-9)
    return pct_k, pct_k.rolling(d).mean()

# =============================================================================
#  TRADINGVIEW-STYLE CHART
# =============================================================================
TV_BG        = "#131722"
TV_PANEL_BG  = "#131722"
TV_GRID      = "#1e222d"
TV_TEXT      = "#d1d4dc"
TV_UP        = "#26a69a"
TV_DOWN      = "#ef5350"
TV_EMA20     = "#2962ff"
TV_EMA50     = "#ff6d00"
TV_BB        = "#9c27b0"
TV_RSI       = "#e91e63"
TV_MACD_LINE = "#2196f3"
TV_SIG_LINE  = "#ff9800"
TV_ZERO      = "#434651"

def create_tv_chart(hist: pd.DataFrame, symbol: str,
                    sig_dict: dict | None = None,
                    period_label: str = "3 Months") -> go.Figure:
    """
    Full TradingView-style chart:
      Row 1 (55%): Candlestick + EMA20 + EMA50 + Bollinger Bands + Entry/Target/SL lines
      Row 2 (12%): Volume bars (green/red)
      Row 3 (17%): RSI with 30/70 bands + coloured background zones
      Row 4 (16%): MACD line + Signal line + Histogram
    """
    if hist.empty or len(hist) < 10:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {symbol}",
                          paper_bgcolor=TV_BG, plot_bgcolor=TV_BG)
        return fig

    # Ensure we have Open column (indices may not have it)
    if "Open" not in hist.columns:
        hist = hist.copy(); hist["Open"] = hist["Close"].shift(1).fillna(hist["Close"])

    c, h, l, o, v = hist["Close"], hist["High"], hist["Low"], hist["Open"], hist["Volume"]
    ema20 = c.ewm(span=20,adjust=False).mean()
    ema50 = c.ewm(span=50,adjust=False).mean()
    bb_u, bb_m, bb_l = calc_bb(c)
    rsi_s           = calc_rsi(c)
    macd_l,sig_l,hist_l = calc_macd(c)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.54, 0.13, 0.17, 0.16],
    )

    # ── ROW 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=hist.index, open=o, high=h, low=l, close=c,
        name="Price",
        increasing=dict(line=dict(color=TV_UP,   width=1), fillcolor=TV_UP),
        decreasing=dict(line=dict(color=TV_DOWN,  width=1), fillcolor=TV_DOWN),
        showlegend=True,
    ), row=1, col=1)

    # Bollinger Bands (fill between)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_u, name="BB Upper",
        line=dict(color=TV_BB, width=1, dash="dot"),
        opacity=0.7, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_l, name="BB Lower",
        line=dict(color=TV_BB, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(156,39,176,0.06)",
        opacity=0.7, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_m, name="BB Mid",
        line=dict(color=TV_BB, width=0.8, dash="dash"),
        opacity=0.4, showlegend=False,
    ), row=1, col=1)

    # EMA lines
    fig.add_trace(go.Scatter(
        x=hist.index, y=ema20, name="EMA 20",
        line=dict(color=TV_EMA20, width=1.8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=ema50, name="EMA 50",
        line=dict(color=TV_EMA50, width=1.8),
    ), row=1, col=1)

    # Entry / Target / SL horizontal lines
    if sig_dict:
        entry = sig_dict.get("entry"); tgt = sig_dict.get("target"); sl = sig_dict.get("sl")
        last_x = hist.index[-1]
        if entry:
            fig.add_hline(y=entry, line=dict(color="#F8FAFC", width=1, dash="dot"),
                          annotation_text=f" Entry ₹{entry:,.0f}",
                          annotation_font=dict(color="#F8FAFC", size=10), row=1, col=1)
        if tgt:
            fig.add_hline(y=tgt, line=dict(color=TV_UP, width=1.2, dash="dash"),
                          annotation_text=f" Target ₹{tgt:,.0f}",
                          annotation_font=dict(color=TV_UP, size=10), row=1, col=1)
        if sl:
            fig.add_hline(y=sl, line=dict(color=TV_DOWN, width=1.2, dash="dash"),
                          annotation_text=f" SL ₹{sl:,.0f}",
                          annotation_font=dict(color=TV_DOWN, size=10), row=1, col=1)

    # ── ROW 2: Volume ─────────────────────────────────────────────────────────
    vol_colors = [TV_UP if float(c.iloc[i]) >= float(o.iloc[i]) else TV_DOWN
                  for i in range(len(hist))]
    fig.add_trace(go.Bar(
        x=hist.index, y=v, name="Volume",
        marker_color=vol_colors, opacity=0.65, showlegend=True,
    ), row=2, col=1)

    # ── ROW 3: RSI ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=hist.index, y=rsi_s, name="RSI (14)",
        line=dict(color=TV_RSI, width=1.8),
    ), row=3, col=1)
    # Overbought / oversold zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,.08)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,.08)",
                  line_width=0, row=3, col=1)
    for lvl, col_, lbl in [(70, TV_DOWN, "OB 70"), (30, TV_UP, "OS 30"), (50, TV_ZERO, "")]:
        fig.add_hline(y=lvl, line=dict(color=col_, width=0.8, dash="dash"),
                      annotation_text=f" {lbl}",
                      annotation_font=dict(color=col_, size=9), row=3, col=1)

    # ── ROW 4: MACD ───────────────────────────────────────────────────────────
    hist_colors = [TV_UP if float(v) >= 0 else TV_DOWN for v in hist_l]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist_l, name="Histogram",
        marker_color=hist_colors, opacity=0.65, showlegend=True,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_l, name="MACD",
        line=dict(color=TV_MACD_LINE, width=1.8),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=sig_l, name="Signal",
        line=dict(color=TV_SIG_LINE, width=1.8),
    ), row=4, col=1)
    fig.add_hline(y=0, line=dict(color=TV_ZERO, width=0.6), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    signal_label = f"  {sig_dict['emoji']} {sig_dict['signal']}  |  RSI {sig_dict['rsi']}  |  Score {sig_dict['score']}/8" if sig_dict else ""
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b>  ·  {period_label}{signal_label}",
            font=dict(color=TV_TEXT, size=14, family="JetBrains Mono"),
            x=0.01, y=0.99,
        ),
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_PANEL_BG,
        font=dict(color=TV_TEXT, family="JetBrains Mono, monospace", size=10),
        height=700,
        margin=dict(l=55, r=30, t=50, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e222d", bordercolor="#434651",
                        font=dict(color=TV_TEXT, family="JetBrains Mono")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            bgcolor="rgba(19,23,34,.85)", bordercolor="#1e222d", borderwidth=1,
            font=dict(size=10),
        ),
    )
    # Grid + axes styling for all 4 rows
    axis_style = dict(gridcolor=TV_GRID, gridwidth=0.5,
                      tickfont=dict(color="#787b86", size=9),
                      linecolor="#2a2e39", showline=True,
                      zeroline=False, showgrid=True)
    for i in range(1, 5):
        fig.update_xaxes(**axis_style, row=i, col=1,
                         showticklabels=(i == 4))
        fig.update_yaxes(**axis_style, row=i, col=1)

    # RSI y-range fixed
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


# =============================================================================
#  SIGNAL ENGINE
# =============================================================================
def generate_signal(hist: pd.DataFrame) -> dict | None:
    if len(hist) < 35: return None
    c,h,l,v = hist["Close"],hist["High"],hist["Low"],hist["Volume"]
    _r=calc_rsi(c); _m,_s,_h=calc_macd(c)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    bbu,_,bbl=calc_bb(c); _atr=calc_atr(h,l,c)
    stk,std_=calc_stoch(h,l,c)
    cv=float(c.iloc[-1]); pv=float(c.iloc[-2])
    r=float(_r.iloc[-1]); m=float(_m.iloc[-1]); ms=float(_s.iloc[-1])
    pm=float(_m.iloc[-3]); ps=float(_s.iloc[-3])
    e2=float(e20.iloc[-1]); e5=float(e50.iloc[-1])
    bu=float(bbu.iloc[-1]); bl=float(bbl.iloc[-1])
    sk=float(stk.iloc[-1]); sd=float(std_.iloc[-1])
    va=float(v.iloc[-20:].mean()); vc=float(v.iloc[-1])
    score=0; reasons=[]
    if r<=30:   score+=2; reasons.append(f"RSI Oversold ({r:.0f}) 🔥")
    elif r<=42: score+=1; reasons.append(f"RSI Bullish Zone ({r:.0f})")
    elif r>=70: score-=2; reasons.append(f"RSI Overbought ({r:.0f}) ⚠️")
    elif r>=58: score-=1; reasons.append(f"RSI Bearish Zone ({r:.0f})")
    else: reasons.append(f"RSI Neutral ({r:.0f})")
    if m>ms and pm<ps:   score+=2; reasons.append("MACD Bullish Crossover ✅")
    elif m>ms:           score+=1; reasons.append("MACD Bullish")
    elif m<ms and pm>ps: score-=2; reasons.append("MACD Bearish Crossover ❌")
    else:                score-=1; reasons.append("MACD Bearish")
    if cv>e2>e5:   score+=2; reasons.append("Price>EMA20>EMA50 (Uptrend) 📈")
    elif cv>e2:    score+=1; reasons.append("Price Above EMA20")
    elif cv<e2<e5: score-=2; reasons.append("Price<EMA20<EMA50 (Downtrend) 📉")
    elif cv<e2:    score-=1; reasons.append("Price Below EMA20")
    if cv<=bl:     score+=1; reasons.append("At Lower Bollinger — Oversold Zone")
    elif cv>=bu:   score-=1; reasons.append("At Upper Bollinger — Overbought Zone")
    if sk<20 and sk>sd:   score+=1; reasons.append(f"Stoch Oversold Crossup ({sk:.0f})")
    elif sk>80 and sk<sd: score-=1; reasons.append(f"Stoch Overbought ({sk:.0f})")
    if vc>1.8*va and cv>pv:   score+=1; reasons.append("Volume Surge + Price Up 📊")
    elif vc>1.8*va and cv<pv: score-=1; reasons.append("Volume Surge + Price Down 📊")
    if score>=5:    sig,col,emj="STRONG BUY","#00C853","🚀"
    elif score>=2:  sig,col,emj="BUY","#4CAF50","🟢"
    elif score<=-5: sig,col,emj="STRONG SELL","#D50000","⛔"
    elif score<=-2: sig,col,emj="SELL","#F44336","🔴"
    else:           sig,col,emj="HOLD","#FF9800","🟡"
    chg=round((cv-pv)/pv*100,2)
    if score>=0: entry=round(cv,2);sl=round(cv-_atr*1.5,2);tgt=round(cv+_atr*3,2)
    else:        entry=round(cv,2);sl=round(cv+_atr*1.5,2);tgt=round(cv-_atr*3,2)
    return {"signal":sig,"color":col,"emoji":emj,"score":score,
            "cmp":round(cv,2),"change_pct":chg,"entry":entry,"target":tgt,"sl":sl,
            "atr":round(_atr,2),"rsi":round(r,1),"reasons":reasons[:4],
            "e20":round(e2,2),"e50":round(e5,2)}


# =============================================================================
#  OPTION CHAIN
# =============================================================================
def fetch_option_chain(ticker_str: str, cmp: float, signal: str) -> dict:
    def est():
        lot=round(cmp/50)*50; p=round(cmp*.015,2)
        return {"expiry":"Near-Month","atm_strike":lot,"call_premium":p,"call_iv":"~20%",
                "call_oi":"N/A","put_premium":p,"put_iv":"~20%","put_oi":"N/A",
                "pcr":"N/A","call_oi_int":0,"put_oi_int":0}
    if not LIVE: return est()
    try:
        t=yf.Ticker(ticker_str); exps=t.options
        if not exps: return est()
        ch=t.option_chain(exps[0])
        calls=ch.calls.copy().fillna(0); puts=ch.puts.copy().fillna(0)
        if calls.empty or puts.empty: return est()
        calls["dist"]=(calls["strike"]-cmp).abs(); puts["dist"]=(puts["strike"]-cmp).abs()
        ac=calls.nsmallest(1,"dist").iloc[0]; ap=puts.nsmallest(1,"dist").iloc[0]
        coi=int(calls["openInterest"].sum()); poi=int(puts["openInterest"].sum())
        return {"expiry":exps[0],"atm_strike":float(ac["strike"]),
                "call_premium":round(float(ac["lastPrice"]),2),
                "call_iv":f"{round(float(ac['impliedVolatility'])*100,1)}%",
                "call_oi":f"{int(ac['openInterest']):,}","call_oi_int":int(ac["openInterest"]),
                "put_premium":round(float(ap["lastPrice"]),2),
                "put_iv":f"{round(float(ap['impliedVolatility'])*100,1)}%",
                "put_oi":f"{int(ap['openInterest']):,}","put_oi_int":int(ap["openInterest"]),
                "pcr":round(poi/max(coi,1),2)}
    except: return est()

def option_rec(signal, opt):
    if signal in ("STRONG BUY","BUY"):
        p=opt["call_premium"]
        return {"action":"BUY CALL (CE)","strike":opt["atm_strike"],"type":"CE",
                "premium":p,"iv":opt["call_iv"],"oi":opt["call_oi"],
                "target":round(p*1.8,2) if isinstance(p,float) else "-",
                "sl":round(p*.5,2) if isinstance(p,float) else "-",
                "color":"#4CAF50","badge":"📞 BUY CE"}
    elif signal in ("STRONG SELL","SELL"):
        p=opt["put_premium"]
        return {"action":"BUY PUT (PE)","strike":opt["atm_strike"],"type":"PE",
                "premium":p,"iv":opt["put_iv"],"oi":opt["put_oi"],
                "target":round(p*1.8,2) if isinstance(p,float) else "-",
                "sl":round(p*.5,2) if isinstance(p,float) else "-",
                "color":"#F44336","badge":"📉 BUY PE"}
    return {"action":"WAIT / HOLD","strike":"-","type":"-","premium":"-",
            "iv":"-","oi":"-","target":"-","sl":"-","color":"#FF9800","badge":"⏳ WAIT"}

def _mock(name, ticker):
    rng=np.random.default_rng(abs(hash(ticker))%9999)
    sc=int(rng.integers(-5,6)); c=round(float(rng.uniform(200,4000)),2)
    a=round(c*float(rng.uniform(.015,.035)),2); chg=round(float(rng.uniform(-3,3)),2)
    if sc>=5:    s,col,e="STRONG BUY","#00C853","🚀"
    elif sc>=2:  s,col,e="BUY","#4CAF50","🟢"
    elif sc<=-5: s,col,e="STRONG SELL","#D50000","⛔"
    elif sc<=-2: s,col,e="SELL","#F44336","🔴"
    else:        s,col,e="HOLD","#FF9800","🟡"
    lot=round(c/50)*50; p=round(c*.015,2)
    opt={"expiry":"Near-Month","atm_strike":lot,"call_premium":p,"call_iv":"~20%",
         "call_oi":"N/A","put_premium":p,"put_iv":"~20%","put_oi":"N/A",
         "pcr":round(float(rng.uniform(.5,2.0)),2),"call_oi_int":0,"put_oi_int":0}
    rec=option_rec(s,opt)
    return {"symbol":name,"ticker":ticker,"sector":get_sector(name),"signal":s,"color":col,
            "emoji":e,"score":sc,"cmp":c,"change_pct":chg,"entry":c,
            "target":round(c+a*3,2) if sc>=0 else round(c-a*3,2),
            "sl":round(c-a*1.5,2) if sc>=0 else round(c+a*1.5,2),
            "atr":a,"rsi":round(float(rng.uniform(25,75)),1),"e20":round(c*.98,2),"e50":round(c*.95,2),
            "reasons":["Demo — commit requirements.txt to GitHub"],"opt":opt,"rec":rec}


# =============================================================================
#  BATCH DATA LOADER
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_screener(_key: int):
    if not LIVE:
        return [_mock(v["name"],k) for k,v in INDICES.items()], \
               [_mock(s,f"{s}.NS") for s in FO_STOCKS]

    results={}
    all_ns=[f"{s}.NS" for s in FO_STOCKS]; idx_sym=list(INDICES.keys())
    all_sym=all_ns+idx_sym

    try:
        raw=yf.download(all_sym,period="60d",auto_adjust=True,
                        progress=False,group_by="ticker",threads=True)
    except: raw=None
    try:
        intra=yf.download(all_sym,period="1d",interval="5m",auto_adjust=True,
                          progress=False,group_by="ticker",threads=True)
    except: intra=None

    def get_hist(sym):
        try:
            if raw is None: return None
            d=raw[sym] if sym in raw else None
            if d is None or (hasattr(d,'empty') and d.empty): return None
            needed=[c for c in ["Close","High","Low","Volume","Open"] if c in d.columns]
            return d[needed].dropna(subset=["Close"])
        except: return None

    def get_cmp(sym):
        try:
            if intra is None: return None
            d=intra[sym] if sym in intra else None
            if d is None or (hasattr(d,'empty') and d.empty): return None
            return float(d["Close"].dropna().iloc[-1])
        except: return None

    for sym in FO_STOCKS:
        ns=f"{sym}.NS"; hist=get_hist(ns)
        if hist is not None and len(hist)>=35:
            sig=generate_signal(hist)
            if sig:
                lc=get_cmp(ns)
                if lc: sig["cmp"]=round(lc,2)
                sig["symbol"]=sym; sig["ticker"]=ns; sig["sector"]=get_sector(sym)
                results[sym]=sig; continue
        results[sym]=_mock(sym,ns)

    idx_results=[]
    for sym,meta in INDICES.items():
        hist=get_hist(sym)
        if hist is not None and len(hist)>=35:
            sig=generate_signal(hist)
            if sig:
                lc=get_cmp(sym)
                if lc: sig["cmp"]=round(lc,2)
                sig["symbol"]=meta["name"]; sig["ticker"]=sym; sig["sector"]="Index"
                idx_results.append(sig); continue
        idx_results.append(_mock(meta["name"],sym))

    actionable=[r for r in results.values() if r["signal"] in ("STRONG BUY","BUY","SELL","STRONG SELL")]
    def fetch_opt(r):
        opt=fetch_option_chain(r["ticker"],r["cmp"],r["signal"])
        return r["symbol"],opt,option_rec(r["signal"],opt)
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs={ex.submit(fetch_opt,r):r["symbol"] for r in actionable}
        for f in as_completed(futs):
            try:
                sym,opt,rec=f.result()
                results[sym]["opt"]=opt; results[sym]["rec"]=rec
            except: pass

    blank_opt=lambda cmp:{"expiry":"N/A","atm_strike":round(cmp/50)*50,
                           "call_premium":round(cmp*.015,2),"call_iv":"~20%","call_oi":"N/A",
                           "put_premium":round(cmp*.015,2),"put_iv":"~20%","put_oi":"N/A",
                           "pcr":"N/A","call_oi_int":0,"put_oi_int":0}
    for sym,r in results.items():
        if "opt" not in r:
            o=blank_opt(r["cmp"]); results[sym]["opt"]=o; results[sym]["rec"]=option_rec(r["signal"],o)
    for r in idx_results:
        if "opt" not in r:
            o=blank_opt(r["cmp"]); r["opt"]=o; r["rec"]=option_rec(r["signal"],o)

    order={"STRONG BUY":0,"BUY":1,"HOLD":2,"SELL":3,"STRONG SELL":4}
    return (sorted(idx_results,key=lambda x:order.get(x["signal"],2)),
            sorted(results.values(),key=lambda x:order.get(x["signal"],2)))


# =============================================================================
#  CHART DATA FETCHER  (for the Chart tab — 6 months OHLCV)
# =============================================================================
@st.cache_data(ttl=180, show_spinner=False)
def fetch_chart_data(ticker_str: str, period: str = "3mo") -> pd.DataFrame | None:
    if not LIVE: return None
    try:
        t=yf.Ticker(ticker_str)
        hist=t.history(period=period,auto_adjust=True)
        if hist.empty: return None
        if "Open" not in hist.columns:
            hist["Open"]=hist["Close"].shift(1).fillna(hist["Close"])
        return hist
    except: return None


# =============================================================================
#  FUNDAMENTALS
# =============================================================================
def get_fundamentals(info):
    def pct(v):
        try: return f"{float(v)*100:.2f}%" if v else "N/A"
        except: return "N/A"
    def fmt(v):
        try:
            n=float(v)
            if n>=1e12: return f"₹{n/1e12:.2f}T"
            if n>=1e9:  return f"₹{n/1e9:.2f}B"
            if n>=1e7:  return f"₹{n/1e7:.2f}Cr"
            return f"₹{n:,.0f}"
        except: return "N/A"
    def val(k,d=2):
        v=info.get(k)
        try: return round(float(v),d) if v else "N/A"
        except: return "N/A"
    pe=val("trailingPE",1); roe=info.get("returnOnEquity"); de=val("debtToEquity",2)
    fs=0
    if pe!="N/A":
        if float(pe)<15:fs+=2
        elif float(pe)<25:fs+=1
        elif float(pe)>50:fs-=2
        elif float(pe)>35:fs-=1
    if roe:
        rf=float(roe)
        if rf>0.20:fs+=2
        elif rf>0.12:fs+=1
        elif rf<0:fs-=1
    if de!="N/A":
        if float(de)<50:fs+=1
        elif float(de)>200:fs-=1
    if fs>=3:fv="💪 Fundamentally Strong"
    elif fs>=1:fv="✅ Fundamentally Decent"
    elif fs<=-2:fv="⚠️ Fundamentally Weak"
    else:fv="🔄 Mixed Fundamentals"
    return {"company":info.get("longName","N/A"),"sector":info.get("sector","N/A"),
            "industry":info.get("industry","N/A"),"market_cap":fmt(info.get("marketCap")),
            "pe":pe,"pb":val("priceToBook",2),"eps":val("trailingEps",2),"roe":pct(roe),
            "de":de,"div_yield":pct(info.get("dividendYield")),"revenue":fmt(info.get("totalRevenue")),
            "net_income":fmt(info.get("netIncomeToCommon")),"w52_high":val("fiftyTwoWeekHigh",2),
            "w52_low":val("fiftyTwoWeekLow",2),"book_value":val("bookValue",2),
            "fund_view":fv,"f_score":fs,
            "desc":(info.get("longBusinessSummary","")[:280]+"…") if info.get("longBusinessSummary") else "N/A"}

@st.cache_data(ttl=120, show_spinner=False)
def search_analysis(ticker_str: str):
    if not LIVE: return None
    try:
        t=yf.Ticker(ticker_str)
        hist=t.history(period="6mo",auto_adjust=True)
        if hist.empty or len(hist)<35: return None
        if "Open" not in hist.columns:
            hist["Open"]=hist["Close"].shift(1).fillna(hist["Close"])
        try:
            intra=t.history(period="1d",interval="5m",auto_adjust=True)
            cmp=float(intra["Close"].iloc[-1]) if not intra.empty else float(hist["Close"].iloc[-1])
        except: cmp=float(hist["Close"].iloc[-1])
        sig=generate_signal(hist)
        if not sig: return None
        sig["cmp"]=round(cmp,2)
        opt=fetch_option_chain(ticker_str,cmp,sig["signal"])
        rec=option_rec(sig["signal"],opt)
        try: fund=get_fundamentals(t.info)
        except: fund={"company":"N/A","sector":"N/A","industry":"N/A","market_cap":"N/A",
                      "pe":"N/A","pb":"N/A","eps":"N/A","roe":"N/A","de":"N/A",
                      "div_yield":"N/A","revenue":"N/A","net_income":"N/A",
                      "w52_high":"N/A","w52_low":"N/A","book_value":"N/A",
                      "fund_view":"N/A","f_score":0,"desc":"N/A"}
        ts=sig["score"]+fund["f_score"]
        if ts>=5:cv="🚀 STRONG BULLISH"
        elif ts>=2:cv="🟢 BULLISH"
        elif ts<=-5:cv="⛔ STRONG BEARISH"
        elif ts<=-2:cv="🔴 BEARISH"
        else:cv="🟡 NEUTRAL"
        return {**sig,"fund":fund,"hist":hist,"opt":opt,"rec":rec,"combined":cv}
    except: return None


# =============================================================================
#  SESSION STATE
# =============================================================================
for k,v in [("cache_key",0),("last_ref",time_ist()),("chart_sym","RELIANCE"),
             ("chart_period","3mo")]:
    if k not in st.session_state: st.session_state[k]=v

# =============================================================================
#  HEADER
# =============================================================================
h1,h2=st.columns([5,2])
with h1:
    st.markdown("""<div style='padding:.3rem 0;'>
    <div style='font-size:.6rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#6366F1;margin-bottom:4px;'>NSE · F&O · TRADINGVIEW-STYLE SIGNALS</div>
    <h1 style='font-size:1.9rem;font-weight:900;margin:0;letter-spacing:-.03em;
       background:linear-gradient(135deg,#F8FAFC,#94A3B8);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>F&O Signal Screener</h1>
    <p style='color:#475569;font-size:.75rem;margin:4px 0 0;'>Candlestick · EMA · Bollinger · RSI · MACD · CALL/PUT Signals</p>
    </div>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""<div style='text-align:right;padding-top:.7rem;'>
    <div style='font-size:.56rem;color:#64748B;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>🕐 Indian Standard Time</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:#6366F1;font-weight:700;margin-top:2px;'>{now_ist()}</div>
    <div style='font-size:.58rem;color:#334155;margin-top:2px;'>Refreshed: {st.session_state['last_ref']}</div>
    </div>""", unsafe_allow_html=True)
st.divider()

# =============================================================================
#  SEARCH BAR
# =============================================================================
st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:5px;'>🔍 Deep Stock Analysis — Search Any NSE Stock or Index</div>", unsafe_allow_html=True)
sb1,sb2=st.columns([5,1])
with sb1:
    q=st.text_input("Search",placeholder="e.g. RELIANCE  ·  TCS  ·  NIFTY  ·  BANKNIFTY  ·  HDFCBANK",label_visibility="collapsed")
with sb2:
    do_search=st.button("🔍 Analyse",use_container_width=True)

raw_q=(q.strip().upper() if do_search or q else "")
if raw_q:
    resolved=SEARCH_ALIASES.get(raw_q, raw_q if raw_q.startswith("^") else raw_q+".NS")
    with st.spinner(f"⏳ Analysing {raw_q}…"):
        ana=search_analysis(resolved)
    if ana:
        f=ana["fund"]; sig=ana["signal"]
        SBD={"STRONG BUY":"#00C853","BUY":"#4CAF50","HOLD":"#FF9800","SELL":"#F44336","STRONG SELL":"#D50000"}
        bd=SBD.get(sig,"#FF9800"); cc="#4CAF50" if ana["change_pct"]>=0 else "#F44336"
        st.markdown(f"""<div style='background:linear-gradient(135deg,#0D1117,#111827);border:1.5px solid {bd};
            border-radius:14px;padding:1rem 1.4rem;margin:8px 0 6px;position:relative;overflow:hidden;'>
          <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{bd},{bd}33);'></div>
          <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
            <div>
              <div style='font-size:.58rem;color:#6366F1;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'>{f['sector']} · {f['industry']}</div>
              <div style='font-size:1.3rem;font-weight:900;color:#F8FAFC;margin:2px 0;'>{f['company']}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:1.7rem;font-weight:800;color:#F8FAFC;'>
                ₹{ana['cmp']:,.2f}<span style='font-size:.82rem;color:{cc};margin-left:8px;'>
                {'▲' if ana['change_pct']>=0 else '▼'} {abs(ana['change_pct'])}%</span></div>
            </div>
            <div style='text-align:center;'>
              <div style='background:{bd};color:{'#000' if sig in ('STRONG BUY','BUY','HOLD') else '#fff'};
                  font-size:.95rem;font-weight:800;padding:7px 18px;border-radius:22px;'>{ana['emoji']} {sig}</div>
              <div style='font-size:.65rem;color:#64748B;margin-top:4px;'>{ana['combined']}</div>
            </div>
          </div></div>""", unsafe_allow_html=True)

        # TradingView chart for search result
        st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 4px;'>📈 TradingView-Style Chart (6 Months)</div>", unsafe_allow_html=True)
        fig=create_tv_chart(ana["hist"], raw_q, ana, "6 Months")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True,"scrollZoom":True})

        # Trade levels + Option rec
        tc1,tc2,tc3=st.columns([2,2,1.8])
        with tc1:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin-bottom:6px;'>🎯 Trade Levels</div>", unsafe_allow_html=True)
            l1,l2,l3=st.columns(3)
            l1.metric("Entry",f"₹{ana['entry']:,.2f}")
            l2.metric("Target",f"₹{ana['target']:,.2f}")
            l3.metric("Stop Loss",f"₹{ana['sl']:,.2f}")
            rec=ana["rec"]; opt=ana["opt"]
            st.markdown(f"""<div style='background:#0D1117;border:1px solid {rec['color']};border-radius:12px;padding:10px;margin-top:8px;'>
              <div style='font-size:.9rem;font-weight:800;color:{rec['color']};margin-bottom:6px;'>{rec['badge']} — {opt.get('expiry','N/A')}</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:5px;'>
                <div style='background:#060B14;border-radius:8px;padding:5px 8px;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Strike</div><div style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:#E2E8F0;font-weight:700;'>₹{rec['strike']}</div></div>
                <div style='background:#060B14;border-radius:8px;padding:5px 8px;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Premium</div><div style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:{rec['color']};font-weight:700;'>₹{rec['premium']}</div></div>
                <div style='background:#060B14;border-radius:8px;padding:5px 8px;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Tgt Premium</div><div style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:#4CAF50;font-weight:700;'>₹{rec['target']}</div></div>
                <div style='background:#060B14;border-radius:8px;padding:5px 8px;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>SL Premium</div><div style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:#F44336;font-weight:700;'>₹{rec['sl']}</div></div>
              </div>
              <div style='margin-top:6px;font-size:.62rem;color:#64748B;'>IV: {rec['iv']} &nbsp;·&nbsp; OI: {rec['oi']} &nbsp;·&nbsp; PCR: {opt.get('pcr','N/A')}</div>
            </div>""", unsafe_allow_html=True)

        with tc2:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin-bottom:6px;'>🔬 Technical</div>", unsafe_allow_html=True)
            def trow(l,v,b=None):
                c_="#4CAF50" if b is True else "#F44336" if b is False else "#94A3B8"
                return f"<div style='display:flex;justify-content:space-between;padding:5px 9px;border-bottom:1px solid #1E293B;'><span style='font-size:.7rem;color:#64748B;'>{l}</span><span style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:{c_};font-weight:600;'>{v}</span></div>"
            rows=[trow("RSI (14)",f"{ana['rsi']}",ana['rsi']<50),
                  trow("MACD","Bullish ✅" if ana['score']>0 else "Bearish ❌",ana['score']>0),
                  trow("EMA 20",f"₹{ana['e20']:,.2f}",ana['cmp']>ana['e20']),
                  trow("EMA 50",f"₹{ana['e50']:,.2f}",ana['cmp']>ana['e50']),
                  trow("ATR",f"₹{ana['atr']:,.2f}"),
                  trow("Signal Score",f"{ana['score']} / 8"),
                  trow("Call IV",opt.get('call_iv','N/A')),
                  trow("Put IV",opt.get('put_iv','N/A')),
                  trow("PCR",str(opt.get('pcr','N/A')))]
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>{''.join(rows)}</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:8px 0 4px;'>💡 Reasons</div>", unsafe_allow_html=True)
            for r in ana["reasons"]: st.markdown(f"<div style='font-size:.7rem;color:#94A3B8;padding:2px 0;'>• {r}</div>", unsafe_allow_html=True)

        with tc3:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin-bottom:6px;'>📊 Fundamentals</div>", unsafe_allow_html=True)
            def frow(l,v): return f"<div style='display:flex;justify-content:space-between;padding:5px 9px;border-bottom:1px solid #1E293B;'><span style='font-size:.68rem;color:#64748B;'>{l}</span><span style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:#E2E8F0;font-weight:600;'>{v}</span></div>"
            fr=[frow("Market Cap",f["market_cap"]),frow("P/E",str(f["pe"])),frow("P/B",str(f["pb"])),
                frow("EPS",str(f["eps"])),frow("ROE",f["roe"]),frow("D/E",str(f["de"])),
                frow("Revenue",f["revenue"]),frow("Net Inc.",f["net_income"]),
                frow("Div Yield",f["div_yield"]),frow("52W Hi",f"₹{f['w52_high']}"),
                frow("52W Lo",f"₹{f['w52_low']}")]
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>{''.join(fr)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;padding:8px;margin-top:8px;text-align:center;'><div style='font-size:.55rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>Fundamental View</div><div style='font-size:.8rem;font-weight:700;color:#6366F1;margin-top:2px;'>{f['fund_view']}</div></div>", unsafe_allow_html=True)
    elif LIVE:
        st.warning(f"⚠️ No data for **'{raw_q}'**. Try: RELIANCE · TCS · HDFCBANK · NIFTY · BANKNIFTY")
    st.divider()

# =============================================================================
#  REFRESH CONTROLS
# =============================================================================
ct1,ct2,ct3=st.columns([2,2,2])
with ct1:
    if st.button("🔄  REFRESH ALL SIGNALS",use_container_width=True):
        st.cache_data.clear(); st.session_state["cache_key"]+=1
        st.session_state["last_ref"]=time_ist(); st.rerun()
with ct2:
    filter_sig=st.selectbox("Signal Filter",["All Signals","STRONG BUY","BUY","HOLD","SELL","STRONG SELL"],label_visibility="collapsed")
with ct3:
    filter_sec=st.selectbox("Sector Filter",["All Sectors"]+list(SECTOR_MAP.keys()),label_visibility="collapsed")
st.divider()

if not LIVE:
    st.warning("⚠️ Demo data — commit `requirements.txt` to GitHub.",icon="📦")

with st.spinner("⏳ Loading signals for all stocks…"):
    idx_sigs,stk_sigs=load_screener(_key=st.session_state["cache_key"])

def filt(lst):
    r=lst
    if filter_sig!="All Signals": r=[x for x in r if x["signal"]==filter_sig]
    if filter_sec!="All Sectors":  r=[x for x in r if x.get("sector")==filter_sec]
    return r
fi=filt(idx_sigs); fs=filt(stk_sigs)

all_s=idx_sigs+stk_sigs
cnts={s:sum(1 for x in all_s if x["signal"]==s) for s in ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]}
m1,m2,m3,m4,m5,m6=st.columns(6)
with m1: st.metric("🚀 Strong Buy",  cnts["STRONG BUY"])
with m2: st.metric("🟢 Buy",         cnts["BUY"])
with m3: st.metric("🟡 Hold",        cnts["HOLD"])
with m4: st.metric("🔴 Sell",        cnts["SELL"])
with m5: st.metric("⛔ Strong Sell", cnts["STRONG SELL"])
with m6: st.metric("📋 Universe",    len(all_s))
st.divider()

# Index cards
SBD={"STRONG BUY":"#00C853","BUY":"#4CAF50","HOLD":"#FF9800","SELL":"#F44336","STRONG SELL":"#D50000"}
if fi:
    st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:8px;'>📊 Market Indices</div>", unsafe_allow_html=True)
    idx_meta={v["name"]:v for v in INDICES.values()}
    for i in range(0,len(fi),3):
        row=fi[i:i+3]; cols=st.columns(len(row))
        for col,d in zip(cols,row):
            bd=SBD.get(d["signal"],"#FF9800"); cc="#4CAF50" if d["change_pct"]>=0 else "#F44336"
            tc="#4CAF50" if d["score"]>=0 else "#F44336"; sc_="#F44336" if d["score"]>=0 else "#4CAF50"
            meta=idx_meta.get(d["symbol"],{"emoji":"📈"})
            col.markdown(f"""<div style='background:linear-gradient(135deg,#0D1117,#111827);border:1.5px solid {bd};border-radius:14px;padding:1.1rem;margin:4px 0;position:relative;overflow:hidden;'>
              <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{bd},{bd}33);'></div>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <div style='font-size:.56rem;color:#6366F1;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'>{meta.get('emoji','📈')} {d['symbol']}</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:800;color:#F8FAFC;'>{d['cmp']:,.2f}</div>
                  <div style='font-size:.78rem;color:{cc};font-weight:700;'>{'▲' if d['change_pct']>=0 else '▼'} {abs(d['change_pct'])}%</div>
                </div>
                <div style='text-align:center;'>
                  <div style='background:{bd};color:{'#000' if d['signal'] in ('STRONG BUY','BUY','HOLD') else '#fff'};font-size:.72rem;font-weight:800;padding:5px 12px;border-radius:18px;'>{d['emoji']} {d['signal']}</div>
                  <div style='font-size:.58rem;color:#64748B;margin-top:3px;'>RSI {d['rsi']}</div>
                </div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-top:8px;'>
                <div style='background:#060B14;border-radius:8px;padding:5px;text-align:center;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Entry</div><div style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:#E2E8F0;font-weight:700;'>₹{d['entry']:,.0f}</div></div>
                <div style='background:#060B14;border-radius:8px;padding:5px;text-align:center;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Target</div><div style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:{tc};font-weight:700;'>₹{d['target']:,.0f}</div></div>
                <div style='background:#060B14;border-radius:8px;padding:5px;text-align:center;'><div style='font-size:.52rem;color:#64748B;text-transform:uppercase;'>Stop Loss</div><div style='font-family:JetBrains Mono,monospace;font-size:.75rem;color:{sc_};font-weight:700;'>₹{d['sl']:,.0f}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.divider()

# =============================================================================
#  MAIN TABS
# =============================================================================
tab_chart,tab_all,tab_buy,tab_sell,tab_opt,tab_sec=st.tabs([
    "📈 Chart View",
    f"📋 All ({len(fs)})",
    f"🚀 BUY ({sum(1 for x in fs if x['signal'] in ('STRONG BUY','BUY'))})",
    f"⛔ SELL ({sum(1 for x in fs if x['signal'] in ('STRONG SELL','SELL'))})",
    "📞 Call / Put",
    "🏭 By Sector",
])

# ── CHART TAB ─────────────────────────────────────────────────────────────────
with tab_chart:
    st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:8px;'>📈 TradingView-Style Chart with All Indicators</div>", unsafe_allow_html=True)

    all_chart_symbols = (
        [("NIFTY 50","^NSEI"),("BANK NIFTY","^NSEBANK"),("NIFTY IT","^CNXIT"),
         ("NIFTY PHARMA","^CNXPHARMA"),("NIFTY AUTO","^CNXAUTO")] +
        [(s,f"{s}.NS") for s in FO_STOCKS]
    )
    sym_names=[s[0] for s in all_chart_symbols]
    sym_map  ={s[0]:s[1] for s in all_chart_symbols}

    cc1,cc2,cc3=st.columns([3,2,1])
    with cc1:
        default_idx=sym_names.index("RELIANCE") if "RELIANCE" in sym_names else 0
        chosen=st.selectbox("Select Stock / Index",sym_names,index=default_idx,label_visibility="collapsed")
    with cc2:
        period_opt=st.selectbox("Period",["1mo","3mo","6mo","1y","2y"],index=1,label_visibility="collapsed")
        PERIOD_LABELS={"1mo":"1 Month","3mo":"3 Months","6mo":"6 Months","1y":"1 Year","2y":"2 Years"}
    with cc3:
        load_chart=st.button("📈 Load Chart",use_container_width=True)

    chart_ticker=sym_map.get(chosen,"RELIANCE.NS")

    with st.spinner(f"⏳ Loading chart for {chosen}…"):
        chart_hist=fetch_chart_data(chart_ticker, period_opt)

    if chart_hist is not None and not chart_hist.empty:
        chart_sig=generate_signal(chart_hist)
        fig=create_tv_chart(chart_hist, chosen, chart_sig, PERIOD_LABELS.get(period_opt,""))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":True,"scrollZoom":True,
                                "modeBarButtonsToAdd":["drawline","drawopenpath","eraseshape"],
                                "toImageButtonOptions":{"format":"png","scale":2}})

        if chart_sig:
            sig=chart_sig["signal"]
            bd=SBD.get(sig,"#FF9800")
            cc="#4CAF50" if chart_sig["change_pct"]>=0 else "#F44336"
            i1,i2,i3,i4,i5=st.columns(5)
            i1.metric("Signal",f"{chart_sig['emoji']} {sig}")
            i2.metric("CMP",f"₹{chart_sig['cmp']:,.2f}",f"{'▲' if chart_sig['change_pct']>=0 else '▼'} {abs(chart_sig['change_pct'])}%")
            i3.metric("Entry → Target",f"₹{chart_sig['entry']:,.0f} → ₹{chart_sig['target']:,.0f}")
            i4.metric("Stop Loss",f"₹{chart_sig['sl']:,.0f}")
            i5.metric("RSI | Score",f"{chart_sig['rsi']} | {chart_sig['score']}/8")

            st.markdown("**Signal Reasons:**  " + "  ·  ".join(chart_sig["reasons"]))
    else:
        st.warning(f"⚠️ Could not load chart data for {chosen}. Try another symbol or period.")

# ── SIGNAL TABLES (shared renderer) ──────────────────────────────────────────
def make_table(data, show_options=False, table_key="default"):
    if not data: st.info("No stocks match the current filter."); return
    rows=[]
    for d in data:
        rec=d.get("rec",{}); opt=d.get("opt",{})
        row={"Symbol":d["symbol"],"Sector":d.get("sector","—"),
             "Signal":f"{d['emoji']} {d['signal']}","CMP (₹)":d["cmp"],
             "Change %":d["change_pct"],"RSI":d["rsi"],
             "Entry (₹)":d["entry"],"Target (₹)":d["target"],"SL (₹)":d["sl"],
             "Score":d["score"]}
        if show_options:
            row["Option"]=rec.get("badge","—")
            row["Strike"]=rec.get("strike","—")
            row["Expiry"]=str(opt.get("expiry","—"))
            row["Premium (₹)"]=rec.get("premium","—")
            row["Opt Target"]=rec.get("target","—")
            row["Opt SL"]=rec.get("sl","—")
            row["IV"]=opt.get("call_iv","—") if d["signal"] in ("STRONG BUY","BUY") else opt.get("put_iv","—")
            row["PCR"]=str(opt.get("pcr","—"))
        rows.append(row)
    df=pd.DataFrame(rows)
    num_cols=["CMP (₹)","Change %","RSI","Entry (₹)","Target (₹)","SL (₹)"]
    if show_options:
        for c in ["Premium (₹)","Opt Target","Opt SL"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
            num_cols.append(c)
    for c in num_cols:
        if c in df.columns: df[c]=df[c].astype("float64")
    df["Score"]=df["Score"].astype("int64")
    cfg={"Symbol":st.column_config.TextColumn(width="small"),
         "Sector":st.column_config.TextColumn(width="small"),
         "Signal":st.column_config.TextColumn(width="medium"),
         "CMP (₹)":st.column_config.NumberColumn(format="₹%.2f"),
         "Change %":st.column_config.NumberColumn(format="%.2f%%"),
         "RSI":st.column_config.NumberColumn(format="%.1f"),
         "Entry (₹)":st.column_config.NumberColumn(format="₹%.2f"),
         "Target (₹)":st.column_config.NumberColumn(format="₹%.2f"),
         "SL (₹)":st.column_config.NumberColumn(format="₹%.2f"),
         "Score":st.column_config.NumberColumn(format="%d")}
    if show_options:
        cfg["Premium (₹)"]=st.column_config.NumberColumn(format="₹%.2f")
        cfg["Opt Target"]=st.column_config.NumberColumn(format="₹%.2f")
        cfg["Opt SL"]=st.column_config.NumberColumn(format="₹%.2f")
    st.dataframe(df,column_config=cfg,use_container_width=True,
                  hide_index=True,height=min(80+38*len(df),700))
    st.download_button("⬇ Export CSV",df.to_csv(index=False).encode("utf-8"),
                        "fo_signals.csv","text/csv",key=f"dl_{table_key}")

with tab_all:  make_table(fs, table_key="all")
with tab_buy:
    bs=[x for x in fs if x["signal"] in ("STRONG BUY","BUY")]
    st.markdown(f"<div style='font-size:.6rem;color:#4CAF50;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>🟢 {len(bs)} BUY signals</div>", unsafe_allow_html=True)
    make_table(bs, table_key="buy")
with tab_sell:
    ss=[x for x in fs if x["signal"] in ("STRONG SELL","SELL")]
    st.markdown(f"<div style='font-size:.6rem;color:#F44336;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>🔴 {len(ss)} SELL signals</div>", unsafe_allow_html=True)
    make_table(ss, table_key="sell")
with tab_opt:
    os_=[x for x in fs if x["signal"] in ("STRONG BUY","BUY","STRONG SELL","SELL")]
    st.markdown("""<div style='background:#0D1117;border:1px solid #1E293B;border-radius:10px;padding:9px 12px;margin-bottom:8px;font-size:.7rem;color:#94A3B8;line-height:1.7;'>
    📞 <b style='color:#4CAF50;'>BUY CALL (CE)</b> on BUY signals &nbsp;·&nbsp;
    📉 <b style='color:#F44336;'>BUY PUT (PE)</b> on SELL signals &nbsp;·&nbsp;
    <span style='color:#64748B;font-size:.62rem;'>Tgt = 1.8× Premium · SL = 0.5× Premium · ATM strike nearest expiry</span>
    </div>""", unsafe_allow_html=True)
    make_table(os_, show_options=True, table_key="opts")
with tab_sec:
    for sector,stocks in SECTOR_MAP.items():
        sd=[x for x in fs if x.get("sector")==sector]
        if not sd: continue
        buys=sum(1 for x in sd if x["signal"] in ("STRONG BUY","BUY"))
        sells=sum(1 for x in sd if x["signal"] in ("STRONG SELL","SELL"))
        mood="🟢 Bullish" if buys>sells else "🔴 Bearish" if sells>buys else "🟡 Mixed"
        with st.expander(f"🏭 {sector}  ·  {len(sd)} stocks  ·  {mood}  ({buys}▲ / {sells}▼)",expanded=False):
            make_table(sd, table_key=f"sec_{sector.replace(' ','_')}")

st.divider()
st.markdown("""<div style='text-align:center;padding:.8rem 0 .2rem;'>
<p style='font-size:.56rem;color:#1E293B;font-family:JetBrains Mono,monospace;letter-spacing:.08em;line-height:1.8;'>
⚠️ FOR EDUCATIONAL PURPOSES ONLY · NOT SEBI-REGISTERED ADVICE<br>
DATA VIA YAHOO FINANCE · ALWAYS DO YOUR OWN RESEARCH BEFORE TRADING
</p></div>""", unsafe_allow_html=True)
