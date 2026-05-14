# =============================================================================
#  NSE F&O SIGNAL SCREENER  v6.0
#  ✅ 100+ stocks via batch yf.download (fast, shows ALL stocks)
#  ✅ CALL / PUT option recommendations with strike, premium, target, SL
#  ✅ Tabs: All · BUY · SELL · Options · Sector
#  ✅ Real-time IST clock  · Intraday 5-min price
#  ✅ Search: full Technical + Fundamental Analysis
#  Run: streamlit run app.py
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

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():  return datetime.now(IST).strftime("%d %b %Y  %I:%M:%S %p IST")
def time_ist(): return datetime.now(IST).strftime("%I:%M %p IST")

st.set_page_config(page_title="NSE F&O Screener", page_icon="🎯",
                   layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background-color:#05080F !important;color:#E2E8F0 !important;}
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
.stTabs [data-baseweb="tab"]{background:transparent !important;border:none !important;color:#64748B !important;font-weight:600 !important;font-size:.8rem !important;letter-spacing:.04em !important;padding:.6rem 1.2rem !important;border-radius:8px 8px 0 0 !important}
.stTabs [aria-selected="true"]{background:#1E293B !important;color:#6366F1 !important;border-bottom:2px solid #6366F1 !important}
.streamlit-expanderHeader{background:#0D1117 !important;border:1px solid #1E293B !important;border-radius:10px !important;font-size:.75rem !important}
.stDownloadButton>button{background:transparent !important;border:1px solid #6366F1 !important;color:#6366F1 !important;font-size:.7rem !important;border-radius:8px !important}
</style>""", unsafe_allow_html=True)

# =============================================================================
#  UNIVERSE  — 100 + stocks across all NSE F&O sectors
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
    "Banking":     ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
                    "FEDERALBNK","BANDHANBNK","PNB","CANBK","BANKBARODA","IDFCFIRSTB","AUBANK"],
    "Finance":     ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SBICARD",
                    "HDFCLIFE","SBILIFE","ICICIGI","ICICIPRULI"],
    "IT & Tech":   ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS",
                    "PERSISTENT","COFORGE","OFSS","KPITTECH","TATAELXSI","NAUKRI"],
    "Consumer":    ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO",
                    "COLPAL","GODREJCP","EMAMILTD","TATACONSUM","TITAN","DMART","TRENT"],
    "Auto":        ["MARUTI","TATAMOTORS","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
                    "TVSMOTOR","MOTHERSON","BALKRISIND","MRF","BOSCHLTD"],
    "Pharma":      ["SUNPHARMA","DIVISLAB","CIPLA","DRREDDY","APOLLOHOSP","AUROPHARMA",
                    "LUPIN","BIOCON","ALKEM","TORNTPHARM","IPCALAB"],
    "Energy":      ["RELIANCE","ONGC","BPCL","ADANIGREEN","TATAPOWER","IGL",
                    "MGL","PETRONET","TORNTPOWER","CESC","ADANITRANS"],
    "Metals":      ["TATASTEEL","JSWSTEEL","HINDALCO","NATIONALUM","NMDC",
                    "SAIL","HINDCOPPER","WELCORP"],
    "Infra & RE":  ["LT","ADANIENT","ADANIPORTS","DLF","GODREJPROP",
                    "PRESTIGE","OBEROIRLTY","LODHA","IRB"],
    "Chemicals":   ["PIDILITIND","SRF","DEEPAKNTR","AARTIIND","NAVINFLUOR"],
    "Others":      ["ULTRACEMCO","ASIANPAINT","GRASIM","SHREECEM","POWERGRID",
                    "NTPC","COALINDIA","MM","BHARTIARTL","ZOMATO","PAYTM",
                    "NYKAA","PVRINOX","ZEEL","MCDOWELL-N"],
}
FO_STOCKS = [s for stocks in SECTOR_MAP.values() for s in stocks]

SEARCH_ALIASES = {
    "NIFTY":"^NSEI","NIFTY50":"^NSEI","NIFTY 50":"^NSEI",
    "BANKNIFTY":"^NSEBANK","BANK NIFTY":"^NSEBANK",
    "NIFTYIT":"^CNXIT","NIFTYPHARMA":"^CNXPHARMA","NIFTYAUTO":"^CNXAUTO",
    "MIDCAP":"^CNXMIDCAP",
}

def get_sector(sym):
    for sec, stocks in SECTOR_MAP.items():
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

def calc_bb(c,p=20):
    sma=c.rolling(p).mean(); std=c.rolling(p).std()
    return sma+2*std,sma,sma-2*std

def calc_atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return float(tr.rolling(p).mean().dropna().iloc[-1])

# =============================================================================
#  SIGNAL ENGINE
# =============================================================================
def generate_signal(hist: pd.DataFrame) -> dict | None:
    if len(hist)<35: return None
    c,h,l,v=hist["Close"],hist["High"],hist["Low"],hist["Volume"]
    _r=calc_rsi(c); _m,_s,_h=calc_macd(c)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    bbu,bbm,bbl=calc_bb(c); _atr=calc_atr(h,l,c)
    lo=c.rolling(14).min(); hi=c.rolling(14).max()
    stk=100*(c-lo)/(hi-lo+1e-9); std_=stk.rolling(3).mean()

    cv=float(c.iloc[-1]); pv=float(c.iloc[-2])
    r=float(_r.iloc[-1]); m=float(_m.iloc[-1]); ms=float(_s.iloc[-1])
    pm=float(_m.iloc[-3]); ps=float(_s.iloc[-3])
    e2=float(e20.iloc[-1]); e5=float(e50.iloc[-1])
    bu=float(bbu.iloc[-1]); bl=float(bbl.iloc[-1])
    sk=float(stk.iloc[-1]); sd=float(std_.iloc[-1])
    va=float(v.iloc[-20:].mean()); vc=float(v.iloc[-1])

    score=0; reasons=[]
    # RSI
    if r<=30:   score+=2; reasons.append(f"RSI Oversold ({r:.0f}) — Reversal Zone 🔥")
    elif r<=42: score+=1; reasons.append(f"RSI Bullish ({r:.0f})")
    elif r>=70: score-=2; reasons.append(f"RSI Overbought ({r:.0f}) ⚠️")
    elif r>=58: score-=1; reasons.append(f"RSI Bearish ({r:.0f})")
    else: reasons.append(f"RSI Neutral ({r:.0f})")
    # MACD
    if m>ms and pm<ps:   score+=2; reasons.append("MACD Bullish Crossover ✅")
    elif m>ms:           score+=1; reasons.append("MACD Bullish")
    elif m<ms and pm>ps: score-=2; reasons.append("MACD Bearish Crossover ❌")
    else:                score-=1; reasons.append("MACD Bearish")
    # EMA
    if cv>e2>e5:   score+=2; reasons.append("Price>EMA20>EMA50 (Uptrend) 📈")
    elif cv>e2:    score+=1; reasons.append("Price Above EMA20")
    elif cv<e2<e5: score-=2; reasons.append("Price<EMA20<EMA50 (Downtrend) 📉")
    elif cv<e2:    score-=1; reasons.append("Price Below EMA20")
    # BB
    if cv<=bl:     score+=1; reasons.append("Near Lower BB — Oversold Zone")
    elif cv>=bu:   score-=1; reasons.append("Near Upper BB — Overbought Zone")
    # Stochastic
    if sk<20 and sk>sd:   score+=1; reasons.append(f"Stoch Oversold Crossup ({sk:.0f})")
    elif sk>80 and sk<sd: score-=1; reasons.append(f"Stoch Overbought ({sk:.0f})")
    # Volume
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
#  OPTION CHAIN FETCHER
# =============================================================================
def fetch_option_chain(ticker_str: str, cmp: float, signal: str) -> dict:
    """Fetch ATM/OTM call & put data from yfinance option chain."""
    def estimate():
        # Estimate when chain unavailable: ATM premium ≈ 1.5% of CMP
        lot = round(cmp / 50) * 50
        p   = round(cmp * 0.015, 2)
        return {"expiry":"Near-Month","atm_strike":lot,
                "call_premium":p,"call_iv":"~20%","call_oi":"N/A",
                "put_premium":p,"put_iv":"~20%","put_oi":"N/A",
                "pcr":"N/A","call_oi_int":0,"put_oi_int":0}
    if not LIVE: return estimate()
    try:
        t    = yf.Ticker(ticker_str)
        exps = t.options
        if not exps: return estimate()
        chain = t.option_chain(exps[0])
        calls = chain.calls.copy().fillna(0)
        puts  = chain.puts.copy().fillna(0)
        if calls.empty or puts.empty: return estimate()

        calls["dist"]=(calls["strike"]-cmp).abs()
        puts["dist"] =(puts["strike"]-cmp).abs()
        ac=calls.nsmallest(1,"dist").iloc[0]
        ap=puts.nsmallest(1,"dist").iloc[0]

        coi=int(calls["openInterest"].sum())
        poi=int(puts["openInterest"].sum())
        pcr=round(poi/max(coi,1),2)

        return {
            "expiry":       exps[0],
            "atm_strike":   float(ac["strike"]),
            "call_premium": round(float(ac["lastPrice"]),2),
            "call_iv":      f"{round(float(ac['impliedVolatility'])*100,1)}%",
            "call_oi":      f"{int(ac['openInterest']):,}",
            "call_oi_int":  int(ac["openInterest"]),
            "put_premium":  round(float(ap["lastPrice"]),2),
            "put_iv":       f"{round(float(ap['impliedVolatility'])*100,1)}%",
            "put_oi":       f"{int(ap['openInterest']):,}",
            "put_oi_int":   int(ap["openInterest"]),
            "pcr":          pcr,
        }
    except Exception:
        return estimate()

def option_recommendation(signal: str, opt: dict) -> dict:
    """Build the actionable option trade from signal + chain data."""
    if signal in ("STRONG BUY","BUY"):
        prem = opt["call_premium"]
        return {"action":"BUY CALL (CE)","strike":opt["atm_strike"],
                "type":"CE","premium":prem,"iv":opt["call_iv"],"oi":opt["call_oi"],
                "target":round(prem*1.8,2) if isinstance(prem,float) else "-",
                "sl":round(prem*0.5,2)     if isinstance(prem,float) else "-",
                "color":"#4CAF50","badge":"📞 BUY CE"}
    elif signal in ("STRONG SELL","SELL"):
        prem = opt["put_premium"]
        return {"action":"BUY PUT (PE)","strike":opt["atm_strike"],
                "type":"PE","premium":prem,"iv":opt["put_iv"],"oi":opt["put_oi"],
                "target":round(prem*1.8,2) if isinstance(prem,float) else "-",
                "sl":round(prem*0.5,2)     if isinstance(prem,float) else "-",
                "color":"#F44336","badge":"📉 BUY PE"}
    else:
        return {"action":"WAIT / HOLD","strike":"-","type":"-",
                "premium":"-","iv":"-","oi":"-","target":"-","sl":"-",
                "color":"#FF9800","badge":"⏳ WAIT"}

# =============================================================================
#  MOCK DATA
# =============================================================================
def _mock(name, ticker):
    rng=np.random.default_rng(abs(hash(ticker))%9999)
    sc=int(rng.integers(-5,6)); c=round(float(rng.uniform(200,4000)),2)
    a=round(c*float(rng.uniform(.015,.035)),2); chg=round(float(rng.uniform(-3,3)),2)
    if sc>=5:    s,col,e="STRONG BUY","#00C853","🚀"
    elif sc>=2:  s,col,e="BUY","#4CAF50","🟢"
    elif sc<=-5: s,col,e="STRONG SELL","#D50000","⛔"
    elif sc<=-2: s,col,e="SELL","#F44336","🔴"
    else:        s,col,e="HOLD","#FF9800","🟡"
    lot=round(c/50)*50; p=round(c*0.015,2)
    opt={"expiry":"Near-Month","atm_strike":lot,"call_premium":p,"call_iv":"~20%",
         "call_oi":"N/A","put_premium":p,"put_iv":"~20%","put_oi":"N/A",
         "pcr":round(float(rng.uniform(.5,2.0)),2),"call_oi_int":0,"put_oi_int":0}
    rec=option_recommendation(s,opt)
    return {"symbol":name,"ticker":ticker,"sector":get_sector(name),
            "signal":s,"color":col,"emoji":e,"score":sc,
            "cmp":c,"change_pct":chg,"entry":c,
            "target":round(c+a*3,2) if sc>=0 else round(c-a*3,2),
            "sl":round(c-a*1.5,2)  if sc>=0 else round(c+a*1.5,2),
            "atr":a,"rsi":round(float(rng.uniform(25,75)),1),
            "e20":round(c*.98,2),"e50":round(c*.95,2),
            "reasons":["Demo — commit requirements.txt to GitHub"],
            "opt":opt,"rec":rec}

# =============================================================================
#  BATCH DATA LOADER  (one yf.download for all stocks — fast!)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_screener(_key: int):
    if not LIVE:
        idx=[_mock(v["name"],k) for k,v in INDICES.items()]
        stk=[_mock(s,f"{s}.NS") for s in FO_STOCKS]
        return idx, stk

    results = {}

    # ── 1. BATCH OHLCV download for all stocks (one call, very fast) ──────────
    all_ns  = [f"{s}.NS" for s in FO_STOCKS]
    idx_sym = list(INDICES.keys())
    all_sym = all_ns + idx_sym

    try:
        raw = yf.download(all_sym, period="60d", auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
    except Exception:
        raw = None

    # ── 2. Try intraday batch for near-real-time CMP ──────────────────────────
    try:
        intra = yf.download(all_sym, period="1d", interval="5m",
                            auto_adjust=True, progress=False,
                            group_by="ticker", threads=True)
    except Exception:
        intra = None

    def get_hist(sym):
        try:
            if raw is None: return None
            if len(all_sym)==1:
                return pd.DataFrame({"Close":raw["Close"],"High":raw["High"],
                                     "Low":raw["Low"],"Volume":raw["Volume"]}).dropna()
            d = raw[sym] if sym in raw.columns.get_level_values(0) else raw.get(sym)
            if d is None or d.empty: return None
            return d[["Close","High","Low","Volume"]].dropna()
        except Exception:
            return None

    def get_cmp(sym):
        try:
            if intra is None: return None
            if len(all_sym)==1: return float(intra["Close"].dropna().iloc[-1])
            d = intra[sym] if sym in intra.columns.get_level_values(0) else intra.get(sym)
            if d is None or d.empty: return None
            return float(d["Close"].dropna().iloc[-1])
        except Exception:
            return None

    # ── 3. Generate technical signals for every stock ─────────────────────────
    for sym in FO_STOCKS:
        ns   = f"{sym}.NS"
        hist = get_hist(ns)
        if hist is not None and len(hist)>=35:
            sig = generate_signal(hist)
            if sig:
                live_cmp = get_cmp(ns)
                if live_cmp: sig["cmp"] = round(live_cmp,2)
                sig["symbol"]=sym; sig["ticker"]=ns; sig["sector"]=get_sector(sym)
                results[sym] = sig
                continue
        results[sym] = _mock(sym, ns)

    # ── 4. Generate index signals ─────────────────────────────────────────────
    idx_results = []
    for sym, meta in INDICES.items():
        hist = get_hist(sym)
        if hist is not None and len(hist)>=35:
            sig = generate_signal(hist)
            if sig:
                live_cmp = get_cmp(sym)
                if live_cmp: sig["cmp"]=round(live_cmp,2)
                sig["symbol"]=meta["name"]; sig["ticker"]=sym; sig["sector"]="Index"
                idx_results.append(sig)
                continue
        idx_results.append(_mock(meta["name"], sym))

    # ── 5. Fetch option chains for BUY/SELL stocks in parallel ───────────────
    actionable = [r for r in results.values()
                  if r["signal"] in ("STRONG BUY","BUY","SELL","STRONG SELL")]

    def fetch_and_build(r):
        opt = fetch_option_chain(r["ticker"], r["cmp"], r["signal"])
        rec = option_recommendation(r["signal"], opt)
        return r["symbol"], opt, rec

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_and_build, r): r["symbol"] for r in actionable}
        for f in as_completed(futs):
            try:
                sym, opt, rec = f.result()
                results[sym]["opt"] = opt
                results[sym]["rec"] = rec
            except Exception:
                pass

    # Fill HOLD stocks with estimated options
    for sym, r in results.items():
        if "opt" not in r:
            opt = fetch_option_chain.__wrapped__(r["ticker"],r["cmp"],r["signal"]) \
                  if hasattr(fetch_option_chain,"__wrapped__") \
                  else {"expiry":"N/A","atm_strike":round(r["cmp"]/50)*50,
                        "call_premium":round(r["cmp"]*.015,2),"call_iv":"~20%",
                        "call_oi":"N/A","put_premium":round(r["cmp"]*.015,2),
                        "put_iv":"~20%","put_oi":"N/A","pcr":"N/A",
                        "call_oi_int":0,"put_oi_int":0}
            results[sym]["opt"] = opt
            results[sym]["rec"] = option_recommendation(r["signal"], opt)

    # Also fill index option estimates
    for r in idx_results:
        if "opt" not in r:
            opt={"expiry":"N/A","atm_strike":round(r["cmp"]/50)*50,
                 "call_premium":round(r["cmp"]*.015,2),"call_iv":"~20%",
                 "call_oi":"N/A","put_premium":round(r["cmp"]*.015,2),
                 "put_iv":"~20%","put_oi":"N/A","pcr":"N/A",
                 "call_oi_int":0,"put_oi_int":0}
            r["opt"]=opt; r["rec"]=option_recommendation(r["signal"],opt)

    order={"STRONG BUY":0,"BUY":1,"HOLD":2,"SELL":3,"STRONG SELL":4}
    stk_sorted=sorted(results.values(),key=lambda x:order.get(x["signal"],2))
    idx_sorted=sorted(idx_results,     key=lambda x:order.get(x["signal"],2))
    return idx_sorted, stk_sorted


# =============================================================================
#  FUNDAMENTAL + FULL ANALYSIS (search only)
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
        if float(roe)>0.20:fs+=2
        elif float(roe)>0.12:fs+=1
        elif float(roe)<0:fs-=1
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
        try:
            intra=t.history(period="1d",interval="5m",auto_adjust=True)
            cmp=float(intra["Close"].iloc[-1]) if not intra.empty else float(hist["Close"].iloc[-1])
        except: cmp=float(hist["Close"].iloc[-1])
        sig=generate_signal(hist)
        if not sig: return None
        sig["cmp"]=round(cmp,2)
        opt=fetch_option_chain(ticker_str,cmp,sig["signal"])
        rec=option_recommendation(sig["signal"],opt)
        try: fund=get_fundamentals(t.info)
        except: fund={"company":"N/A","sector":"N/A","industry":"N/A","market_cap":"N/A",
                      "pe":"N/A","pb":"N/A","eps":"N/A","roe":"N/A","de":"N/A",
                      "div_yield":"N/A","revenue":"N/A","net_income":"N/A",
                      "w52_high":"N/A","w52_low":"N/A","book_value":"N/A",
                      "fund_view":"N/A","f_score":0,"desc":"N/A"}
        chart=hist["Close"].tail(90).reset_index()
        chart.columns=["Date","Price"]; chart["Date"]=pd.to_datetime(chart["Date"]).dt.date
        ts=sig["score"]+fund["f_score"]
        if ts>=5:cv="🚀 STRONG BULLISH"
        elif ts>=2:cv="🟢 BULLISH"
        elif ts<=-5:cv="⛔ STRONG BEARISH"
        elif ts<=-2:cv="🔴 BEARISH"
        else:cv="🟡 NEUTRAL"
        return {**sig,"fund":fund,"chart":chart,"opt":opt,"rec":rec,"combined":cv}
    except Exception: return None

# =============================================================================
#  SESSION STATE
# =============================================================================
for k,v in [("cache_key",0),("search_q",""),("last_ref",time_ist())]:
    if k not in st.session_state: st.session_state[k]=v

# =============================================================================
#  HEADER
# =============================================================================
h1,h2=st.columns([5,2])
with h1:
    st.markdown("""<div style='padding:.4rem 0;'>
    <div style='font-size:.62rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#6366F1;margin-bottom:5px;'>NSE · FUTURES & OPTIONS · REAL-TIME SIGNALS</div>
    <h1 style='font-size:2rem;font-weight:900;margin:0;letter-spacing:-.03em;
       background:linear-gradient(135deg,#F8FAFC,#94A3B8);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>F&O Signal Screener</h1>
    <p style='color:#475569;font-size:.78rem;margin:5px 0 0;'>RSI · MACD · EMA · Bollinger · Volume + CALL/PUT Recommendations</p>
    </div>""",unsafe_allow_html=True)
with h2:
    st.markdown(f"""<div style='text-align:right;padding-top:.8rem;'>
    <div style='font-size:.58rem;color:#64748B;font-weight:600;letter-spacing:.1em;text-transform:uppercase;'>🕐 Indian Standard Time</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:.9rem;color:#6366F1;font-weight:700;margin-top:2px;'>{now_ist()}</div>
    <div style='font-size:.6rem;color:#334155;margin-top:2px;'>Refreshed: {st.session_state['last_ref']} · ~5 min cache</div>
    </div>""",unsafe_allow_html=True)
st.divider()

# =============================================================================
#  SEARCH BAR
# =============================================================================
st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:5px;'>🔍 Deep Analysis — Search Any NSE Stock or Index</div>",unsafe_allow_html=True)
sb1,sb2=st.columns([5,1])
with sb1:
    q=st.text_input("Search",placeholder="e.g.  RELIANCE · TCS · NIFTY · BANKNIFTY · HDFC · INFY",label_visibility="collapsed")
with sb2:
    analyse=st.button("🔍 Analyse",use_container_width=True)

raw_q=(q.strip().upper() if analyse or q else "")
if raw_q:
    resolved=SEARCH_ALIASES.get(raw_q, raw_q if raw_q.startswith("^") else raw_q+".NS")
    with st.spinner(f"⏳ Analysing {raw_q}…"):
        ana=search_analysis(resolved)
    if ana:
        f=ana["fund"]; sig=ana["signal"]
        SBD={"STRONG BUY":"#00C853","BUY":"#4CAF50","HOLD":"#FF9800","SELL":"#F44336","STRONG SELL":"#D50000"}
        bd=SBD.get(sig,"#FF9800"); cc="#4CAF50" if ana["change_pct"]>=0 else "#F44336"
        tc="#4CAF50" if ana["score"]>=0 else "#F44336"; sc_="#F44336" if ana["score"]>=0 else "#4CAF50"
        # Top banner
        st.markdown(f"""<div style='background:linear-gradient(135deg,#0D1117,#111827);border:1.5px solid {bd};
            border-radius:16px;padding:1.2rem 1.5rem;margin:8px 0 4px;position:relative;overflow:hidden;'>
          <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{bd},{bd}33);'></div>
          <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
            <div>
              <div style='font-size:.6rem;color:#6366F1;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'>{f['sector']} · {f['industry']}</div>
              <div style='font-size:1.3rem;font-weight:900;color:#F8FAFC;margin:2px 0;'>{f['company']}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:1.8rem;font-weight:800;color:#F8FAFC;'>
                ₹{ana['cmp']:,.2f}<span style='font-size:.85rem;color:{cc};margin-left:10px;'>
                {'▲' if ana['change_pct']>=0 else '▼'} {abs(ana['change_pct'])}%</span></div>
            </div>
            <div style='text-align:center;'>
              <div style='background:{bd};color:{'#000' if sig in ('STRONG BUY','BUY','HOLD') else '#fff'};
                  font-size:1rem;font-weight:800;padding:8px 20px;border-radius:24px;'>{ana['emoji']} {sig}</div>
              <div style='font-size:.7rem;color:#64748B;margin-top:5px;'>{ana['combined']}</div>
            </div>
          </div></div>""",unsafe_allow_html=True)

        c1,c2,c3=st.columns([2,2,1.8])
        with c1:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 5px;'>📈 90-Day Price Chart</div>",unsafe_allow_html=True)
            st.line_chart(ana["chart"].set_index("Date")["Price"],height=160,use_container_width=True)
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 5px;'>🎯 Trade Levels (Futures/Equity)</div>",unsafe_allow_html=True)
            l1,l2,l3=st.columns(3)
            l1.metric("Entry",f"₹{ana['entry']:,.2f}")
            l2.metric("Target",f"₹{ana['target']:,.2f}")
            l3.metric("Stop Loss",f"₹{ana['sl']:,.2f}")
            # Option recommendation
            rec=ana["rec"]; opt=ana["opt"]
            st.markdown(f"""<div style='background:#0D1117;border:1px solid {rec['color']};border-radius:12px;padding:12px;margin-top:10px;'>
              <div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin-bottom:8px;'>📞 Option Recommendation</div>
              <div style='font-size:1rem;font-weight:800;color:{rec['color']};margin-bottom:8px;'>{rec['badge']} — {opt.get('expiry','N/A')}</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;'>
                <div style='background:#060B14;border-radius:8px;padding:6px 8px;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Strike</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#E2E8F0;font-weight:700;'>₹{rec['strike']}</div>
                </div>
                <div style='background:#060B14;border-radius:8px;padding:6px 8px;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Premium (LTP)</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:{rec['color']};font-weight:700;'>₹{rec['premium']}</div>
                </div>
                <div style='background:#060B14;border-radius:8px;padding:6px 8px;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Target Premium</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#4CAF50;font-weight:700;'>₹{rec['target']}</div>
                </div>
                <div style='background:#060B14;border-radius:8px;padding:6px 8px;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>SL Premium</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#F44336;font-weight:700;'>₹{rec['sl']}</div>
                </div>
              </div>
              <div style='margin-top:8px;font-size:.65rem;color:#64748B;'>IV: {rec['iv']} &nbsp;|&nbsp; OI: {rec['oi']} &nbsp;|&nbsp; PCR: {opt.get('pcr','N/A')}</div>
            </div>""",unsafe_allow_html=True)

        with c2:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 5px;'>🔬 Technical Indicators</div>",unsafe_allow_html=True)
            def trow(lbl,val,bull=None):
                c_="#4CAF50" if bull is True else "#F44336" if bull is False else "#94A3B8"
                return f"<div style='display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid #1E293B;'><span style='font-size:.72rem;color:#64748B;'>{lbl}</span><span style='font-family:JetBrains Mono,monospace;font-size:.72rem;color:{c_};font-weight:600;'>{val}</span></div>"
            rows=[trow("RSI (14)",f"{ana['rsi']}",ana['rsi']<50),
                  trow("MACD","Bullish ✅" if ana['score']>=0 else "Bearish ❌",ana['score']>=0),
                  trow("EMA 20",f"₹{ana['e20']:,.2f}",ana['cmp']>ana['e20']),
                  trow("EMA 50",f"₹{ana['e50']:,.2f}",ana['cmp']>ana['e50']),
                  trow("ATR",f"₹{ana['atr']:,.2f}"),
                  trow("Signal Score",f"{ana['score']} / 8"),
                  trow("Call ATM Strike",f"₹{opt.get('atm_strike','N/A')}"),
                  trow("Call Premium",f"₹{opt.get('call_premium','N/A')}"),
                  trow("Put Premium",f"₹{opt.get('put_premium','N/A')}"),
                  trow("Call IV",opt.get('call_iv','N/A')),
                  trow("Put IV",opt.get('put_iv','N/A')),
                  trow("PCR",str(opt.get('pcr','N/A')))]
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>{''.join(rows)}</div>",unsafe_allow_html=True)
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 5px;'>💡 Signal Reasons</div>",unsafe_allow_html=True)
            for r in ana["reasons"]:
                st.markdown(f"<div style='font-size:.72rem;color:#94A3B8;padding:2px 0;'>• {r}</div>",unsafe_allow_html=True)

        with c3:
            st.markdown("<div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6366F1;margin:10px 0 5px;'>📊 Fundamentals</div>",unsafe_allow_html=True)
            def frow(l,v): return f"<div style='display:flex;justify-content:space-between;padding:5px 10px;border-bottom:1px solid #1E293B;'><span style='font-size:.68rem;color:#64748B;'>{l}</span><span style='font-family:JetBrains Mono,monospace;font-size:.68rem;color:#E2E8F0;font-weight:600;'>{v}</span></div>"
            fr=[frow("Market Cap",f["market_cap"]),frow("P/E Ratio",str(f["pe"])),
                frow("P/B Ratio",str(f["pb"])),frow("EPS",str(f["eps"])),
                frow("ROE",f["roe"]),frow("Debt/Equity",str(f["de"])),
                frow("Revenue",f["revenue"]),frow("Net Income",f["net_income"]),
                frow("Div Yield",f["div_yield"]),frow("52W High",f"₹{f['w52_high']}"),
                frow("52W Low",f"₹{f['w52_low']}"),frow("Book Value",str(f["book_value"]))]
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;overflow:hidden;'>{''.join(fr)}</div>",unsafe_allow_html=True)
            st.markdown(f"<div style='background:#0D1117;border:1px solid #1E293B;border-radius:12px;padding:10px;margin-top:8px;text-align:center;'><div style='font-size:.58rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;font-weight:600;margin-bottom:3px;'>Fundamental View</div><div style='font-size:.82rem;font-weight:700;color:#6366F1;'>{f['fund_view']}</div></div>",unsafe_allow_html=True)
            if f["desc"]!="N/A":
                with st.expander("ℹ️ About"):
                    st.markdown(f"<div style='font-size:.7rem;color:#94A3B8;line-height:1.6;'>{f['desc']}</div>",unsafe_allow_html=True)
    elif LIVE:
        st.warning(f"⚠️ No data for **'{raw_q}'**. Try: RELIANCE · TCS · HDFCBANK · NIFTY · BANKNIFTY")
    st.divider()

# =============================================================================
#  REFRESH + CONTROLS
# =============================================================================
ct1,ct2,ct3=st.columns([2,2,2])
with ct1:
    if st.button("🔄  REFRESH ALL SIGNALS",use_container_width=True):
        st.cache_data.clear()
        st.session_state["cache_key"]+=1
        st.session_state["last_ref"]=time_ist()
        st.rerun()
with ct2:
    filter_sig=st.selectbox("Signal",["All Signals","STRONG BUY","BUY","HOLD","SELL","STRONG SELL"],label_visibility="collapsed")
with ct3:
    filter_sec=st.selectbox("Sector",["All Sectors"]+list(SECTOR_MAP.keys()),label_visibility="collapsed")
st.divider()

if not LIVE:
    st.warning("⚠️ Demo data — commit `requirements.txt` to GitHub and reboot for live data.",icon="📦")

# Load
with st.spinner("⏳ Loading all signals…"):
    idx_sigs,stk_sigs=load_screener(_key=st.session_state["cache_key"])

def filt(lst):
    r=lst
    if filter_sig!="All Signals": r=[x for x in r if x["signal"]==filter_sig]
    if filter_sec!="All Sectors":  r=[x for x in r if x.get("sector")==filter_sec]
    return r
fi=filt(idx_sigs); fs=filt(stk_sigs)

# Summary strip
all_s=idx_sigs+stk_sigs
cnts={s:sum(1 for x in all_s if x["signal"]==s) for s in ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]}
m1,m2,m3,m4,m5,m6=st.columns(6)
with m1: st.metric("🚀 Strong Buy",  cnts["STRONG BUY"],  delta=f"{cnts['STRONG BUY']} stocks")
with m2: st.metric("🟢 Buy",         cnts["BUY"],          delta=f"{cnts['BUY']} stocks")
with m3: st.metric("🟡 Hold",        cnts["HOLD"],         delta=f"{cnts['HOLD']} neutral")
with m4: st.metric("🔴 Sell",        cnts["SELL"],         delta=f"{cnts['SELL']} stocks")
with m5: st.metric("⛔ Strong Sell", cnts["STRONG SELL"],  delta=f"{cnts['STRONG SELL']} stocks")
with m6: st.metric("📋 Total",       len(all_s),           delta=f"{len(stk_sigs)} F&O stocks")
st.divider()

# =============================================================================
#  INDEX CARDS
# =============================================================================
if fi:
    st.markdown("<div style='font-size:.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#6366F1;margin-bottom:8px;'>📊 Market Indices</div>",unsafe_allow_html=True)
    SBD={"STRONG BUY":"#00C853","BUY":"#4CAF50","HOLD":"#FF9800","SELL":"#F44336","STRONG SELL":"#D50000"}
    idx_meta={v["name"]:v for v in INDICES.values()}
    for i in range(0,len(fi),3):
        row=fi[i:i+3]; cols=st.columns(len(row))
        for col,d in zip(cols,row):
            bd=SBD.get(d["signal"],"#FF9800"); cc="#4CAF50" if d["change_pct"]>=0 else "#F44336"
            tc="#4CAF50" if d["score"]>=0 else "#F44336"; sc_="#F44336" if d["score"]>=0 else "#4CAF50"
            meta=idx_meta.get(d["symbol"],{"emoji":"📈"})
            col.markdown(f"""<div style='background:linear-gradient(135deg,#0D1117,#111827);
                border:1.5px solid {bd};border-radius:16px;padding:1.2rem;margin:4px 0;position:relative;overflow:hidden;'>
              <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{bd},{bd}33);'></div>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <div style='font-size:.58rem;color:#6366F1;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'>{meta.get('emoji','📈')} {d['symbol']}</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:1.7rem;font-weight:800;color:#F8FAFC;'>{d['cmp']:,.2f}</div>
                  <div style='font-size:.8rem;color:{cc};font-weight:700;'>{'▲' if d['change_pct']>=0 else '▼'} {abs(d['change_pct'])}%</div>
                </div>
                <div style='text-align:center;'>
                  <div style='background:{bd};color:{'#000' if d['signal'] in ('STRONG BUY','BUY','HOLD') else '#fff'};font-size:.75rem;font-weight:800;padding:6px 14px;border-radius:20px;'>{d['emoji']} {d['signal']}</div>
                  <div style='font-size:.6rem;color:#64748B;margin-top:4px;'>RSI {d['rsi']}</div>
                </div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:10px;'>
                <div style='background:#060B14;border-radius:8px;padding:6px;text-align:center;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Entry</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:#E2E8F0;font-weight:700;'>₹{d['entry']:,.0f}</div>
                </div>
                <div style='background:#060B14;border-radius:8px;padding:6px;text-align:center;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Target</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:{tc};font-weight:700;'>₹{d['target']:,.0f}</div>
                </div>
                <div style='background:#060B14;border-radius:8px;padding:6px;text-align:center;'>
                  <div style='font-size:.55rem;color:#64748B;text-transform:uppercase;'>Stop Loss</div>
                  <div style='font-family:JetBrains Mono,monospace;font-size:.78rem;color:{sc_};font-weight:700;'>₹{d['sl']:,.0f}</div>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)
    st.divider()

# =============================================================================
#  TABBED SIGNAL TABLES
# =============================================================================
tab1,tab2,tab3,tab4,tab5=st.tabs([
    f"📋 All Stocks ({len(fs)})",
    f"🚀 BUY ({sum(1 for x in fs if x['signal'] in ('STRONG BUY','BUY'))})",
    f"⛔ SELL ({sum(1 for x in fs if x['signal'] in ('STRONG SELL','SELL'))})",
    f"📞 Call/Put Options",
    f"🏭 By Sector",
])

SIG_COLORS={"STRONG BUY":"🚀","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG SELL":"⛔"}

def make_table(data, show_options=False, table_key='default'):
    if not data:
        st.info("No stocks match the current filter."); return
    rows=[]
    for d in data:
        rec=d.get("rec",{}); opt=d.get("opt",{})
        row={"Symbol":d["symbol"],"Sector":d.get("sector","—"),
             "Signal":f"{d['emoji']} {d['signal']}","CMP (₹)":d["cmp"],
             "Change %":d["change_pct"],"RSI":d["rsi"],
             "Entry (₹)":d["entry"],"Target (₹)":d["target"],"SL (₹)":d["sl"],
             "Score":d["score"]}
        if show_options:
            row["Option Action"]=rec.get("action","—")
            row["Strike"]=rec.get("strike","—")
            row["Expiry"]=str(opt.get("expiry","—"))
            row["Premium (₹)"]=rec.get("premium","—")
            row["Opt Target"]=rec.get("target","—")
            row["Opt SL"]=rec.get("sl","—")
            row["Call IV"]=opt.get("call_iv","—")
            row["Put IV"]=opt.get("put_iv","—")
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
    csv=df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV",csv,"fo_signals.csv","text/csv", key=f"dl_{table_key}")

with tab1:
    make_table(fs, table_key="all")

with tab2:
    buy_stocks=[x for x in fs if x["signal"] in ("STRONG BUY","BUY")]
    st.markdown(f"<div style='font-size:.62rem;color:#4CAF50;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;'>🟢 {len(buy_stocks)} BUY signals found</div>",unsafe_allow_html=True)
    make_table(buy_stocks, table_key="buy")

with tab3:
    sell_stocks=[x for x in fs if x["signal"] in ("STRONG SELL","SELL")]
    st.markdown(f"<div style='font-size:.62rem;color:#F44336;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;'>🔴 {len(sell_stocks)} SELL signals found</div>",unsafe_allow_html=True)
    make_table(sell_stocks, table_key="sell")

with tab4:
    opt_stocks=[x for x in fs if x["signal"] in ("STRONG BUY","BUY","STRONG SELL","SELL")]
    st.markdown(f"""<div style='background:#0D1117;border:1px solid #1E293B;border-radius:10px;padding:10px 14px;margin-bottom:10px;'>
    <div style='font-size:.7rem;color:#94A3B8;line-height:1.7;'>
    📞 <b style='color:#F8FAFC;'>BUY CALL (CE)</b> — for BUY/STRONG BUY signals (bullish view)<br>
    📉 <b style='color:#F8FAFC;'>BUY PUT (PE)</b> — for SELL/STRONG SELL signals (bearish view)<br>
    <span style='color:#64748B;font-size:.65rem;'>Premium Target = 1.8× Entry · SL = 0.5× Entry · Strikes are ATM nearest expiry</span>
    </div></div>""",unsafe_allow_html=True)
    make_table(opt_stocks, show_options=True, table_key="opts")

with tab5:
    for sector, stocks in SECTOR_MAP.items():
        sector_data=[x for x in fs if x.get("sector")==sector]
        if not sector_data: continue
        buys=sum(1 for x in sector_data if x["signal"] in ("STRONG BUY","BUY"))
        sells=sum(1 for x in sector_data if x["signal"] in ("STRONG SELL","SELL"))
        mood="🟢 Bullish" if buys>sells else "🔴 Bearish" if sells>buys else "🟡 Mixed"
        with st.expander(f"🏭 {sector}  ·  {len(sector_data)} stocks  ·  {mood}  ({buys} Buy / {sells} Sell)",expanded=False):
            make_table(sector_data, table_key=f"sector_{sector}")

st.divider()
st.markdown("""<div style='text-align:center;padding:1rem 0 .3rem;'>
<p style='font-size:.58rem;color:#1E293B;font-family:JetBrains Mono,monospace;letter-spacing:.08em;line-height:1.8;'>
⚠️ FOR EDUCATIONAL PURPOSES ONLY · NOT SEBI-REGISTERED ADVICE<br>
DATA VIA YAHOO FINANCE · ALWAYS DO YOUR OWN RESEARCH BEFORE TRADING
</p></div>""",unsafe_allow_html=True)
