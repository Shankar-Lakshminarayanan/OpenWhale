from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import io
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================================
# PAGE CONFIGURATION (OPENWHALE BRANDING)
# ==========================================
st.set_page_config(
    page_title="OpenWhale | Free Smart-Money & S&P Movers Terminal",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# SESSION STATE FOR CROSS-PAGE TICKER LINKING
# ==========================================
PAGES = [
    "🏆 1. S&P Movers & Shakers (Top 50 Each Way)",
    "🔥 2. Top 20 Options Whale Scanner",
    "⚡ 3. Master Actionable Watchlist (Page 1+2 Merged)",
    "🔍 4. Single-Ticker Deep Dive (+ Trade Ticket)",
    "📖 5. Help & Trading Playbook",
]

if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = "SPY"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]


def jump_to_deep_dive(ticker_sym):
    """Callback that links any selected ticker directly to Page 4 Deep Dive."""
    st.session_state.active_ticker = ticker_sym.upper().strip()
    st.session_state.nav_page = PAGES[3]


# ==========================================
# HOVER GLOSSARY & PLAIN-ENGLISH DEFINITIONS
# ==========================================
GLOSSARY_TOOLTIPS = {
    "Ticker": "The short stock market code for the company or index (e.g., AAPL = Apple, SPY = S&P 500). Click any row to open its Deep-Dive Trade Ticket!",
    "Confluence_Tier": "Shows whether this stock was triggered by BOTH Page 1 (Price/Volume Mover) AND Page 2 (Options Whales) — '⭐⭐ DOUBLE CONFIRMED' is the highest-conviction signal.",
    "Verdict": "Automated Trade Ticket Verdict: 🟢 BUY LONG or 🔴 SELL SHORT when institutional factors align with >=1.2x Reward-to-Risk. Otherwise 🟡 WAIT.",
    "Direction": "Shows whether momentum and smart-money buying are pushing the stock UP (Bullish Long) or DOWN (Bearish Short).",
    "Last_Price": "The most recent price a share traded at, INCLUDING early 4:00 AM Pre-Market and 8:00 PM After-Hours trading.",
    "Total_Move_Pct": "How much the stock price has gained or lost compared to yesterday's 4:00 PM closing price (includes overnight action).",
    "Ext_Gap_Pct": "How much the stock has jumped or dropped in Pre-Market / After-Hours trading alone compared to the regular close.",
    "VWAP": "Volume-Weighted Average Price: The true 'Fair Price' paid by the average trader today. Above VWAP = Buyers are winning; Below VWAP = Sellers are in control.",
    "VWAP_Dist_Pct": "How far the current price is stretched above or below today's Fair Price (VWAP). If stretched too far, it often snaps back.",
    "RSI_14": "Relative Strength Index (0 to 100 'Fever Score'): Above 70 means the stock is Overheated (too many buyers, risk of pullback). Below 30 means it is Oversold / On Sale (due for a bounce).",
    "RVOL": "Relative Volume Surge: Compares today's trading volume to a normal day. 2.0x means twice as many shares are trading as usual—a sign big institutions are active.",
    "Trend_Signal": "Combines short-term price momentum (9 vs 21 EMA) with the Fair Price (VWAP) to tell you if the trend is Strong, Weak, or Choppy.",
    "Est_Floor": "Safety Net Floor (Support): The estimated bottom of today's normal price range where dip-buyers typically step in.",
    "Est_Ceiling": "Profit Target Ceiling (Resistance): The estimated top of today's normal price range where traders often take profits.",
    "Heat_Score": "OpenWhale Action Score (0 to 100): Blends price move, volume surge, Fair Price trend, and options buying into one simple strength rating.",
    "Call_Flow_Pct": "Percentage of options money spent on CALLS (bets that the stock will go UP) vs. PUTS (bets that the stock will go DOWN). Above 60% = strong bullish betting.",
    "Whale_Sweeps": "Counts how many urgent, large institutional options bets (where today's volume is >1.5x existing contracts and >$50,000) hit the tape today.",
    "Put_Wall": "Price Floor / Put Wall (-GEX): The exact strike price with the heaviest Put hedging. Big banks mechanically buy shares here, acting like a trampoline floor.",
    "Call_Wall": "Price Ceiling / Call Wall (+GEX): The exact strike price with the heaviest Call hedging. Big banks mechanically sell shares here, acting like a brick ceiling.",
    "Zero_Gamma": "Danger Line (Zero Gamma Flip): Above this price, market makers cushion moves and keep trading calm. Below this price, they sell into drops and amplify volatility.",
    "Entry_Zone": "Suggested price range to enter the trade based on current price and Today's Fair Value Price (VWAP).",
    "Stop_Loss": "Exact price where the setup breaks down and is invalidated. Exit immediately if hit to protect your capital.",
    "Profit_Target": "Suggested price to take profits, placed just inside the major Price Ceiling (Call Wall) or Price Floor (Put Wall).",
    "RR_Ratio": "Reward-to-Risk Ratio: Shows how many dollars you stand to gain for every $1 risked. Aim for 1.5 : 1 or higher.",
}

st.markdown(
    """
    <style>
    .ow-glossary-bar {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }
    .ow-pill {
        position: relative;
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
        cursor: help;
        border-bottom: 1px dashed #38bdf8;
    }
    .ow-pill .ow-tip {
        visibility: hidden;
        width: 290px;
        background-color: #0f172a;
        color: #f8fafc;
        text-align: left;
        border-radius: 8px;
        padding: 10px 12px;
        position: absolute;
        z-index: 9999;
        top: 130%;
        left: 50%;
        margin-left: -145px;
        opacity: 0;
        transition: opacity 0.2s;
        border: 1px solid #38bdf8;
        box-shadow: 0 10px 25px rgba(0,0,0,0.75);
        font-weight: 400;
        font-size: 0.80rem;
        line-height: 1.4;
    }
    .ow-pill:hover .ow-tip {
        visibility: visible;
        opacity: 1;
    }
    .ticket-box {
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        border: 2px solid;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hover_glossary_banner():
    pills_html = """
    <div class="ow-glossary-bar">
        <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">💡 Hover any term for Plain-English meaning:</span>
        <div class="ow-pill">⭐⭐ Double Confirmed<span class="ow-tip"><b>Double Confirmed Setup:</b> A stock that simultaneously appears in Page 1 (Extended-Hours Price/Volume Mover) AND Page 2 (Heavy Options Whale Buying).</span></div>
        <div class="ow-pill">Price Floor (Put Wall)<span class="ow-tip"><b>Price Floor / Put Wall:</b> The price level where big banks mechanically step in and buy shares to hedge options. Acts like a trampoline safety net.</span></div>
        <div class="ow-pill">Price Ceiling (Call Wall)<span class="ow-tip"><b>Price Ceiling / Call Wall:</b> The price level where big banks start selling shares to hedge options. Acts like a brick ceiling—great place to take profits.</span></div>
        <div class="ow-pill">Danger Line (Zero Gamma)<span class="ow-tip"><b>Danger Line (Flip Level):</b> Above this price, trading is calm and dips get bought. Below this price, swings become fast and violent.</span></div>
        <div class="ow-pill">Fair Price (VWAP)<span class="ow-tip"><b>Fair Value Price (VWAP):</b> The average price paid per share today weighted by volume (including Pre-Market & After-Hours). Above = Buyers winning.</span></div>
        <div class="ow-pill">Fever Score (RSI)<span class="ow-tip"><b>Fever Score (RSI 0–100):</b> Measures if a stock is running too hot. Above 70 = Overheated (due for a breather). Below 30 = On Sale / Oversold (due for a bounce).</span></div>
        <div class="ow-pill">Whale Sweeps (Vol > OI)<span class="ow-tip"><b>Whale Sweeps:</b> Urgent options orders where today's volume exceeds total existing contracts (Open Interest), proving brand-new smart-money bets are opening.</span></div>
    </div>
    """
    st.markdown(pills_html, unsafe_allow_html=True)


def render_quick_jump_bar(available_tickers, key_prefix):
    """Renders a 1-click ticker linker bar on every scanner page."""
    if not available_tickers:
        return
    c1, c2 = st.columns([0.72, 0.28])
    with c1:
        default_idx = (
            available_tickers.index(st.session_state.active_ticker)
            if st.session_state.active_ticker in available_tickers
            else 0
        )
        chosen = st.selectbox(
            "🔗 Quick-Link Any Screened Ticker to Page 4 (Deep Dive & Trade Ticket):",
            options=available_tickers,
            index=default_idx,
            key=f"{key_prefix}_select",
        )
    with c2:
        st.write("")
        st.button(
            f"🔍 Open {chosen} Deep-Dive & Trade Ticket ➔",
            key=f"{key_prefix}_btn",
            type="primary",
            use_container_width=True,
            on_click=jump_to_deep_dive,
            args=(chosen,),
        )


# ==========================================
# S&P 500 + MAJOR INDICES LIQUID UNIVERSE (~125 TICKERS)
# ==========================================
SP500_LIQUID_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "ARKK", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM", "ADBE", "CSCO",
    "INTC", "QCOM", "TXN", "AMAT", "MU", "LRCX", "ADI", "KLAC", "PANW", "SNPS", "CDNS", "CRWD", "PLTR", "NOW",
    "IBM", "INTU", "UBER", "ABNB", "SHOP", "SNOW", "ARM", "SMCI", "DELL", "HPQ", "MRVL", "ON", "MCHP", "FTNT",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP", "BLK", "C", "SCHW", "CB", "PGR", "PYPL",
    "COIN", "HOOD", "SOFI", "AFRM", "SQ", "COF", "USB", "PNC", "TFC", "CME", "ICE", "MCO",
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN", "ISRG", "ELV", "VRTX", "REGN", "SYK",
    "BSX", "BMY", "GILD", "ZTS", "CVS", "CI", "HUM", "MRNA", "BIIB", "GEHC",
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "MAR", "CMG",
    "YUM", "LULU", "ORLY", "AZO", "ROST", "CL", "KMB", "MDLZ", "MO", "PM", "F", "GM",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL", "CAT", "GE", "RTX", "HON", "UNP", "BA",
    "DE", "LMT", "UPS", "FDX", "ADP", "ETN", "ITW", "GD", "NOC", "WM", "LIN", "SHW", "FCX", "NEM", "NEE", "SO", "DUK"
]
SP500_LIQUID_UNIVERSE = list(dict.fromkeys(SP500_LIQUID_UNIVERSE))
OCC_REGEX = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")


# ==========================================
# 1. TECHNICAL INDICATOR HELPERS
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


# ==========================================
# 2. BATCH S&P MOVERS & SHAKERS ENGINE (PAGE 1)
# ==========================================
@st.cache_data(ttl=120)
def fetch_sp_movers_and_shakers(tickers):
    intraday_raw = yf.download(
        tickers, period="5d", interval="15m", prepost=True, group_by="ticker", threads=True, progress=False
    )
    daily_raw = yf.download(
        tickers, period="10d", interval="1d", prepost=False, group_by="ticker", threads=True, progress=False
    )

    rows = []
    for sym in tickers:
        try:
            df_intra = intraday_raw[sym].dropna(subset=["Close"]) if sym in intraday_raw else pd.DataFrame()
            df_daily = daily_raw[sym].dropna(subset=["Close"]) if sym in daily_raw else pd.DataFrame()
            if df_intra.empty or len(df_daily) < 2:
                continue

            last_price = float(df_intra["Close"].iloc[-1])
            prev_close = float(df_daily["Close"].iloc[-2])
            reg_close = float(df_daily["Close"].iloc[-1])

            total_move_pct = ((last_price - prev_close) / prev_close) * 100.0
            ext_gap_pct = ((last_price - reg_close) / reg_close) * 100.0

            rsi_series = compute_rsi(df_intra["Close"], 14)
            rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
            ema9 = float(df_intra["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(df_intra["Close"].ewm(span=21, adjust=False).mean().iloc[-1])

            latest_day = df_intra.index[-1].date()
            today_slice = df_intra[df_intra.index.date == latest_day]
            if not today_slice.empty and today_slice["Volume"].sum() > 0:
                typ = (today_slice["High"] + today_slice["Low"] + today_slice["Close"]) / 3.0
                vwap = float((typ * today_slice["Volume"]).sum() / today_slice["Volume"].sum())
            else:
                vwap = last_price

            vwap_dist_pct = ((last_price - vwap) / vwap) * 100.0 if vwap > 0 else 0.0

            avg_vol = float(df_daily["Volume"].iloc[:-1].mean())
            today_vol = float(df_daily["Volume"].iloc[-1])
            rvol = round(today_vol / avg_vol, 2) if avg_vol > 0 else 1.0

            daily_range_pct = float(((df_daily["High"] - df_daily["Low"]) / df_daily["Close"]).mean())
            daily_range_pct = max(daily_range_pct, 0.012)
            est_floor = round(last_price * (1.0 - daily_range_pct * 1.25), 2)
            est_ceiling = round(last_price * (1.0 + daily_range_pct * 1.25), 2)

            ema_boost = 12.0 if ema9 >= ema21 else -12.0
            vwap_boost = np.clip(vwap_dist_pct * 8.0, -20.0, 20.0)
            move_boost = np.clip(total_move_pct * 10.0, -50.0, 50.0)
            rsi_boost = (rsi_val - 50.0) * 0.4
            raw_score = move_boost + vwap_boost + ema_boost + rsi_boost
            heat_score = round(min(99.0, max(10.0, 50.0 + abs(raw_score) * 0.65)), 1)

            direction_tag = "🟢 Top Bullish Mover" if raw_score >= 0 else "🔴 Top Bearish Mover"
            trend_status = (
                "🚀 Strong Uptrend"
                if (ema9 > ema21 and last_price >= vwap)
                else ("⚠️ Pullback / Weak" if (ema9 < ema21 and last_price < vwap) else "⚖️ Choppy / Mixed")
            )

            rows.append(
                {
                    "Ticker": sym,
                    "Direction": direction_tag,
                    "Last_Price": round(last_price, 2),
                    "Total_Move_Pct": round(total_move_pct, 2),
                    "Ext_Gap_Pct": round(ext_gap_pct, 2),
                    "VWAP": round(vwap, 2),
                    "VWAP_Dist_Pct": round(vwap_dist_pct, 2),
                    "RSI_14": round(rsi_val, 1),
                    "RVOL": rvol,
                    "Trend_Signal": trend_status,
                    "Est_Floor": est_floor,
                    "Est_Ceiling": est_ceiling,
                    "Heat_Score": heat_score,
                    "ema_bullish": bool(ema9 >= ema21),
                    "above_vwap": bool(last_price >= vwap),
                    "_raw_score": raw_score,
                    "_abs_move": max(abs(total_move_pct), 0.15),
                }
            )
        except Exception:
            continue

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_sorted = df_all.sort_values("Total_Move_Pct", ascending=False).reset_index(drop=True)
    top_50_bulls = df_sorted.head(50).copy()
    top_50_bulls["Direction"] = "🟢 Top 50 Bullish (Long)"

    top_50_bears = df_sorted.tail(50).sort_values("Total_Move_Pct", ascending=True).copy()
    top_50_bears["Direction"] = "🔴 Top 50 Bearish (Short)"

    combined_100 = pd.concat([top_50_bulls, top_50_bears], ignore_index=True)
    return top_50_bulls, top_50_bears, combined_100


# ==========================================
# 3. CBOE OPTIONS & WHALE ENGINE (LINKED TO PAGE 1)
# ==========================================
def calc_bs_gamma(S, K, T, r, sigma):
    T = np.maximum(T, 4 / (365 * 24))
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def fetch_cboe_chain(ticker_symbol, num_expiries=4, risk_free_rate=0.042):
    clean_sym = ticker_symbol.upper().strip().replace("^", "")
    index_tickers = {"SPX", "VIX", "NDX", "RUT", "DJX", "XSP"}
    cboe_sym = f"_{clean_sym}" if clean_sym in index_tickers else clean_sym

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_sym}.json"
    resp = requests.get(url, headers=headers, timeout=12)

    if resp.status_code != 200 and not cboe_sym.startswith("_"):
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{clean_sym}.json"
        resp = requests.get(url, headers=headers, timeout=12)

    resp.raise_for_status()
    payload = resp.json().get("data", {})

    spot = float(payload.get("current_price") or payload.get("close") or payload.get("bid") or 0)
    raw_options = payload.get("options", [])
    if not raw_options or spot <= 0:
        raise ValueError(f"Empty options list for {ticker_symbol}")

    today_str = dt.date.today().strftime("%Y-%m-%d")
    parsed_rows = []

    for opt in raw_options:
        sym = opt.get("option", "")
        m = OCC_REGEX.match(sym)
        if not m:
            continue
        _, ymd, cp, strike_raw = m.groups()
        exp_str = f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}"
        if exp_str < today_str:
            continue

        strike = int(strike_raw) / 1000.0
        if strike < spot * 0.80 or strike > spot * 1.20:
            continue

        last_p = float(opt.get("last_trade_price") or 0.0)
        bid = float(opt.get("bid") or 0.0)
        ask = float(opt.get("ask") or 0.0)
        mark = last_p if last_p > 0 else ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)

        parsed_rows.append(
            {
                "contractSymbol": sym,
                "type": "call" if cp == "C" else "put",
                "expiration": exp_str,
                "strike": strike,
                "lastPrice": round(mark, 2),
                "volume": float(opt.get("volume") or 0.0),
                "openInterest": float(opt.get("open_interest") or 0.0),
                "impliedVolatility": float(opt.get("iv") or 0.0),
                "cboe_gamma": float(opt.get("gamma") or 0.0),
            }
        )

    df = pd.DataFrame(parsed_rows)
    if df.empty:
        raise ValueError("No valid contracts in range.")

    valid_expiries = sorted(df["expiration"].unique())[:num_expiries]
    df = df[df["expiration"].isin(valid_expiries)].copy()

    now = dt.datetime.now()
    exp_dates = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = np.maximum((exp_dates - now).dt.total_seconds() / (365.0 * 24 * 3600), 1 / (365 * 24))

    bs_gamma = calc_bs_gamma(
        spot, df["strike"].values, df["T"].values, risk_free_rate, df["impliedVolatility"].values
    )
    df["gamma"] = np.where(df["cboe_gamma"] > 0, df["cboe_gamma"], bs_gamma)

    sign = np.where(df["type"] == "call", 1.0, -1.0)
    df["gex_dollars"] = df["gamma"] * df["openInterest"] * 100 * (spot**2) * 0.01 * sign
    df["est_premium"] = df["volume"] * df["lastPrice"] * 100
    df["vol_oi_ratio"] = np.where(df["openInterest"] > 0, df["volume"] / df["openInterest"], 0)

    return spot, df


def fetch_extended_hours_single(ticker):
    tk = yf.Ticker(ticker)
    df = tk.history(period="5d", interval="5m", prepost=True)
    daily = tk.history(period="5d", interval="1d", prepost=False)
    if df.empty or daily.empty:
        raise ValueError(f"No intraday data for {ticker}")

    latest_price = float(df["Close"].iloc[-1])
    prev_reg_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(daily["Close"].iloc[-1])
    ext_gap_pct = ((latest_price - prev_reg_close) / prev_reg_close) * 100.0

    df["RSI"] = compute_rsi(df["Close"], 14)
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    latest_day = df.index[-1].date()
    today_df = df[df.index.date == latest_day].copy()
    if not today_df.empty and today_df["Volume"].sum() > 0:
        typical_price = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3.0
        vwap = float((typical_price * today_df["Volume"]).sum() / today_df["Volume"].sum())
    else:
        vwap = latest_price

    current_rsi = float(df["RSI"].iloc[-1]) if not np.isnan(df["RSI"].iloc[-1]) else 50.0
    ema_bullish = bool(df["EMA9"].iloc[-1] > df["EMA21"].iloc[-1])
    above_vwap = bool(latest_price >= vwap)

    return {
        "latest_price": round(latest_price, 2),
        "ext_gap_pct": round(ext_gap_pct, 2),
        "vwap": round(vwap, 2),
        "above_vwap": above_vwap,
        "rsi": round(current_rsi, 1),
        "ema_bullish": ema_bullish,
    }


def compute_zero_gamma(gex_df_by_strike, spot):
    near_spot = gex_df_by_strike[
        (gex_df_by_strike["strike"] >= spot * 0.93) & (gex_df_by_strike["strike"] <= spot * 1.07)
    ].sort_values("strike")
    zero_gamma = spot
    signs = np.sign(near_spot["net_gex"].values)
    strikes = near_spot["strike"].values
    net_vals = near_spot["net_gex"].values
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and (net_vals[i + 1] - net_vals[i]) != 0:
            zero_gamma = strikes[i] - net_vals[i] * (strikes[i + 1] - strikes[i]) / (
                net_vals[i + 1] - net_vals[i]
            )
            break
    return round(float(zero_gamma), 2)


def scan_single_whale_ticker(ticker, precomputed_tech=None):
    """
    Reuses Page 1's precomputed price/VWAP/RSI data when available so Page 1, Page 2,
    and Page 3 are 100% linked and synchronized!
    """
    try:
        spot, opt_df = fetch_cboe_chain(ticker, num_expiries=3)
        if precomputed_tech is not None:
            tech = precomputed_tech
            spot = tech["latest_price"]
        else:
            try:
                tech = fetch_extended_hours_single(ticker)
                spot = tech["latest_price"]
            except Exception:
                tech = {
                    "latest_price": round(spot, 2),
                    "ext_gap_pct": 0.0,
                    "vwap": round(spot, 2),
                    "above_vwap": True,
                    "rsi": 50.0,
                    "ema_bullish": True,
                }

        call_prem = opt_df[opt_df["type"] == "call"]["est_premium"].sum()
        put_prem = opt_df[opt_df["type"] == "put"]["est_premium"].sum()
        total_prem = call_prem + put_prem
        call_pct = (call_prem / total_prem * 100.0) if total_prem > 0 else 50.0

        gex_by_strike = (
            opt_df.groupby(["strike", "type"])["gex_dollars"].sum().unstack(fill_value=0).reset_index()
        )
        for col in ["call", "put"]:
            if col not in gex_by_strike:
                gex_by_strike[col] = 0.0
        gex_by_strike["net_gex"] = gex_by_strike["call"] + gex_by_strike["put"]

        call_wall = float(gex_by_strike.loc[gex_by_strike["call"].idxmax(), "strike"])
        put_wall = float(gex_by_strike.loc[gex_by_strike["put"].idxmin(), "strike"])
        zero_gamma = compute_zero_gamma(gex_by_strike, spot)

        whale_calls = len(
            opt_df[(opt_df["type"] == "call") & (opt_df["vol_oi_ratio"] >= 1.5) & (opt_df["est_premium"] >= 50000)]
        )
        whale_puts = len(
            opt_df[(opt_df["type"] == "put") & (opt_df["vol_oi_ratio"] >= 1.5) & (opt_df["est_premium"] >= 50000)]
        )

        flow_edge = (call_pct - 50.0) * 1.2
        gap_edge = np.clip(tech["ext_gap_pct"] * 4.0, -20.0, 20.0)
        vwap_edge = 10.0 if tech["above_vwap"] else -10.0
        ema_edge = 10.0 if tech["ema_bullish"] else -10.0

        net_edge = flow_edge + gap_edge + vwap_edge + ema_edge
        direction = "🟢 HOT LONG (Buy)" if net_edge >= 0 else "🔴 HOT SHORT (Sell)"
        conviction_score = round(min(99.0, abs(net_edge) + 25.0), 1)

        return {
            "Ticker": ticker,
            "Signal": direction,
            "Last_Price": tech["latest_price"],
            "Ext_Gap_Pct": tech["ext_gap_pct"],
            "VWAP": tech["vwap"],
            "RSI_14": tech["rsi"],
            "Call_Flow_Pct": round(call_pct, 1),
            "Whale_Sweeps": f"{whale_calls}C / {whale_puts}P",
            "Put_Wall": put_wall,
            "Call_Wall": call_wall,
            "Zero_Gamma": zero_gamma,
            "Heat_Score": conviction_score,
            "_raw_edge": net_edge,
        }
    except Exception:
        return None


@st.cache_data(ttl=120)
def run_linked_whale_scanner():
    """
    Dynamically links Page 1 -> Page 2:
    Takes the Top 30 Bullish Movers + Top 30 Bearish Movers from Page 1 (+ Core Indices)
    and scans all of them for CBOE Options Whale Activity!
    """
    top_bulls, top_bears, combined_100 = fetch_sp_movers_and_shakers(SP500_LIQUID_UNIVERSE)

    tech_lookup = {}
    if not combined_100.empty:
        for _, r in combined_100.iterrows():
            tech_lookup[r["Ticker"]] = {
                "latest_price": float(r["Last_Price"]),
                "ext_gap_pct": float(r["Total_Move_Pct"]),
                "vwap": float(r["VWAP"]),
                "above_vwap": bool(r["above_vwap"]),
                "rsi": float(r["RSI_14"]),
                "ema_bullish": bool(r["ema_bullish"]),
            }

    core_benchmarks = ["SPY", "QQQ", "IWM", "DIA", "SMH", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "META"]
    p1_top_bulls = top_bulls.head(30)["Ticker"].tolist() if not top_bulls.empty else []
    p1_top_bears = top_bears.head(30)["Ticker"].tolist() if not top_bears.empty else []

    linked_universe = list(dict.fromkeys(core_benchmarks + p1_top_bulls + p1_top_bears))

    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(scan_single_whale_ticker, sym, tech_lookup.get(sym)): sym
            for sym in linked_universe
        }
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
    return pd.DataFrame(results)


@st.cache_data(ttl=3600)
def fetch_finra_darkpool_history(ticker_symbol, lookback_days=10):
    records = []
    today = dt.date.today()
    headers = {"User-Agent": "Mozilla/5.0"}
    for offset in range(1, lookback_days + 5):
        check_date = today - dt.timedelta(days=offset)
        if check_date.weekday() >= 5:
            continue
        date_str = check_date.strftime("%Y%m%d")
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), sep="|")
                row = df[df["Symbol"] == ticker_symbol.upper()]
                if not row.empty:
                    short_vol = float(row["ShortVolume"].iloc[0]) + float(row["ShortExemptVolume"].iloc[0])
                    total_vol = float(row["TotalVolume"].iloc[0])
                    pct = (short_vol / total_vol * 100) if total_vol > 0 else 0
                    records.append(
                        {
                            "Date": check_date.strftime("%Y-%m-%d"),
                            "Total_Vol": int(total_vol),
                            "Hidden_Buying_Pct": round(pct, 1),
                        }
                    )
            if len(records) >= 6:
                break
        except Exception:
            continue
    return pd.DataFrame(records).sort_values("Date") if records else pd.DataFrame()


# ==========================================
# 4. AUTOMATED BUY / SELL / WAIT TRADE TICKET ENGINE
# ==========================================
def build_automated_trade_ticket(spot, put_wall, call_wall, zero_gamma, vwap, rsi, call_pct, dp_pct=54.0):
    bull_points = 0
    bear_points = 0
    reasons = []

    if spot >= vwap:
        bull_points += 1
        reasons.append(f"✅ Price (${spot}) is holding ABOVE Today's Fair Value Price (${vwap})")
    else:
        bear_points += 1
        reasons.append(f"🔻 Price (${spot}) is trading BELOW Today's Fair Value Price (${vwap})")

    if spot >= zero_gamma:
        bull_points += 1
        reasons.append(f"✅ Price is ABOVE the Danger Line (${zero_gamma}) — Calm, dip-buying regime")
    else:
        bear_points += 1
        reasons.append(f"🔻 Price is BELOW the Danger Line (${zero_gamma}) — Fast, volatile regime")

    if call_pct >= 58.0:
        bull_points += 1
        reasons.append(f"✅ Smart-Money Options Flow is Bullish ({call_pct:.1f}% Call Buying)")
    elif call_pct <= 42.0:
        bear_points += 1
        reasons.append(f"🔻 Smart-Money Options Flow is Bearish ({100 - call_pct:.1f}% Put Buying)")
    else:
        reasons.append(f"⚖️ Smart-Money Options Flow is Mixed ({call_pct:.1f}% Calls)")

    if 35.0 <= rsi <= 68.0:
        bull_points += 1
        bear_points += 1
        reasons.append(f"✅ Fever Score (RSI {rsi}) is in the sweet spot (not overheated)")
    elif rsi > 68.0:
        reasons.append(f"⚠️ Fever Score (RSI {rsi}) is OVERHEATED (>68) — High pullback risk if chasing")
    else:
        bull_points += 1
        reasons.append(f"💡 Fever Score (RSI {rsi}) is ON SALE / OVERSOLD (<35) — Watch for bounce")

    if dp_pct >= 53.0:
        bull_points += 1
        reasons.append(f"✅ Hidden Dark Pool Activity ({dp_pct:.1f}%) shows quiet institutional buying")
    elif dp_pct > 0:
        bear_points += 1
        reasons.append(f"🔻 Hidden Dark Pool Activity ({dp_pct:.1f}%) is light (<53%)")

    eff_ceiling = call_wall if call_wall > spot * 1.003 else round(spot * 1.025, 2)
    eff_floor = put_wall if put_wall < spot * 0.997 else round(spot * 0.975, 2)

    room_to_ceiling_pct = ((eff_ceiling - spot) / spot) * 100.0
    room_to_floor_pct = ((spot - eff_floor) / spot) * 100.0

    if bull_points >= 4 and rsi <= 70.0 and room_to_ceiling_pct >= 0.8:
        verdict = "🟢 ACTIONABLE LONG (BUY CALLS OR SHARES)"
        short_verdict = "🟢 BUY LONG"
        bg_color = "#052e16"
        border_color = "#22c55e"
        entry_zone = f"${round(min(spot, vwap * 1.002), 2)} – ${spot}"
        anchor_support = min(vwap, zero_gamma) if min(vwap, zero_gamma) < spot else spot * 0.99
        stop_loss = round(max(eff_floor * 0.995, anchor_support * 0.993, spot * 0.985), 2)
        profit_target = round(eff_ceiling * 0.998, 2)
        risk = max(spot - stop_loss, 0.01)
        reward = max(profit_target - spot, 0.01)
    elif bear_points >= 3 and spot < vwap and room_to_floor_pct >= 0.8:
        verdict = "🔴 ACTIONABLE SHORT (BUY PUTS OR SHORT)"
        short_verdict = "🔴 SELL SHORT"
        bg_color = "#450a0a"
        border_color = "#ef4444"
        entry_zone = f"${spot} – ${round(max(spot, vwap * 0.998), 2)}"
        anchor_res = max(vwap, zero_gamma) if max(vwap, zero_gamma) > spot else spot * 1.01
        stop_loss = round(min(eff_ceiling * 1.005, anchor_res * 1.007, spot * 1.015), 2)
        profit_target = round(eff_floor * 1.002, 2)
        risk = max(stop_loss - spot, 0.01)
        reward = max(spot - profit_target, 0.01)
    else:
        verdict = "🟡 WAIT / NO CLEAR EDGE (SIT ON HANDS)"
        short_verdict = "🟡 WAIT (Pullback)"
        bg_color = "#1e293b"
        border_color = "#f59e0b"
        entry_zone = f"Wait near ${vwap}"
        stop_loss = round(eff_floor, 2)
        profit_target = round(eff_ceiling, 2)
        risk = max(abs(spot - stop_loss), 0.01)
        reward = max(abs(profit_target - spot), 0.01)

    rr_ratio = round(reward / risk, 2)
    return {
        "verdict": verdict,
        "short_verdict": short_verdict,
        "bg_color": bg_color,
        "border_color": border_color,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "profit_target": profit_target,
        "rr_ratio": rr_ratio,
        "bull_points": bull_points,
        "bear_points": bear_points,
        "reasons": reasons,
    }


# ==========================================
# 5. MASTER CONFLUENCE MERGE ENGINE (PAGE 1 + PAGE 2)
# ==========================================
@st.cache_data(ttl=120)
def build_master_confluence_watchlist():
    top_bulls, top_bears, combined_100 = fetch_sp_movers_and_shakers(SP500_LIQUID_UNIVERSE)
    whale_df = run_linked_whale_scanner()

    # Top 30 Bullish + Top 30 Bearish from Page 1
    p1_subset = pd.concat([top_bulls.head(30), top_bears.head(30)], ignore_index=True) if not combined_100.empty else pd.DataFrame()
    p1_map = {row["Ticker"]: row for _, row in p1_subset.iterrows()} if not p1_subset.empty else {}
    p1_all_map = {row["Ticker"]: row for _, row in combined_100.iterrows()} if not combined_100.empty else {}

    # All Whale-Screened Tickers from Page 2
    whale_all_map = {row["Ticker"]: row for _, row in whale_df.iterrows()} if not whale_df.empty else {}
    if not whale_df.empty:
        w_longs = whale_df[whale_df["_raw_edge"] >= 0].sort_values("Heat_Score", ascending=False).head(15)
        w_shorts = whale_df[whale_df["_raw_edge"] < 0].sort_values("Heat_Score", ascending=False).head(15)
        p2_top_set = set(pd.concat([w_longs, w_shorts])["Ticker"].tolist())
    else:
        p2_top_set = set()

    all_screened_tickers = sorted(list(set(p1_map.keys()) | p2_top_set))
    master_rows = []

    for sym in all_screened_tickers:
        in_p1 = sym in p1_map
        in_p2 = sym in p2_top_set
        m_row = p1_all_map.get(sym, {})
        w_row = whale_all_map.get(sym, {})

        if in_p1 and in_p2:
            tier = "⭐⭐ DOUBLE CONFIRMED (Page 1 Mover + Page 2 Whale)"
            tier_rank = 3
        elif in_p2:
            tier = "🔥 Options Whale Driven (Page 2 Top 20)"
            tier_rank = 2
        else:
            tier = "📈 Extended-Hours Mover (Page 1 Top Mover)"
            tier_rank = 1

        spot = float(w_row.get("Last_Price") or m_row.get("Last_Price") or 0.0)
        move_pct = float(m_row.get("Total_Move_Pct") or w_row.get("Ext_Gap_Pct") or 0.0)
        vwap = float(m_row.get("VWAP") or w_row.get("VWAP") or spot)
        rsi = float(m_row.get("RSI_14") or w_row.get("RSI_14") or 50.0)
        rvol = float(m_row.get("RVOL") or 1.2)

        call_pct = float(w_row.get("Call_Flow_Pct") or (62.0 if move_pct > 0 else 38.0))
        sweeps = str(w_row.get("Whale_Sweeps") or "0C / 0P")
        put_wall = float(w_row.get("Put_Wall") or m_row.get("Est_Floor") or round(spot * 0.97, 2))
        call_wall = float(w_row.get("Call_Wall") or m_row.get("Est_Ceiling") or round(spot * 1.03, 2))
        zero_gamma = float(w_row.get("Zero_Gamma") or vwap)

        ticket = build_automated_trade_ticket(
            spot=spot,
            put_wall=put_wall,
            call_wall=call_wall,
            zero_gamma=zero_gamma,
            vwap=vwap,
            rsi=rsi,
            call_pct=call_pct,
        )

        combined_heat = round(
            (float(m_row.get("Heat_Score", 60)) * 0.5)
            + (float(w_row.get("Heat_Score", 60)) * 0.5)
            + (12 if tier_rank == 3 else 0),
            1,
        )
        combined_heat = min(99.0, combined_heat)

        master_rows.append(
            {
                "Ticker": sym,
                "Confluence_Tier": tier,
                "Verdict": ticket["short_verdict"],
                "Last_Price": spot,
                "Total_Move_Pct": move_pct,
                "VWAP": vwap,
                "RSI_14": rsi,
                "RVOL": rvol,
                "Call_Flow_Pct": call_pct,
                "Whale_Sweeps": sweeps,
                "Entry_Zone": ticket["entry_zone"],
                "Stop_Loss": ticket["stop_loss"],
                "Profit_Target": ticket["profit_target"],
                "RR_Ratio": ticket["rr_ratio"],
                "Heat_Score": combined_heat,
                "_tier_rank": tier_rank,
            }
        )

    master_df = pd.DataFrame(master_rows)
    if not master_df.empty:
        master_df = master_df.sort_values(["_tier_rank", "Heat_Score"], ascending=[False, False]).reset_index(drop=True)
    return master_df


# ==========================================
# 6. COLUMN HEADER & STYLING HELPERS
# ==========================================
MOVERS_PLAIN_HEADERS = {
    "Ticker": "Stock Symbol",
    "Direction": "Market Direction",
    "Last_Price": "Last Traded Price ($)",
    "Total_Move_Pct": "Overnight & Today Move (%)",
    "Ext_Gap_Pct": "Pre/After-Market Gap (%)",
    "VWAP": "Fair Value Price ($)",
    "VWAP_Dist_Pct": "Distance from Fair Price (%)",
    "RSI_14": "Fever Score (RSI 0-100)",
    "RVOL": "Unusual Volume Surge (x)",
    "Trend_Signal": "Plain-English Trend",
    "Est_Floor": "Safety Net Floor ($)",
    "Est_Ceiling": "Profit Target Ceiling ($)",
    "Heat_Score": "Action Heat Score (0-100)",
}

MOVERS_PRO_HEADERS = {
    "Ticker": "Ticker",
    "Direction": "Regime Bias",
    "Last_Price": "Last Price ($)",
    "Total_Move_Pct": "Net Change (%)",
    "Ext_Gap_Pct": "Ext-Hours Gap (%)",
    "VWAP": "Session VWAP ($)",
    "VWAP_Dist_Pct": "VWAP Spread (%)",
    "RSI_14": "Ext RSI (14)",
    "RVOL": "Rel Volume (RVOL)",
    "Trend_Signal": "EMA(9/21) & VWAP Regime",
    "Est_Floor": "ATR Support Band ($)",
    "Est_Ceiling": "ATR Resistance Band ($)",
    "Heat_Score": "Composite Momentum (0-100)",
}


def build_movers_column_config(use_plain_headers=True):
    col_map = MOVERS_PLAIN_HEADERS if use_plain_headers else MOVERS_PRO_HEADERS
    return {
        col_map[k]: st.column_config.Column(
            label=col_map[k],
            help=GLOSSARY_TOOLTIPS.get(k, ""),
        )
        for k in col_map
    }


def style_dark_movers_table(df, use_plain_headers=True):
    display_cols = list(MOVERS_PLAIN_HEADERS.keys())
    clean_df = df[display_cols].copy().reset_index(drop=True)

    col_map = MOVERS_PLAIN_HEADERS if use_plain_headers else MOVERS_PRO_HEADERS
    clean_df = clean_df.rename(columns=col_map)

    move_col = col_map["Total_Move_Pct"]
    gap_col = col_map["Ext_Gap_Pct"]
    vwap_dist_col = col_map["VWAP_Dist_Pct"]
    rsi_col = col_map["RSI_14"]
    heat_col = col_map["Heat_Score"]
    price_col = col_map["Last_Price"]
    vwap_col = col_map["VWAP"]
    floor_col = col_map["Est_Floor"]
    ceil_col = col_map["Est_Ceiling"]
    rvol_col = col_map["RVOL"]

    def color_pct(val):
        if pd.isna(val):
            return ""
        if val > 2.0:
            return "background-color: #064e3b; color: #6ee7b7; font-weight: bold;"
        elif val > 0:
            return "background-color: #022c22; color: #34d399;"
        elif val < -2.0:
            return "background-color: #7f1d1d; color: #fca5a5; font-weight: bold;"
        elif val < 0:
            return "background-color: #450a0a; color: #f87171;"
        return "color: #94a3b8;"

    def color_rsi(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "background-color: #7c2d12; color: #fdba74; font-weight: bold;"
        elif val <= 30:
            return "background-color: #1e3a8a; color: #93c5fd; font-weight: bold;"
        return "color: #e2e8f0;"

    styler = (
        clean_df.style.map(color_pct, subset=[move_col, gap_col, vwap_dist_col])
        .map(color_rsi, subset=[rsi_col])
        .format(
            {
                price_col: "${:,.2f}",
                vwap_col: "${:,.2f}",
                floor_col: "${:,.2f}",
                ceil_col: "${:,.2f}",
                move_col: "{:+.2f}%",
                gap_col: "{:+.2f}%",
                vwap_dist_col: "{:+.2f}%",
                rsi_col: "{:.1f}",
                rvol_col: "{:.2f}x",
                heat_col: "{:.1f}",
            }
        )
    )
    return styler


# ==========================================
# 7. SIDEBAR NAVIGATION & GLOBAL LINKED STATE
# ==========================================
st.sidebar.title("🐋 OpenWhale Navigation")
st.sidebar.caption("100% Free Smart-Money, GEX & Extended-Hours Radar")

page = st.sidebar.radio(
    "Select Dashboard Page:",
    PAGES,
    key="nav_page",
    help="Click any row in Page 1, 2, or 3 to automatically jump to Page 4 (Deep-Dive & Trade Ticket) for that ticker!",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🎯 Currently Linked Ticker:** `{st.session_state.active_ticker}`")
use_plain_headers = st.sidebar.toggle(
    "🗣️ Use Everyday Plain-English Column Headers",
    value=True,
    help="ON: Uses beginner-friendly headers like 'Last Traded Price', 'Fever Score', and 'Safety Net Floor'. OFF: Uses institutional quant terms like 'VWAP', 'RSI-14', and 'RVOL'.",
)

# ==========================================
# PAGE 1: S&P MOVERS & SHAKERS
# ==========================================
if page == PAGES[0]:
    st.title("🐋 OpenWhale: S&P Movers & Shakers (Top 50 Bullish & Bearish)")
    render_hover_glossary_banner()

    col_btn, col_info = st.columns([0.25, 0.75])
    with col_btn:
        if st.button("🔄 Refresh S&P Movers Now", type="primary"):
            st.cache_data.clear()
    with col_info:
        st.info(
            "🔗 **Interactive Ticker Linking:** Click the checkbox on **any row** in the tables below (or use the Quick-Link bar) to immediately open that stock's **Deep-Dive & Trade Ticket (Page 4)**!"
        )

    with st.spinner("OpenWhale is scanning 125+ S&P & Nasdaq Liquid Leaders..."):
        top_bulls, top_bears, combined_100 = fetch_sp_movers_and_shakers(SP500_LIQUID_UNIVERSE)

    if not combined_100.empty:
        render_quick_jump_bar(combined_100["Ticker"].tolist(), key_prefix="p1_jump")

        b1, b2, b3, b4 = st.columns(4)
        top_winner = top_bulls.iloc[0]
        top_loser = top_bears.iloc[0]
        avg_rsi = round(combined_100["RSI_14"].mean(), 1)
        high_rvol_count = len(combined_100[combined_100["RVOL"] >= 1.5])

        b1.metric("🟢 #1 Bullish Leader", f"{top_winner['Ticker']} (${top_winner['Last_Price']})", f"{top_winner['Total_Move_Pct']:+.2f}%")
        b2.metric("🔴 #1 Bearish Shaker", f"{top_loser['Ticker']} (${top_loser['Last_Price']})", f"{top_loser['Total_Move_Pct']:+.2f}%")
        b3.metric("🌡️ Market Fever (Avg RSI)", f"{avg_rsi}", "Overheated >70 | On Sale <30", help=GLOSSARY_TOOLTIPS["RSI_14"])
        b4.metric("⚡ Unusual Volume Surge (>1.5x)", f"{high_rvol_count} Stocks", "Big Money Active", help=GLOSSARY_TOOLTIPS["RVOL"])

        st.subheader("🗺️ OpenWhale Dark-View Heatmap: Top 50 Bullish vs. Top 50 Bearish Movers")
        fig_heat = px.treemap(
            combined_100,
            path=[px.Constant("OpenWhale S&P Top 100 Movers & Shakers"), "Direction", "Ticker"],
            values="_abs_move",
            color="Total_Move_Pct",
            color_continuous_scale=[
                [0.0, "#7f1d1d"],
                [0.35, "#dc2626"],
                [0.50, "#1e293b"],
                [0.65, "#16a34a"],
                [1.0, "#059669"],
            ],
            color_continuous_midpoint=0.0,
            custom_data=[
                "Last_Price", "Total_Move_Pct", "Ext_Gap_Pct", "VWAP",
                "RSI_14", "RVOL", "Trend_Signal", "Est_Floor", "Est_Ceiling",
            ],
        )
        fig_heat.update_traces(
            texttemplate="<b>%{label}</b><br>$%{customdata[0]:.2f}<br>%{customdata[1]:+.2f}%",
            textposition="middle center",
            hovertemplate=(
                "<b>%{label} — OpenWhale Plain-English Breakdown</b><br>"
                "• <b>Last Traded Price:</b> $%{customdata[0]:.2f}<br>"
                "• <b>Total Move vs Yesterday:</b> %{customdata[1]:+.2f}%<br>"
                "• <b>Pre/After-Market Gap Only:</b> %{customdata[2]:+.2f}%<br>"
                "• <b>Fair Value Price (VWAP):</b> $%{customdata[3]:.2f}<br>"
                "• <b>Fever Score (RSI 0-100):</b> %{customdata[4]:.1f}<br>"
                "• <b>Unusual Volume Surge:</b> %{customdata[5]:.2f}x<br>"
                "• <b>Trend Verdict:</b> %{customdata[6]}<br>"
                "• <b>Safety Net Floor / Target Ceiling:</b> $%{customdata[7]:.2f} – $%{customdata[8]:.2f}<extra></extra>"
            ),
        )
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=520, margin=dict(t=25, l=10, r=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("📋 Tabulated Movers & Shakers (Click Any Row to Jump to Page 4 Deep-Dive)")
        col_cfg = build_movers_column_config(use_plain_headers)

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(
            [
                "🟢 Top 50 Bullish Movers (Click Row to Deep-Dive)",
                "🔴 Top 50 Bearish Shakers (Click Row to Deep-Dive)",
                "📊 All 100 Combined Table",
            ]
        )

        with sub_tab1:
            bulls_reset = top_bulls.reset_index(drop=True)
            ev1 = st.dataframe(
                style_dark_movers_table(bulls_reset, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=560,
                on_select="rerun",
                selection_mode="single-row",
                key="p1_bulls_table",
            )
            if ev1.selection.rows:
                picked = bulls_reset.iloc[ev1.selection.rows[0]]["Ticker"]
                jump_to_deep_dive(picked)
                st.rerun()

        with sub_tab2:
            bears_reset = top_bears.reset_index(drop=True)
            ev2 = st.dataframe(
                style_dark_movers_table(bears_reset, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=560,
                on_select="rerun",
                selection_mode="single-row",
                key="p1_bears_table",
            )
            if ev2.selection.rows:
                picked = bears_reset.iloc[ev2.selection.rows[0]]["Ticker"]
                jump_to_deep_dive(picked)
                st.rerun()

        with sub_tab3:
            comb_reset = combined_100.reset_index(drop=True)
            ev3 = st.dataframe(
                style_dark_movers_table(comb_reset, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=560,
                on_select="rerun",
                selection_mode="single-row",
                key="p1_comb_table",
            )
            if ev3.selection.rows:
                picked = comb_reset.iloc[ev3.selection.rows[0]]["Ticker"]
                jump_to_deep_dive(picked)
                st.rerun()


# ==========================================
# PAGE 2: TOP 20 OPTIONS WHALE SCANNER (LINKED TO PAGE 1)
# ==========================================
elif page == PAGES[1]:
    st.title("🐋 OpenWhale: Top 20 Hot Stocks Today (Linked to Page 1 Movers)")
    render_hover_glossary_banner()
    st.info(
        "🔗 **Linked Pipeline:** This page automatically takes the **Top Bullish & Bearish Movers from Page 1** and scans their live CBOE Options Chains. **Click any row** below to open that stock's Deep-Dive & Trade Ticket on Page 4!"
    )

    with st.spinner("Scanning CBOE Options Chains for Page 1's Top Movers..."):
        scan_df = run_linked_whale_scanner()

    if not scan_df.empty:
        render_quick_jump_bar(scan_df["Ticker"].tolist(), key_prefix="p2_jump")

        whale_col_map = (
            {
                "Ticker": "Stock Symbol",
                "Signal": "Plain-English Verdict",
                "Last_Price": "Last Traded Price ($)",
                "Ext_Gap_Pct": "Overnight & Today Move (%)",
                "RSI_14": "Fever Score (RSI)",
                "Call_Flow_Pct": "Smart Money Call Buying (%)",
                "Whale_Sweeps": "Urgent Whale Bets (Calls/Puts)",
                "Put_Wall": "Price Floor / Support ($)",
                "Call_Wall": "Price Ceiling / Target ($)",
                "Heat_Score": "Conviction Score (0-100)",
            }
            if use_plain_headers
            else {
                "Ticker": "Ticker",
                "Signal": "Bias",
                "Last_Price": "Spot ($)",
                "Ext_Gap_Pct": "Ext Move (%)",
                "RSI_14": "RSI(14)",
                "Call_Flow_Pct": "Call Prem (%)",
                "Whale_Sweeps": "Sweeps (Vol>1.5x OI)",
                "Put_Wall": "Put Wall (-GEX)",
                "Call_Wall": "Call Wall (+GEX)",
                "Heat_Score": "Heat Score",
            }
        )

        display_whale_cols = list(whale_col_map.keys())
        whale_help_cfg = {
            whale_col_map["Ticker"]: st.column_config.Column(help=GLOSSARY_TOOLTIPS["Ticker"]),
            whale_col_map["Signal"]: st.column_config.Column(help=GLOSSARY_TOOLTIPS["Direction"]),
            whale_col_map["Last_Price"]: st.column_config.NumberColumn(format="$%.2f", help=GLOSSARY_TOOLTIPS["Last_Price"]),
            whale_col_map["Ext_Gap_Pct"]: st.column_config.NumberColumn(format="%+.2f%%", help=GLOSSARY_TOOLTIPS["Total_Move_Pct"]),
            whale_col_map["RSI_14"]: st.column_config.NumberColumn(format="%.1f", help=GLOSSARY_TOOLTIPS["RSI_14"]),
            whale_col_map["Call_Flow_Pct"]: st.column_config.NumberColumn(format="%.1f%%", help=GLOSSARY_TOOLTIPS["Call_Flow_Pct"]),
            whale_col_map["Whale_Sweeps"]: st.column_config.Column(help=GLOSSARY_TOOLTIPS["Whale_Sweeps"]),
            whale_col_map["Put_Wall"]: st.column_config.NumberColumn(format="$%.2f", help=GLOSSARY_TOOLTIPS["Put_Wall"]),
            whale_col_map["Call_Wall"]: st.column_config.NumberColumn(format="$%.2f", help=GLOSSARY_TOOLTIPS["Call_Wall"]),
            whale_col_map["Heat_Score"]: st.column_config.NumberColumn(format="%.1f", help=GLOSSARY_TOOLTIPS["Heat_Score"]),
        }

        longs_raw = scan_df[scan_df["_raw_edge"] >= 0].sort_values("Heat_Score", ascending=False).head(10).reset_index(drop=True)
        shorts_raw = scan_df[scan_df["_raw_edge"] < 0].sort_values("Heat_Score", ascending=False).head(10).reset_index(drop=True)

        longs_shown = longs_raw[display_whale_cols].rename(columns=whale_col_map)
        shorts_shown = shorts_raw[display_whale_cols].rename(columns=whale_col_map)

        c_l, c_s = st.columns(2)
        with c_l:
            st.markdown("### 🟢 Top 10 Hot LONGS (Click Row to Deep-Dive)")
            ev_l = st.dataframe(
                longs_shown,
                column_config=whale_help_cfg,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="p2_longs_table",
            )
            if ev_l.selection.rows:
                picked = longs_raw.iloc[ev_l.selection.rows[0]]["Ticker"]
                jump_to_deep_dive(picked)
                st.rerun()

        with c_s:
            st.markdown("### 🔴 Top 10 Hot SHORTS (Click Row to Deep-Dive)")
            ev_s = st.dataframe(
                shorts_shown,
                column_config=whale_help_cfg,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="p2_shorts_table",
            )
            if ev_s.selection.rows:
                picked = shorts_raw.iloc[ev_s.selection.rows[0]]["Ticker"]
                jump_to_deep_dive(picked)
                st.rerun()


# ==========================================
# PAGE 3: MASTER ACTIONABLE WATCHLIST (PAGE 1 + 2 COMBINED)
# ==========================================
elif page == PAGES[2]:
    st.title("⚡ OpenWhale Master Actionable Watchlist (Page 1 + Page 2 Linked)")
    render_hover_glossary_banner()
    st.info(
        "🔗 **Click Any Row to Open Its Deep-Dive Charts & Trade Ticket:** This table unites all screened tickers from Page 1 and Page 2. Select any row below to inspect its GEX Walls and Dark Pool print history on Page 4."
    )

    with st.spinner("Linking Page 1 Movers & Page 2 Options Whales into Master Watchlist..."):
        master_df = build_master_confluence_watchlist()

    if not master_df.empty:
        render_quick_jump_bar(master_df["Ticker"].tolist(), key_prefix="p3_jump")

        double_conf_count = len(master_df[master_df["_tier_rank"] == 3])
        buy_long_count = len(master_df[master_df["Verdict"] == "🟢 BUY LONG"])
        sell_short_count = len(master_df[master_df["Verdict"] == "🔴 SELL SHORT"])
        high_rr_count = len(master_df[master_df["RR_Ratio"] >= 1.5])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⭐⭐ Double-Confirmed Stocks", f"{double_conf_count} Tickers", "Page 1 + Page 2 Overlap", help=GLOSSARY_TOOLTIPS["Confluence_Tier"])
        k2.metric("🟢 Actionable BUY LONG Tickets", f"{buy_long_count} Setups", "4+ Bullish Factors Aligned", help=GLOSSARY_TOOLTIPS["Verdict"])
        k3.metric("🔴 Actionable SELL SHORT Tickets", f"{sell_short_count} Setups", "Bearish Breakdown Confirmed", help=GLOSSARY_TOOLTIPS["Verdict"])
        k4.metric("⚖️ Setups w/ Reward:Risk >= 1.5x", f"{high_rr_count} Tickers", "High Asymmetry", help=GLOSSARY_TOOLTIPS["RR_Ratio"])

        f1, f2, f3 = st.columns([0.4, 0.35, 0.25])
        with f1:
            tier_filter = st.multiselect(
                "Filter by Confirmation Source:",
                options=[
                    "⭐⭐ DOUBLE CONFIRMED (Page 1 Mover + Page 2 Whale)",
                    "🔥 Options Whale Driven (Page 2 Top 20)",
                    "📈 Extended-Hours Mover (Page 1 Top Mover)",
                ],
                default=[
                    "⭐⭐ DOUBLE CONFIRMED (Page 1 Mover + Page 2 Whale)",
                    "🔥 Options Whale Driven (Page 2 Top 20)",
                    "📈 Extended-Hours Mover (Page 1 Top Mover)",
                ],
            )
        with f2:
            verdict_filter = st.multiselect(
                "Filter by Trade Ticket Verdict:",
                options=["🟢 BUY LONG", "🔴 SELL SHORT", "🟡 WAIT (Pullback)"],
                default=["🟢 BUY LONG", "🔴 SELL SHORT", "🟡 WAIT (Pullback)"],
            )
        with f3:
            min_rr = st.slider("Minimum Reward:Risk Ratio", 0.5, 4.0, 1.0, 0.25)

        filtered_master = (
            master_df[
                (master_df["Confluence_Tier"].isin(tier_filter))
                & (master_df["Verdict"].isin(verdict_filter))
                & (master_df["RR_Ratio"] >= min_rr)
            ]
            .copy()
            .reset_index(drop=True)
        )

        master_col_map = (
            {
                "Ticker": "Stock Symbol",
                "Confluence_Tier": "Why It's Here (Source)",
                "Verdict": "Trade Ticket Verdict",
                "Last_Price": "Last Traded Price ($)",
                "Total_Move_Pct": "Overnight & Today Move (%)",
                "VWAP": "Fair Price VWAP ($)",
                "RSI_14": "Fever Score (RSI)",
                "RVOL": "Volume Surge (x)",
                "Call_Flow_Pct": "Smart Money Call %",
                "Whale_Sweeps": "Whale Sweeps (C/P)",
                "Entry_Zone": "Ideal Entry Zone",
                "Stop_Loss": "Hard Stop-Loss ($)",
                "Profit_Target": "Profit Target ($)",
                "RR_Ratio": "Reward:Risk (x)",
                "Heat_Score": "Master Score (0-100)",
            }
            if use_plain_headers
            else {
                "Ticker": "Ticker",
                "Confluence_Tier": "Confluence Tier",
                "Verdict": "Ticket Signal",
                "Last_Price": "Spot ($)",
                "Total_Move_Pct": "Net Change (%)",
                "VWAP": "Ext VWAP ($)",
                "RSI_14": "RSI(14)",
                "RVOL": "RVOL (x)",
                "Call_Flow_Pct": "Call Flow (%)",
                "Whale_Sweeps": "Sweeps (Vol>1.5x OI)",
                "Entry_Zone": "Entry Band",
                "Stop_Loss": "Stop Loss ($)",
                "Profit_Target": "Wall Target ($)",
                "RR_Ratio": "R:R Ratio",
                "Heat_Score": "Composite Score",
            }
        )

        display_master_cols = list(master_col_map.keys())
        shown_df = filtered_master[display_master_cols].rename(columns=master_col_map)

        master_help_cfg = {
            master_col_map[k]: st.column_config.Column(
                label=master_col_map[k],
                help=GLOSSARY_TOOLTIPS.get(k, ""),
            )
            for k in display_master_cols
        }

        ev_m = st.dataframe(
            shown_df,
            column_config=master_help_cfg,
            use_container_width=True,
            hide_index=True,
            height=540,
            on_select="rerun",
            selection_mode="single-row",
            key="p3_master_table",
        )
        if ev_m.selection.rows:
            picked = filtered_master.iloc[ev_m.selection.rows[0]]["Ticker"]
            jump_to_deep_dive(picked)
            st.rerun()

        csv_bytes = shown_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Today's Master Actionable Watchlist (CSV)",
            data=csv_bytes,
            file_name=f"openwhale_master_watchlist_{dt.date.today()}.csv",
            mime="text/csv",
        )


# ==========================================
# PAGE 4: SINGLE-TICKER DEEP DIVE + AUTOMATED TRADE TICKET
# ==========================================
elif page == PAGES[3]:
    # Pulls the linked ticker from Page 1, 2, or 3 via st.session_state.active_ticker
    master_df_for_nav = build_master_confluence_watchlist()
    screened_list = master_df_for_nav["Ticker"].tolist() if not master_df_for_nav.empty else SP500_LIQUID_UNIVERSE[:30]

    c_nav1, c_nav2, c_nav3 = st.columns([0.45, 0.35, 0.20])
    with c_nav1:
        if st.session_state.active_ticker not in screened_list:
            screened_list = [st.session_state.active_ticker] + screened_list
        chosen_from_list = st.selectbox(
            "🔗 Switch Between Today's Screened Tickers (from Pages 1–3):",
            options=screened_list,
            index=screened_list.index(st.session_state.active_ticker),
        )
        if chosen_from_list != st.session_state.active_ticker:
            st.session_state.active_ticker = chosen_from_list
            st.rerun()

    with c_nav2:
        manual_sym = st.text_input(
            "Or Type Any Custom Ticker / Index (e.g. SPX, NVDA):",
            value=st.session_state.active_ticker,
        ).upper().strip()
        if manual_sym != st.session_state.active_ticker:
            st.session_state.active_ticker = manual_sym
            st.rerun()

    with c_nav3:
        num_expiries = st.slider("Options Expiries:", 1, 8, 4)

    selected_ticker = st.session_state.active_ticker
    st.title(f"🐋 OpenWhale Deep-Dive & Automated Trade Ticket: {selected_ticker}")
    render_hover_glossary_banner()

    try:
        spot, full_chain = fetch_cboe_chain(selected_ticker, num_expiries)
        finra_df = fetch_finra_darkpool_history(selected_ticker)
        try:
            tech = fetch_extended_hours_single(selected_ticker)
            spot = tech["latest_price"]
        except Exception:
            tech = {"latest_price": spot, "ext_gap_pct": 0.0, "vwap": spot, "above_vwap": True, "rsi": 50.0}

        gex_by_strike = (
            full_chain.groupby(["strike", "type"])["gex_dollars"].sum().unstack(fill_value=0).reset_index()
        )
        for col in ["call", "put"]:
            if col not in gex_by_strike:
                gex_by_strike[col] = 0.0
        gex_by_strike["net_gex"] = gex_by_strike["call"] + gex_by_strike["put"]

        call_wall = float(gex_by_strike.loc[gex_by_strike["call"].idxmax(), "strike"])
        put_wall = float(gex_by_strike.loc[gex_by_strike["put"].idxmin(), "strike"])
        zero_gamma = compute_zero_gamma(gex_by_strike, spot)

        call_prem = full_chain[full_chain["type"] == "call"]["est_premium"].sum()
        put_prem = full_chain[full_chain["type"] == "put"]["est_premium"].sum()
        call_pct = (call_prem / (call_prem + put_prem) * 100.0) if (call_prem + put_prem) > 0 else 50.0
        latest_dp_pct = float(finra_df["Hidden_Buying_Pct"].iloc[-1]) if not finra_df.empty else 50.0

        ticket = build_automated_trade_ticket(
            spot=spot,
            put_wall=put_wall,
            call_wall=call_wall,
            zero_gamma=zero_gamma,
            vwap=tech["vwap"],
            rsi=tech["rsi"],
            call_pct=call_pct,
            dp_pct=latest_dp_pct,
        )

        reasons_html = "".join([f"<li style='margin-bottom:4px;'>{r}</li>" for r in ticket["reasons"]])
        st.markdown(
            f"""
            <div class="ticket-box" style="background-color:{ticket['bg_color']}; border-color:{ticket['border_color']};">
                <h3 style="margin-top:0; color:#f8fafc;">🎫 OpenWhale Automated Trade Ticket ({selected_ticker}): {ticket['verdict']}</h3>
                <p style="color:#cbd5e1; margin-bottom:8px;"><b>5-Point Confluence Check (Bullish Score: {ticket['bull_points']}/5 | Bearish Score: {ticket['bear_points']}/5):</b></p>
                <ul style="color:#e2e8f0; margin-bottom:0;">{reasons_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("🎯 Ideal Entry Zone", f"{ticket['entry_zone']}", help=GLOSSARY_TOOLTIPS["Entry_Zone"])
        t2.metric("🛑 Hard Stop-Loss", f"${ticket['stop_loss']}", help=GLOSSARY_TOOLTIPS["Stop_Loss"])
        t3.metric("💰 Profit Target", f"${ticket['profit_target']}", help=GLOSSARY_TOOLTIPS["Profit_Target"])
        t4.metric(
            "⚖️ Reward-to-Risk Ratio",
            f"{ticket['rr_ratio']} : 1",
            "Target >= 1.5 : 1" if ticket["rr_ratio"] >= 1.5 else "Tight R:R (<1.5)",
            help=GLOSSARY_TOOLTIPS["RR_Ratio"],
        )

        st.markdown("---")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Last Traded Price", f"${spot}", f"{tech['ext_gap_pct']:+.2f}%", help=GLOSSARY_TOOLTIPS["Last_Price"])
        m2.metric("Price Floor (Safety Net)" if use_plain_headers else "Put Wall (-GEX)", f"${put_wall}", help=GLOSSARY_TOOLTIPS["Put_Wall"])
        m3.metric("Price Ceiling (Target)" if use_plain_headers else "Call Wall (+GEX)", f"${call_wall}", help=GLOSSARY_TOOLTIPS["Call_Wall"])
        m4.metric("Danger Line (Pivot)" if use_plain_headers else "Zero Gamma Flip", f"${zero_gamma}", help=GLOSSARY_TOOLTIPS["Zero_Gamma"])
        m5.metric("Fair Price (VWAP)" if use_plain_headers else "Extended VWAP", f"${tech['vwap']}", help=GLOSSARY_TOOLTIPS["VWAP"])
        m6.metric("Fever Score (RSI)" if use_plain_headers else "Ext RSI(14)", f"{tech['rsi']}", help=GLOSSARY_TOOLTIPS["RSI_14"])

        col_l, col_r = st.columns([0.62, 0.38])
        with col_l:
            plot_df = gex_by_strike[(gex_by_strike["strike"] >= spot * 0.92) & (gex_by_strike["strike"] <= spot * 1.08)]
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in plot_df["net_gex"]]

            fig = go.Figure(
                go.Bar(
                    x=plot_df["strike"],
                    y=plot_df["net_gex"] / 1e6,
                    marker_color=colors,
                    name="Net GEX ($M)",
                    hovertemplate=(
                        "<b>Strike Price: $%{x}</b><br>"
                        "Dealer Hedging Power (Net GEX): $%{y:.2f}M<br>"
                        "<i>Green = Stabilizing Ceiling/Magnet | Red = Volatility Floor/Trapdoor</i><extra></extra>"
                    ),
                )
            )
            fig.add_vline(x=spot, line_dash="dash", line_color="#3b82f6", annotation_text=f"Last Price: ${spot}")
            fig.add_vline(x=zero_gamma, line_dash="dot", line_color="#f59e0b", annotation_text=f"Danger Line: ${zero_gamma}")
            fig.update_layout(
                title=f"{selected_ticker} Price Floors (Red) & Price Ceilings (Green)",
                template="plotly_dark",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if not finra_df.empty:
                fig_dp = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dp.add_trace(
                    go.Bar(x=finra_df["Date"], y=finra_df["Total_Vol"], name="Dark Pool Volume", marker_color="#6366f1", opacity=0.6),
                    secondary_y=False,
                )
                fig_dp.add_trace(
                    go.Scatter(
                        x=finra_df["Date"],
                        y=finra_df["Hidden_Buying_Pct"],
                        name="Hidden Buying %",
                        mode="lines+markers",
                        line=dict(color="#10b981", width=3),
                    ),
                    secondary_y=True,
                )
                fig_dp.add_hline(y=50, line_dash="dot", line_color="#94a3b8", secondary_y=True)
                fig_dp.update_layout(title=f"{selected_ticker} FINRA Dark Pool Buying %", template="plotly_dark", height=420)
                st.plotly_chart(fig_dp, use_container_width=True)
            else:
                st.info("Dark Pool volume is reported for stocks & ETFs (not cash indices like SPX).")

    except Exception as e:
        st.error(f"Could not load deep-dive data for {selected_ticker}: {e}")


# ==========================================
# PAGE 5: HELP & TRADING PLAYBOOK (USER MANUAL)
# ==========================================
else:
    st.title("📖 OpenWhale Help Center & Everyday Trading Playbook")
    st.markdown(
        """
        ### 🔗 How Cross-Page Ticker Linking Works
        1. **Page 1 (`S&P Movers & Shakers`)** scans 125+ S&P & Nasdaq leaders and automatically feeds the day's top bullish and bearish movers into **Page 2 (`Options Whale Scanner`)** and **Page 3 (`Master Actionable Watchlist`)**.
        2. **Click Any Row in Any Table:** On Pages 1, 2, or 3, clicking the selection box on the left side of any row immediately opens that ticker inside **Page 4 (`Single-Ticker Deep Dive + Trade Ticket`)**.
        3. **Switch Tickers Effortlessly on Page 4:** Use the dropdown selector at the top of Page 4 to cycle through all of today's screened tickers without having to go back and forth.
        """
    )
