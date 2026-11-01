from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import io
import re
import time
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
# HOVER GLOSSARY & PLAIN-ENGLISH DEFINITIONS
# ==========================================
GLOSSARY_TOOLTIPS = {
    "Ticker": "The short stock market code for the company or index (e.g., AAPL = Apple, SPY = S&P 500).",
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
    "Net_GEX": "Net Gamma Exposure ($M): Positive (+) means big banks stabilize the market (good for buying dips). Negative (-) means big banks accelerate fast breakouts or crashes.",
    "Dark_Pool_Pct": "Off-Exchange Hidden Volume %: Tracks private institutional trades reported to FINRA. Readings above 55% on a green/flat day signal quiet institutional accumulation.",
}

# Inject CSS for interactive hover-definition pills at the top of the dashboard
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
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hover_glossary_banner():
    """Renders an always-accessible hover bar where hovering any term pops up a plain-English card."""
    pills_html = """
    <div class="ow-glossary-bar">
        <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">💡 Hover any term for Plain-English meaning:</span>
        <div class="ow-pill">Price Floor (Put Wall)<span class="ow-tip"><b>Price Floor / Put Wall:</b> The price level where big banks mechanically step in and buy shares to hedge options. Acts like a trampoline safety net.</span></div>
        <div class="ow-pill">Price Ceiling (Call Wall)<span class="ow-tip"><b>Price Ceiling / Call Wall:</b> The price level where big banks start selling shares to hedge options. Acts like a brick ceiling—great place to take profits.</span></div>
        <div class="ow-pill">Danger Line (Zero Gamma)<span class="ow-tip"><b>Danger Line (Flip Level):</b> Above this price, trading is calm and dips get bought. Below this price, swings become fast and violent.</span></div>
        <div class="ow-pill">Fair Price (VWAP)<span class="ow-tip"><b>Fair Value Price (VWAP):</b> The average price paid per share today weighted by volume (including Pre-Market & After-Hours). Above = Buyers winning.</span></div>
        <div class="ow-pill">Fever Score (RSI)<span class="ow-tip"><b>Fever Score (RSI 0–100):</b> Measures if a stock is running too hot. Above 70 = Overheated (due for a breather). Below 30 = On Sale / Oversold (due for a bounce).</span></div>
        <div class="ow-pill">Volume Surge (RVOL)<span class="ow-tip"><b>Unusual Volume Surge (RVOL):</b> Compares today's activity to a normal day. 2.0x means double the normal volume—proof that big money is active.</span></div>
        <div class="ow-pill">Whale Sweeps (Vol > OI)<span class="ow-tip"><b>Whale Sweeps:</b> Urgent options orders where today's volume exceeds total existing contracts (Open Interest), proving brand-new smart-money bets are opening.</span></div>
        <div class="ow-pill">Dark Pool Buying %<span class="ow-tip"><b>Hidden Dark Pool %:</b> Private off-exchange institutional volume reported to FINRA. Over 55% on a green or flat day indicates quiet institutional accumulation.</span></div>
    </div>
    """
    st.markdown(pills_html, unsafe_allow_html=True)


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

DEFAULT_WHALE_UNIVERSE = SP500_LIQUID_UNIVERSE[:40]
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
# 2. BATCH S&P MOVERS & SHAKERS ENGINE
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
# 3. CBOE OPTIONS & WHALE ENGINE
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


def scan_single_whale_ticker(ticker):
    try:
        spot, opt_df = fetch_cboe_chain(ticker, num_expiries=3)
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

        gex_by_strike = opt_df.groupby(["strike", "type"])["gex_dollars"].sum().unstack(fill_value=0)
        call_wall = gex_by_strike["call"].idxmax() if "call" in gex_by_strike else round(spot * 1.05, 2)
        put_wall = gex_by_strike["put"].idxmin() if "put" in gex_by_strike else round(spot * 0.95, 2)

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
            "RSI_14": tech["rsi"],
            "Call_Flow_Pct": round(call_pct, 1),
            "Whale_Sweeps": f"{whale_calls}C / {whale_puts}P",
            "Put_Wall": put_wall,
            "Call_Wall": call_wall,
            "Heat_Score": conviction_score,
            "_raw_edge": net_edge,
        }
    except Exception:
        return None


@st.cache_data(ttl=180)
def run_whale_scanner(universe_list):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_single_whale_ticker, sym): sym for sym in universe_list}
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
# 4. COLUMN HEADER & HOVER TOOLTIP CONFIG
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
    """Attaches a plain-English hover tooltip (`help=...`) to every column header in st.dataframe."""
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
    clean_df = df[display_cols].copy()

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
# 5. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# ==========================================
st.sidebar.title("🐋 OpenWhale Navigation")
st.sidebar.caption("100% Free Smart-Money, GEX & Extended-Hours Radar")

page = st.sidebar.radio(
    "Select Dashboard Page:",
    [
        "🏆 S&P Movers & Shakers (Top 50 Each Way)",
        "🔥 Top 20 Options Whale Scanner",
        "🔍 Single-Ticker Deep Dive",
    ],
    help="Switch between the Top 100 S&P Extended-Hours Movers, the Top 20 Unusual Options Whale Scanner, or a Single-Stock Deep Dive.",
)

st.sidebar.markdown("---")
use_plain_headers = st.sidebar.toggle(
    "🗣️ Use Everyday Plain-English Column Headers",
    value=True,
    help="ON: Uses beginner-friendly headers like 'Last Traded Price', 'Fever Score', and 'Safety Net Floor'. OFF: Uses institutional quant terms like 'VWAP', 'RSI-14', and 'RVOL'. Note: Hovering ANY column header always shows the Plain-English definition!",
)

# ==========================================
# PAGE 1: S&P MOVERS & SHAKERS (TOP 50 IN EACH DIRECTION)
# ==========================================
if page == "🏆 S&P Movers & Shakers (Top 50 Each Way)":
    st.title("🐋 OpenWhale: S&P Movers & Shakers (Top 50 Bullish & Bearish)")
    render_hover_glossary_banner()

    col_btn, col_info = st.columns([0.25, 0.75])
    with col_btn:
        if st.button("🔄 Refresh S&P Movers Now", type="primary", help="Clears cache and pulls fresh Pre-Market, Intraday, and After-Hours trades."):
            st.cache_data.clear()
    with col_info:
        st.info(
            "🖱️ **Hover Over Any Column Header or Metric:** Every table column and KPI card below has a built-in Plain-English tooltip when you hover your mouse over it."
        )

    with st.spinner("OpenWhale is scanning 125+ S&P & Nasdaq Liquid Leaders (including Pre-Market & After-Hours tape)..."):
        top_bulls, top_bears, combined_100 = fetch_sp_movers_and_shakers(SP500_LIQUID_UNIVERSE)

    if not combined_100.empty:
        b1, b2, b3, b4 = st.columns(4)
        top_winner = top_bulls.iloc[0]
        top_loser = top_bears.iloc[0]
        avg_rsi = round(combined_100["RSI_14"].mean(), 1)
        high_rvol_count = len(combined_100[combined_100["RVOL"] >= 1.5])

        b1.metric(
            "🟢 #1 Bullish Leader",
            f"{top_winner['Ticker']} (${top_winner['Last_Price']})",
            f"{top_winner['Total_Move_Pct']:+.2f}%",
            help="The strongest upward-moving stock in our S&P liquid universe right now, including Pre-Market and After-Hours trading.",
        )
        b2.metric(
            "🔴 #1 Bearish Shaker",
            f"{top_loser['Ticker']} (${top_loser['Last_Price']})",
            f"{top_loser['Total_Move_Pct']:+.2f}%",
            help="The sharpest downward-moving stock in our S&P liquid universe right now, including Pre-Market and After-Hours trading.",
        )
        b3.metric(
            "🌡️ Market Fever (Avg RSI)",
            f"{avg_rsi}",
            "Overheated >70 | On Sale <30",
            help=GLOSSARY_TOOLTIPS["RSI_14"],
        )
        b4.metric(
            "⚡ Unusual Volume Surge (>1.5x)",
            f"{high_rvol_count} Stocks",
            "Big Money Active",
            help=GLOSSARY_TOOLTIPS["RVOL"],
        )

        # DARK-VIEW TREEMAP HEATMAP
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
                "Last_Price",
                "Total_Move_Pct",
                "Ext_Gap_Pct",
                "VWAP",
                "RSI_14",
                "RVOL",
                "Trend_Signal",
                "Est_Floor",
                "Est_Ceiling",
            ],
        )

        fig_heat.update_traces(
            texttemplate="<b>%{label}</b><br>$%{customdata[0]:.2f}<br>%{customdata[1]:+.2f}%",
            textposition="middle center",
            hovertemplate=(
                "<b>%{label} — OpenWhale Plain-English Breakdown</b><br>"
                "• <b>Last Traded Price:</b> $%{customdata[0]:.2f} <i>(includes Pre/After-Hours)</i><br>"
                "• <b>Total Move vs Yesterday:</b> %{customdata[1]:+.2f}%<br>"
                "• <b>Pre/After-Market Gap Only:</b> %{customdata[2]:+.2f}%<br>"
                "• <b>Fair Value Price (VWAP):</b> $%{customdata[3]:.2f} <i>(Avg price paid today)</i><br>"
                "• <b>Fever Score (RSI 0-100):</b> %{customdata[4]:.1f} <i>(>70 Overheated, <30 On Sale)</i><br>"
                "• <b>Unusual Volume Surge:</b> %{customdata[5]:.2f}x normal volume<br>"
                "• <b>Trend Verdict:</b> %{customdata[6]}<br>"
                "• <b>Safety Net Floor / Target Ceiling:</b> $%{customdata[7]:.2f} – $%{customdata[8]:.2f}<extra></extra>"
            ),
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=560,
            margin=dict(t=30, l=10, r=10, b=10),
            coloraxis_colorbar=dict(title="Move %"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # TABULATED COLOR-CODED TABLES WITH HOVER COLUMN TOOLTIPS
        st.subheader("📋 Tabulated Movers & Shakers (Hover Any Column Header for Plain-English Definition)")
        col_cfg = build_movers_column_config(use_plain_headers)

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(
            [
                "🟢 Top 50 Bullish Movers (Upward Momentum)",
                "🔴 Top 50 Bearish Shakers (Downward Momentum)",
                "📊 All 100 Combined Table",
            ]
        )

        with sub_tab1:
            st.dataframe(
                style_dark_movers_table(top_bulls, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=600,
            )
        with sub_tab2:
            st.dataframe(
                style_dark_movers_table(top_bears, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=600,
            )
        with sub_tab3:
            st.dataframe(
                style_dark_movers_table(combined_100, use_plain_headers),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=600,
            )
    else:
        st.error("Could not fetch batch price data. Click 'Refresh S&P Movers Now' to retry.")


# ==========================================
# PAGE 2: TOP 20 OPTIONS WHALE SCANNER
# ==========================================
elif page == "🔥 Top 20 Options Whale Scanner":
    st.title("🐋 OpenWhale: Top 20 Hot Stocks Today (Options Whales + Extended-Hours)")
    render_hover_glossary_banner()

    with st.spinner("OpenWhale is scanning CBOE Options Chains & Extended-Hours Tape..."):
        scan_df = run_whale_scanner(DEFAULT_WHALE_UNIVERSE)

    if not scan_df.empty:
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

        longs = (
            scan_df[scan_df["_raw_edge"] >= 0]
            .sort_values("Heat_Score", ascending=False)
            .head(10)
            .drop(columns=["_raw_edge"])
            .rename(columns=whale_col_map)
        )
        shorts = (
            scan_df[scan_df["_raw_edge"] < 0]
            .sort_values("Heat_Score", ascending=False)
            .head(10)
            .drop(columns=["_raw_edge"])
            .rename(columns=whale_col_map)
        )

        c_l, c_s = st.columns(2)
        with c_l:
            st.markdown("### 🟢 Top 10 Hot LONGS (Bullish Whale Flow)")
            st.dataframe(longs, column_config=whale_help_cfg, use_container_width=True, hide_index=True)
        with c_s:
            st.markdown("### 🔴 Top 10 Hot SHORTS (Bearish Whale Flow)")
            st.dataframe(shorts, column_config=whale_help_cfg, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 3: SINGLE-TICKER DEEP DIVE
# ==========================================
else:
    selected_ticker = st.sidebar.text_input(
        "🔍 Enter Stock or Index Symbol",
        value="SPY",
        help="Type any US Stock, ETF (SPY, QQQ), or Cash Index (SPX, VIX, NDX).",
    ).upper().strip()
    num_expiries = st.sidebar.slider(
        "Expirations to Aggregate",
        1,
        8,
        4,
        help="How many upcoming options expiration dates to combine when calculating Price Floors and Ceilings.",
    )

    st.title(f"🐋 OpenWhale Deep-Dive Radar: {selected_ticker}")
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

        call_wall = gex_by_strike.loc[gex_by_strike["call"].idxmax(), "strike"]
        put_wall = gex_by_strike.loc[gex_by_strike["put"].idxmin(), "strike"]

        near_spot = gex_by_strike[
            (gex_by_strike["strike"] >= spot * 0.93) & (gex_by_strike["strike"] <= spot * 1.07)
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

        # KPI Cards with built-in Hover Tooltips (help=...)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Last Traded Price",
            f"${spot}",
            f"{tech['ext_gap_pct']:+.2f}%",
            help=GLOSSARY_TOOLTIPS["Last_Price"],
        )
        m2.metric(
            "Price Floor (Safety Net)" if use_plain_headers else "Put Wall (-GEX)",
            f"${put_wall}",
            help=GLOSSARY_TOOLTIPS["Put_Wall"],
        )
        m3.metric(
            "Price Ceiling (Target)" if use_plain_headers else "Call Wall (+GEX)",
            f"${call_wall}",
            help=GLOSSARY_TOOLTIPS["Call_Wall"],
        )
        m4.metric(
            "Danger Line (Pivot)" if use_plain_headers else "Zero Gamma Flip",
            f"${round(float(zero_gamma), 2)}",
            help=GLOSSARY_TOOLTIPS["Zero_Gamma"],
        )
        m5.metric(
            "Fair Price (VWAP)" if use_plain_headers else "Extended VWAP",
            f"${tech['vwap']}",
            help=GLOSSARY_TOOLTIPS["VWAP"],
        )
        m6.metric(
            "Fever Score (RSI)" if use_plain_headers else "Ext RSI(14)",
            f"{tech['rsi']}",
            help=GLOSSARY_TOOLTIPS["RSI_14"],
        )

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
            fig.add_vline(
                x=round(float(zero_gamma), 2),
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Danger Line: ${round(float(zero_gamma), 2)}",
            )
            fig.update_layout(
                title=f"{selected_ticker} Price Floors (Red) & Price Ceilings (Green) — Hover Bars for Details",
                template="plotly_dark",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if not finra_df.empty:
                fig_dp = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dp.add_trace(
                    go.Bar(
                        x=finra_df["Date"],
                        y=finra_df["Total_Vol"],
                        name="Dark Pool Volume",
                        marker_color="#6366f1",
                        opacity=0.6,
                    ),
                    secondary_y=False,
                )
                fig_dp.add_trace(
                    go.Scatter(
                        x=finra_df["Date"],
                        y=finra_df["Hidden_Buying_Pct"],
                        name="Hidden Buying %",
                        mode="lines+markers",
                        line=dict(color="#10b981", width=3),
                        hovertemplate="<b>Date: %{x}</b><br>Hidden Dark Pool Buying: %{y:.1f}%<br><i>Above 55% = Quiet Institutional Accumulation</i><extra></extra>",
                    ),
                    secondary_y=True,
                )
                fig_dp.add_hline(y=50, line_dash="dot", line_color="#94a3b8", secondary_y=True)
                fig_dp.update_layout(
                    title=f"{selected_ticker} FINRA Dark Pool Buying %",
                    template="plotly_dark",
                    height=420,
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_dp, use_container_width=True)
            else:
                st.info("Dark Pool volume is reported for stocks & ETFs (not cash indices like SPX).")

    except Exception as e:
        st.error(f"Could not load deep-dive data for {selected_ticker}: {e}")
