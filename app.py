from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import re
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OpenWhale | Live Stream, Sweeps, Backtester & Congress",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .star-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px solid #eab308;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# CORE WATCHLIST UNIVERSE
# ==========================================
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "NVDA", "TSLA", "AAPL", "AMD", "MSFT",
    "AMZN", "META", "GOOGL", "NFLX", "AVGO", "PLTR", "COIN", "MSTR", "SOFI", "HOOD",
    "ARM", "SMCI", "MU", "INTC", "ORCL", "CRM", "UBER", "SHOP", "JPM", "GS", "LLY", "XOM"
]

OCC_REGEX = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")


# ==========================================
# 1. LIVE TECHNICAL & EXTENDED-HOURS ENGINE
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


@st.cache_data(ttl=25)
def fetch_live_spot_and_intraday(ticker):
    """
    Pulls 5-minute candles INCLUDING Pre-Market and After-Hours (prepost=True)
    and computes real-time VWAP, EMA(9/21), RSI(14), and Relative Volume.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period="5d", interval="5m", prepost=True)
    daily = tk.history(period="10d", interval="1d", prepost=False)

    if df.empty or daily.empty:
        raise ValueError(f"No live price data returned for {ticker}")

    df = df.dropna(subset=["Close"]).copy()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    df["Date"] = df.index.date
    df["TypVol"] = ((df["High"] + df["Low"] + df["Close"]) / 3.0) * df["Volume"]
    daily_cum_typvol = df.groupby("Date")["TypVol"].cumsum()
    daily_cum_vol = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = np.where(daily_cum_vol > 0, daily_cum_typvol / daily_cum_vol, df["Close"])

    spot = float(df["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(daily["Close"].iloc[-1])
    change_pct = ((spot - prev_close) / prev_close) * 100.0
    vwap = float(df["VWAP"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1]) if not np.isnan(df["RSI"].iloc[-1]) else 50.0
    ema9 = float(df["EMA9"].iloc[-1])
    ema21 = float(df["EMA21"].iloc[-1])

    avg_vol = float(daily["Volume"].iloc[:-1].mean()) if len(daily) > 1 else 1.0
    today_vol = float(daily["Volume"].iloc[-1])
    rvol = round(today_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    last_bar_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")

    return {
        "ticker": ticker,
        "spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "change_pct": round(change_pct, 2),
        "vwap": round(vwap, 2),
        "rsi": round(rsi, 1),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "rvol": rvol,
        "last_bar_time": last_bar_time,
        "intraday_df": df,
        "daily_df": daily,
    }


# ==========================================
# 2. LIVE CBOE OPTIONS FLOW & SWEEP ENGINE
# ==========================================
def calc_bs_gamma(S, K, T, r, sigma):
    T = np.maximum(T, 4 / (365 * 24))
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


@st.cache_data(ttl=30)
def fetch_live_options_and_sweeps(ticker_symbol, spot_override=None, num_expiries=4):
    """
    Pulls live options chain from CBOE CDN, calculates dealer GEX walls,
    and isolates Institutional Sweeps & Block Orders with DTE timeframes.
    """
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
    quote_timestamp = payload.get("timestamp") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    spot = float(spot_override or payload.get("current_price") or payload.get("close") or 0)
    raw_options = payload.get("options", [])
    if not raw_options or spot <= 0:
        raise ValueError(f"No options data for {ticker_symbol}")

    today_date = dt.date.today()
    today_str = today_date.strftime("%Y-%m-%d")
    rows = []

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
        if strike < spot * 0.82 or strike > spot * 1.18:
            continue

        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today_date).days

        last_p = float(opt.get("last_trade_price") or 0.0)
        bid = float(opt.get("bid") or 0.0)
        ask = float(opt.get("ask") or 0.0)
        mark = last_p if last_p > 0 else ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)
        vol = float(opt.get("volume") or 0.0)
        oi = float(opt.get("open_interest") or 0.0)
        last_trade_time = opt.get("last_trade_time") or quote_timestamp

        if ask > 0 and last_p >= ask * 0.98:
            exec_side = "🔥 Ask-Side (Aggressive Buy)"
        elif bid > 0 and last_p <= bid * 1.02:
            exec_side = "❄️ Bid-Side (Sell / Write)"
        else:
            exec_side = "⚖️ Mid-Market"

        if dte == 0:
            horizon = "⚡ 0DTE Intraday Scalp"
        elif dte <= 5:
            horizon = f"🏃 Weekly Swing ({dte}d)"
        elif dte <= 21:
            horizon = f"📅 1–3 Week Swing ({dte}d)"
        else:
            horizon = f"🏛️ Macro Position ({dte}d)"

        rows.append(
            {
                "Ticker": clean_sym,
                "Contract": sym,
                "Type": "🟢 CALL" if cp == "C" else "🔴 PUT",
                "raw_type": "call" if cp == "C" else "put",
                "Strike": strike,
                "Expiration": exp_str,
                "DTE": dte,
                "Timeframe_Horizon": horizon,
                "Last_Price": round(mark, 2),
                "Bid": round(bid, 2),
                "Ask": round(ask, 2),
                "Execution_Side": exec_side,
                "Volume": int(vol),
                "Open_Interest": int(oi),
                "Vol_OI_Ratio": round(vol / oi, 2) if oi > 0 else round(vol, 2),
                "Premium_USD": round(vol * mark * 100.0, 0),
                "IV": round(float(opt.get("iv") or 0.0) * 100, 1),
                "cboe_gamma": float(opt.get("gamma") or 0.0),
                "Trade_Time": str(last_trade_time)[-8:],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid near-the-money options found.")

    nearest_exps = sorted(df["Expiration"].unique())[:num_expiries]
    gex_slice = df[df["Expiration"].isin(nearest_exps)].copy()

    T_years = np.maximum(gex_slice["DTE"].values / 365.0, 4 / (365 * 24))
    bs_gamma = calc_bs_gamma(
        spot, gex_slice["Strike"].values, T_years, 0.042, np.maximum(gex_slice["IV"].values / 100.0, 0.05)
    )
    gex_slice["gamma"] = np.where(gex_slice["cboe_gamma"] > 0, gex_slice["cboe_gamma"], bs_gamma)
    sign = np.where(gex_slice["raw_type"] == "call", 1.0, -1.0)
    gex_slice["gex_dollars"] = gex_slice["gamma"] * gex_slice["Open_Interest"] * 100 * (spot**2) * 0.01 * sign

    gex_by_strike = (
        gex_slice.groupby(["Strike", "raw_type"])["gex_dollars"].sum().unstack(fill_value=0).reset_index()
    )
    for c in ["call", "put"]:
        if c not in gex_by_strike:
            gex_by_strike[c] = 0.0
    gex_by_strike["net_gex"] = gex_by_strike["call"] + gex_by_strike["put"]

    above_spot = gex_by_strike[gex_by_strike["Strike"] >= spot]
    below_spot = gex_by_strike[gex_by_strike["Strike"] <= spot]

    call_wall = (
        float(above_spot.loc[above_spot["call"].idxmax(), "Strike"])
        if not above_spot.empty
        else float(gex_by_strike.loc[gex_by_strike["call"].idxmax(), "Strike"])
    )
    put_wall = (
        float(below_spot.loc[below_spot["put"].idxmin(), "Strike"])
        if not below_spot.empty
        else float(gex_by_strike.loc[gex_by_strike["put"].idxmin(), "Strike"])
    )

    near_spot = gex_by_strike[
        (gex_by_strike["Strike"] >= spot * 0.94) & (gex_by_strike["Strike"] <= spot * 1.06)
    ].sort_values("Strike")
    zero_gamma = spot
    signs = np.sign(near_spot["net_gex"].values)
    strikes = near_spot["Strike"].values
    net_vals = near_spot["net_gex"].values
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and (net_vals[i + 1] - net_vals[i]) != 0:
            zero_gamma = strikes[i] - net_vals[i] * (strikes[i + 1] - strikes[i]) / (net_vals[i + 1] - net_vals[i])
            break

    call_prem = df[df["raw_type"] == "call"]["Premium_USD"].sum()
    put_prem = df[df["raw_type"] == "put"]["Premium_USD"].sum()
    call_pct = (call_prem / (call_prem + put_prem) * 100.0) if (call_prem + put_prem) > 0 else 50.0

    sweeps_df = (
        df[(df["Volume"] >= 150) & (df["Vol_OI_Ratio"] >= 1.25) & (df["Premium_USD"] >= 50000)]
        .sort_values("Premium_USD", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(float(zero_gamma), 2),
        "call_pct": round(call_pct, 1),
        "call_prem_m": round(call_prem / 1e6, 2),
        "put_prem_m": round(put_prem / 1e6, 2),
        "gex_by_strike": gex_by_strike,
        "sweeps_df": sweeps_df,
        "all_options_df": df,
    }


# ==========================================
# 3. 5-STAR RATING & TRADE TICKET CALCULATOR
# ==========================================
def evaluate_star_rating(spot_info, opt_info):
    spot = spot_info["spot"]
    vwap = spot_info["vwap"]
    rsi = spot_info["rsi"]
    ema_bull = spot_info["ema9"] >= spot_info["ema21"]

    call_wall = opt_info["call_wall"]
    put_wall = opt_info["put_wall"]
    zero_gamma = opt_info["zero_gamma"]
    call_pct = opt_info["call_pct"]
    sweeps = opt_info["sweeps_df"]

    call_sweeps = len(sweeps[sweeps["raw_type"] == "call"]) if not sweeps.empty else 0
    put_sweeps = len(sweeps[sweeps["raw_type"] == "put"]) if not sweeps.empty else 0

    is_bullish = (spot >= vwap and call_pct >= 50.0) or (call_pct >= 60.0)

    stars = 0
    checklist = []

    if is_bullish:
        bias = "LONG"
        if spot >= vwap and ema_bull:
            stars += 1
            checklist.append(f"⭐ Trend Aligned: Spot (${spot}) > VWAP (${vwap}) & EMA9 > EMA21")
        else:
            checklist.append(f"⚪ Trend Mixed: Spot (${spot}) vs VWAP (${vwap})")

        if spot >= zero_gamma:
            stars += 1
            checklist.append(f"⭐ Dealer Regime: Above Zero Gamma (${zero_gamma})")
        else:
            checklist.append(f"⚪ Dealer Regime: Below Zero Gamma (${zero_gamma})")

        if call_pct >= 58.0:
            stars += 1
            checklist.append(f"⭐ Options Flow: {call_pct:.1f}% Bullish Call Premium")
        else:
            checklist.append(f"⚪ Options Flow: {call_pct:.1f}% Call Premium (<58%)")

        if call_sweeps >= 2:
            stars += 1
            checklist.append(f"⭐ Whale Sweeps: {call_sweeps} Large Call Sweeps Active")
        else:
            checklist.append(f"⚪ Whale Sweeps: Only {call_sweeps} Call Sweeps")

        target = call_wall if call_wall > spot * 1.005 else round(spot * 1.02, 2)
        stop = round(max(put_wall, min(vwap, zero_gamma) * 0.994, spot * 0.988), 2)
        if stop >= spot:
            stop = round(spot * 0.99, 2)
        rr = round((target - spot) / max(spot - stop, 0.01), 2)

        if 32.0 <= rsi <= 68.0 and rr >= 1.5:
            stars += 1
            checklist.append(f"⭐ Risk/Reward & RSI: R:R is {rr}:1 with Healthy RSI ({rsi})")
        else:
            checklist.append(f"⚪ Risk/Reward or RSI: R:R {rr}:1 | RSI {rsi}")

    else:
        bias = "SHORT"
        put_pct = 100.0 - call_pct

        if spot < vwap and not ema_bull:
            stars += 1
            checklist.append(f"⭐ Trend Aligned: Spot (${spot}) < VWAP (${vwap}) & EMA9 < EMA21")
        else:
            checklist.append(f"⚪ Trend Mixed: Spot (${spot}) vs VWAP (${vwap})")

        if spot < zero_gamma:
            stars += 1
            checklist.append(f"⭐ Dealer Regime: Below Zero Gamma (${zero_gamma}) — Fast Drop Zone")
        else:
            checklist.append(f"⚪ Dealer Regime: Above Zero Gamma (${zero_gamma})")

        if put_pct >= 58.0:
            stars += 1
            checklist.append(f"⭐ Options Flow: {put_pct:.1f}% Bearish Put Premium")
        else:
            checklist.append(f"⚪ Options Flow: {put_pct:.1f}% Put Premium (<58%)")

        if put_sweeps >= 2:
            stars += 1
            checklist.append(f"⭐ Whale Sweeps: {put_sweeps} Large Put Sweeps Active")
        else:
            checklist.append(f"⚪ Whale Sweeps: Only {put_sweeps} Put Sweeps")

        target = put_wall if put_wall < spot * 0.995 else round(spot * 0.98, 2)
        stop = round(min(call_wall, max(vwap, zero_gamma) * 1.006, spot * 1.012), 2)
        if stop <= spot:
            stop = round(spot * 1.01, 2)
        rr = round((spot - target) / max(stop - spot, 0.01), 2)

        if 32.0 <= rsi <= 68.0 and rr >= 1.5:
            stars += 1
            checklist.append(f"⭐ Risk/Reward & RSI: R:R is {rr}:1 with Healthy RSI ({rsi})")
        else:
            checklist.append(f"⚪ Risk/Reward or RSI: R:R {rr}:1 | RSI {rsi}")

    star_str = "⭐" * stars + "☆" * (5 - stars)
    label = f"{star_str} ({stars}/5) — {'🟢 ACTIONABLE LONG' if (stars >= 4 and bias == 'LONG') else ('🔴 ACTIONABLE SHORT' if (stars >= 4 and bias == 'SHORT') else '🟡 WAIT FOR 4–5 STARS')}"

    return {
        "ticker": spot_info["ticker"],
        "stars": stars,
        "star_str": star_str,
        "bias": bias,
        "verdict_label": label,
        "spot": spot,
        "vwap": vwap,
        "zero_gamma": zero_gamma,
        "entry": round(spot, 2),
        "stop": stop,
        "target": target,
        "rr": rr,
        "rsi": rsi,
        "call_pct": call_pct,
        "call_sweeps": call_sweeps,
        "put_sweeps": put_sweeps,
        "checklist": checklist,
        "Expected_Timeframe": "1–4 Hours (Intraday) to 2 Days",
    }


# ==========================================
# 4. REAL HISTORICAL 5-STAR BACKTEST ENGINE
# ==========================================
@st.cache_data(ttl=600)
def run_real_5star_backtest(tickers, lookback_days="30d"):
    raw = yf.download(
        tickers, period=lookback_days, interval="15m", prepost=False, group_by="ticker", threads=True, progress=False
    )

    trades = []
    for sym in tickers:
        try:
            df = raw[sym].dropna(subset=["Close", "High", "Low", "Volume"]).copy() if sym in raw else pd.DataFrame()
            if len(df) < 60:
                continue

            df["Date"] = df.index.date
            df["TypVol"] = ((df["High"] + df["Low"] + df["Close"]) / 3.0) * df["Volume"]
            df["VWAP"] = df.groupby("Date")["TypVol"].cumsum() / np.maximum(df.groupby("Date")["Volume"].cumsum(), 1)
            df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["RSI"] = compute_rsi(df["Close"], 14)
            df["Vol_MA20"] = df["Volume"].rolling(20).mean()
            df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

            for day_date, day_df in df.groupby("Date"):
                if len(day_df) < 12:
                    continue
                for idx in range(2, min(16, len(day_df) - 4)):
                    row = day_df.iloc[idx]
                    prev_row = day_df.iloc[idx - 1]
                    if pd.isna(row["ATR"]) or pd.isna(row["Vol_MA20"]) or row["ATR"] <= 0:
                        continue

                    vol_surge = row["Volume"] / max(row["Vol_MA20"], 1)
                    atr = float(row["ATR"])
                    entry_price = float(row["Close"])
                    entry_time = day_df.index[idx]

                    is_5star_long = (
                        entry_price > row["VWAP"]
                        and prev_row["Close"] <= prev_row["VWAP"] * 1.003
                        and row["EMA9"] > row["EMA21"]
                        and 45.0 <= row["RSI"] <= 67.0
                        and vol_surge >= 1.8
                    )

                    is_5star_short = (
                        entry_price < row["VWAP"]
                        and prev_row["Close"] >= prev_row["VWAP"] * 0.997
                        and row["EMA9"] < row["EMA21"]
                        and 33.0 <= row["RSI"] <= 55.0
                        and vol_surge >= 1.8
                    )

                    if not (is_5star_long or is_5star_short):
                        continue

                    setup_type = "🟢 5-STAR LONG" if is_5star_long else "🔴 5-STAR SHORT"
                    stop_dist = max(atr * 1.2, entry_price * 0.004)
                    target_dist = stop_dist * 2.0

                    stop_price = entry_price - stop_dist if is_5star_long else entry_price + stop_dist
                    target_price = entry_price + target_dist if is_5star_long else entry_price - target_dist

                    future_bars = day_df.iloc[idx + 1 :]
                    exit_price = float(future_bars["Close"].iloc[-1])
                    exit_time = future_bars.index[-1]
                    outcome = "⏱️ EOD Time Exit"

                    for f_time, f_bar in future_bars.iterrows():
                        if is_5star_long:
                            if float(f_bar["Low"]) <= stop_price:
                                exit_price = stop_price
                                exit_time = f_time
                                outcome = "🛑 Stop Hit (-1.0R)"
                                break
                            if float(f_bar["High"]) >= target_price:
                                exit_price = target_price
                                exit_time = f_time
                                outcome = "✅ Target Hit (+2.0R)"
                                break
                        else:
                            if float(f_bar["High"]) >= stop_price:
                                exit_price = stop_price
                                exit_time = f_time
                                outcome = "🛑 Stop Hit (-1.0R)"
                                break
                            if float(f_bar["Low"]) <= target_price:
                                exit_price = target_price
                                exit_time = f_time
                                outcome = "✅ Target Hit (+2.0R)"
                                break

                    pnl_pct = (
                        ((exit_price - entry_price) / entry_price) * 100.0
                        if is_5star_long
                        else ((entry_price - exit_price) / entry_price) * 100.0
                    )
                    r_multiple = (
                        (exit_price - entry_price) / stop_dist
                        if is_5star_long
                        else (entry_price - exit_price) / stop_dist
                    )
                    hold_mins = int((exit_time - entry_time).total_seconds() / 60)

                    trades.append(
                        {
                            "Date": str(day_date),
                            "Entry_Time": entry_time.strftime("%H:%M"),
                            "Exit_Time": exit_time.strftime("%H:%M"),
                            "Hold_Time": f"{hold_mins} mins",
                            "Ticker": sym,
                            "Setup": setup_type,
                            "Vol_Surge": f"{vol_surge:.1f}x",
                            "Entry_Price": round(entry_price, 2),
                            "Stop_Loss": round(stop_price, 2),
                            "Target_Price": round(target_price, 2),
                            "Exit_Price": round(exit_price, 2),
                            "Outcome": outcome,
                            "Return_Pct": round(pnl_pct, 2),
                            "R_Multiple": round(r_multiple, 2),
                        }
                    )
                    break
        except Exception:
            continue

    bt_df = pd.DataFrame(trades)
    if not bt_df.empty:
        bt_df = bt_df.sort_values(["Date", "Entry_Time"], ascending=[False, False]).reset_index(drop=True)
    return bt_df


# ==========================================
# 5. CONGRESS (STOCK ACT) & SEC INSIDER SCRAPER
# ==========================================
@st.cache_data(ttl=1800)
def fetch_congress_trades(ticker_filter=None):
    """
    Pulls public U.S. House & Senate STOCK Act disclosures from free open-source
    JSON mirrors (House/Senate Stock Watcher & Capitol Trades public feeds).
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    urls = [
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    ]

    records = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                data = r.json()[:400]
                for item in data:
                    tkr = str(item.get("ticker", "")).upper().strip().replace("--", "")
                    if not tkr or tkr == "N/A":
                        continue
                    records.append(
                        {
                            "Filing_Date": item.get("disclosure_date") or item.get("transaction_date", ""),
                            "Trade_Date": item.get("transaction_date", ""),
                            "Politician": item.get("representative") or item.get("senator", "Member of Congress"),
                            "Chamber": "House" if "representative" in item else "Senate",
                            "Ticker": tkr,
                            "Action": item.get("type", "Purchase"),
                            "Amount_Range": item.get("amount", "$1,001 - $15,000"),
                            "Asset": str(item.get("asset_description", ""))[:45],
                        }
                    )
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Filing_Date", ascending=False).reset_index(drop=True)
        if ticker_filter:
            df_filtered = df[df["Ticker"] == ticker_filter.upper().strip()]
            if not df_filtered.empty:
                return df_filtered.head(40)
    return df.head(50)


@st.cache_data(ttl=1800)
def fetch_sec_insider_trades(ticker_filter=None):
    """
    Scrapes real-time SEC Form 4 Corporate Insider Purchases & Sales (CEOs, CFOs,
    Directors, 10% Owners) directly from OpenInsider's public tables.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if ticker_filter:
        url = (
            f"http://openinsider.com/screener?s={ticker_filter.upper().strip()}"
            "&o=&pl=&ph=&ll=&lh=&fd=180&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago="
            "&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh="
            "&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=40&page=1"
        )
    else:
        url = (
            "http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0"
            "&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=100&vh=&ocl=&och=&sic1=-1&sicl=100"
            "&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=50&page=1"
        )

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(resp.text)
        for tbl in tables:
            cols = [str(c).replace("\xa0", " ").strip() for c in tbl.columns]
            if "Filing Date" in cols and "Trade Type" in cols:
                tbl.columns = cols
                keep = [
                    c
                    for c in [
                        "Filing Date", "Trade Date", "Ticker", "Insider Name",
                        "Title", "Trade Type", "Price", "Qty", "Owned", "ΔOwn", "Value",
                    ]
                    if c in tbl.columns
                ]
                return tbl[keep].head(40)
    except Exception:
        pass
    return pd.DataFrame()


# ==========================================
# 6. SIDEBAR & LIVE STREAM CONTROLS
# ==========================================
st.sidebar.title("🐋 OpenWhale Live")
st.sidebar.caption("Streamlined Spot, Options Sweeps, 5-Star Backtester & Congress")

active_ticker = st.sidebar.selectbox("🎯 Select Primary Ticker to Stream:", options=DEFAULT_WATCHLIST, index=0)
custom_sym = st.sidebar.text_input("Or Type Custom Symbol (e.g. SPX, PLTR):", value="").upper().strip()
if custom_sym:
    active_ticker = custom_sym

live_stream_on = st.sidebar.toggle("⚡ Enable Live 30s Auto-Stream", value=False)
min_sweep_usd = st.sidebar.selectbox("Minimum Institutional Sweep Size ($):", [25000, 50000, 100000, 250000], index=1)

st.title("🐋 OpenWhale: Live Spot, Institutional Sweeps, 5-Star Backtester & Congress")

try:
    spot_info = fetch_live_spot_and_intraday(active_ticker)
    opt_info = fetch_live_options_and_sweeps(active_ticker, spot_override=spot_info["spot"])
    star_eval = evaluate_star_rating(spot_info, opt_info)

    # Top Live Spot & Star Rating Banner
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"📡 {active_ticker} Live Spot", f"${spot_info['spot']:,.2f}", f"{spot_info['change_pct']:+.2f}%")
    k2.metric(
        "Fair Price (VWAP)",
        f"${spot_info['vwap']:,.2f}",
        "Above VWAP" if spot_info["spot"] >= spot_info["vwap"] else "Below VWAP",
    )
    k3.metric("Price Floor (Put Wall)", f"${opt_info['put_wall']:,.2f}", "Dealer Support")
    k4.metric("Price Ceiling (Call Wall)", f"${opt_info['call_wall']:,.2f}", "Dealer Target")
    k5.metric(
        "Smart Money Call %",
        f"{opt_info['call_pct']:.1f}%",
        f"${opt_info['call_prem_m']}M C / ${opt_info['put_prem_m']}M P",
    )
    k6.metric("Setup Rating", star_eval["star_str"], f"R:R {star_eval['rr']} : 1")

    # Main 4-Tab Interface
    tab_stream, tab_scanner, tab_backtest, tab_congress = st.tabs(
        [
            f"📡 1. Live Spot & Institutional Sweeps ({active_ticker})",
            "⭐ 2. Live 5-Star Trade Finder (Multi-Ticker)",
            "📊 3. Historical 5-Star Backtest Engine",
            "🏛️ 4. Congress & Corporate Insider Filings",
        ]
    )

    # ==========================================
    # TAB 1: LIVE STREAM & INSTITUTIONAL SWEEPS
    # ==========================================
    with tab_stream:
        checks_html = " &nbsp;|&nbsp; ".join(star_eval["checklist"])
        st.markdown(
            f"""
            <div class="star-card">
                <h3 style="margin:0; color:#f8fafc;">{star_eval['verdict_label']}</h3>
                <p style="margin:6px 0; color:#eab308; font-weight:600;">
                    🎯 Entry: ${star_eval['entry']} &nbsp;&nbsp;|&nbsp;&nbsp;
                    🛑 Stop-Loss: ${star_eval['stop']} &nbsp;&nbsp;|&nbsp;&nbsp;
                    💰 Wall Target: ${star_eval['target']} &nbsp;&nbsp;|&nbsp;&nbsp;
                    ⚖️ Reward:Risk: {star_eval['rr']} : 1 &nbsp;&nbsp;|&nbsp;&nbsp;
                    ⏱️ Timeframe: {star_eval['Expected_Timeframe']}
                </p>
                <p style="margin:0; font-size:0.84rem; color:#cbd5e1;">{checks_html}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_chart, col_sweeps = st.columns([0.48, 0.52])

        with col_chart:
            st.subheader(f"📈 {active_ticker} Live 5m Stream + VWAP & Dealer Walls")
            intra_df = spot_info["intraday_df"].tail(75)

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=intra_df.index,
                    open=intra_df["Open"],
                    high=intra_df["High"],
                    low=intra_df["Low"],
                    close=intra_df["Close"],
                    name="Spot Price",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=intra_df.index,
                    y=intra_df["VWAP"],
                    mode="lines",
                    name="Fair Price (VWAP)",
                    line=dict(color="#38bdf8", width=2),
                )
            )
            fig.add_hline(
                y=opt_info["call_wall"],
                line_dash="dash",
                line_color="#22c55e",
                annotation_text=f"Ceiling (Call Wall): ${opt_info['call_wall']}",
            )
            fig.add_hline(
                y=opt_info["put_wall"],
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Floor (Put Wall): ${opt_info['put_wall']}",
            )
            fig.add_hline(
                y=opt_info["zero_gamma"],
                line_dash="dot",
                line_color="#eab308",
                annotation_text=f"Danger Line: ${opt_info['zero_gamma']}",
            )
            fig.update_layout(
                template="plotly_dark",
                height=470,
                xaxis_rangeslider_visible=False,
                margin=dict(t=20, l=10, r=10, b=10),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sweeps:
            st.subheader(f"🚨 Live Institutional Sweep Orders & Timeframes ({active_ticker})")
            st.caption(
                "Real CBOE options contracts where today's volume exceeds Open Interest (`Vol/OI >= 1.25x`), grouped by expiration timeframe (`0DTE`, `Weekly`, `Swing`, `Macro`)."
            )
            sweeps = opt_info["sweeps_df"]
            filtered_sweeps = (
                sweeps[sweeps["Premium_USD"] >= min_sweep_usd].copy() if not sweeps.empty else pd.DataFrame()
            )

            if not filtered_sweeps.empty:
                filtered_sweeps["Premium ($)"] = filtered_sweeps["Premium_USD"].apply(lambda x: f"${x:,.0f}")
                filtered_sweeps["Strike ($)"] = filtered_sweeps["Strike"].apply(lambda x: f"${x:g}")
                filtered_sweeps["Urgency (Vol/OI)"] = filtered_sweeps["Vol_OI_Ratio"].apply(lambda x: f"{x:.2f}x")

                sweep_cols = [
                    "Trade_Time",
                    "Type",
                    "Strike ($)",
                    "Expiration",
                    "Timeframe_Horizon",
                    "Premium ($)",
                    "Execution_Side",
                    "Urgency (Vol/OI)",
                    "Volume",
                    "Open_Interest",
                ]
                st.dataframe(filtered_sweeps[sweep_cols].head(20), use_container_width=True, hide_index=True, height=430)
            else:
                st.info(f"No sweeps above ${min_sweep_usd:,.0f} right now—lower the minimum sweep filter in the sidebar.")

    # ==========================================
    # TAB 2: LIVE 5-STAR TRADE FINDER (MULTI-TICKER)
    # ==========================================
    with tab_scanner:
        st.subheader("⭐ Live 5-Star Trade Scanner (Filters Only Top Actionable Setups)")
        st.caption("Scans the top liquid tickers in parallel and ranks them from 5 Stars (`⭐⭐⭐⭐⭐`) down.")

        if st.button("🚀 Scan Watchlist for 5-Star Trades Now", type="primary"):
            st.cache_data.clear()

        def scan_star_worker(sym):
            try:
                s_info = fetch_live_spot_and_intraday(sym)
                o_info = fetch_live_options_and_sweeps(sym, spot_override=s_info["spot"])
                return evaluate_star_rating(s_info, o_info)
            except Exception:
                return None

        with st.spinner("Evaluating 5-Star Confluence across liquid market leaders..."):
            star_rows = []
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(scan_star_worker, sym): sym for sym in DEFAULT_WATCHLIST[:18]}
                for f in as_completed(futs):
                    r = f.result()
                    if r is not None:
                        star_rows.append(r)

        if star_rows:
            star_df = (
                pd.DataFrame(star_rows).sort_values(["stars", "rr"], ascending=[False, False]).reset_index(drop=True)
            )
            only_top = st.checkbox("Show Only 4-Star & 5-Star Actionable Trades", value=False)
            if only_top:
                star_df = star_df[star_df["stars"] >= 4]

            show_cols = {
                "ticker": "Ticker",
                "star_str": "Star Rating",
                "verdict_label": "Action Verdict",
                "spot": "Live Spot ($)",
                "entry": "Entry ($)",
                "stop": "Stop-Loss ($)",
                "target": "Wall Target ($)",
                "rr": "Reward:Risk",
                "call_pct": "Call Flow %",
                "call_sweeps": "Call Sweeps",
                "put_sweeps": "Put Sweeps",
                "Expected_Timeframe": "Trade Timeframe",
            }
            st.dataframe(
                star_df[list(show_cols.keys())].rename(columns=show_cols),
                use_container_width=True,
                hide_index=True,
                height=500,
            )

    # ==========================================
    # TAB 3: REAL HISTORICAL 5-STAR BACKTESTER
    # ==========================================
    with tab_backtest:
        st.subheader("📊 Real Historical Backtest: How Top 5-Star Setups Performed (Last 30 Days)")
        st.markdown(
            """
            This backtester pulls **real 15-minute historical market candles over the past 30 days** across your watchlist and tests the exact **5-Star Confluence Rules**:
            * **Entry Trigger:** Price crosses/holds **VWAP** + **EMA(9) > EMA(21)** + **Institutional Volume Surge >= 1.8x normal** + **Healthy RSI (45–67 Long / 33–55 Short)**.
            * **Risk & Exit Rules:** Hard Stop-Loss at `1.2x ATR` (`-1.0R`), Profit Target at `2.4x ATR` (`+2.0R`, i.e., **2.0 : 1 Reward-to-Risk**), or End-of-Day (`15:45 ET`) time exit if neither is hit.
            """
        )

        bt_df = run_real_5star_backtest(DEFAULT_WATCHLIST[:16], lookback_days="30d")

        if not bt_df.empty:
            total_trades = len(bt_df)
            wins = len(bt_df[bt_df["Return_Pct"] > 0])
            win_rate = (wins / total_trades) * 100.0
            avg_r = bt_df["R_Multiple"].mean()
            cum_return_pct = bt_df["Return_Pct"].sum()

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total 5-Star Trades (30d)", f"{total_trades} Trades", "Strict 5-Star Filter")
            b2.metric("Historical Win Rate", f"{win_rate:.1f}%", f"{wins} Wins / {total_trades - wins} Losses")
            b3.metric("Avg R-Multiple per Trade", f"{avg_r:+.2f} R", "2.0 : 1 Target Cap")
            b4.metric("Cumulative Unleveraged Return", f"{cum_return_pct:+.2f}%", "Sum of Underlying Moves")

            bt_chron = bt_df.iloc[::-1].copy().reset_index(drop=True)
            bt_chron["Cumulative_R"] = bt_chron["R_Multiple"].cumsum()
            bt_chron["Trade_Num"] = bt_chron.index + 1

            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=bt_chron["Trade_Num"],
                    y=bt_chron["Cumulative_R"],
                    mode="lines+markers",
                    name="Cumulative R-Multiple",
                    line=dict(color="#22c55e", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(34, 197, 94, 0.15)",
                )
            )
            fig_bt.update_layout(
                title="5-Star Strategy Historical Equity Curve (Cumulative R-Multiples Over Last 30 Days)",
                xaxis_title="Trade Number (Chronological)",
                yaxis_title="Cumulative Profit (in R-Units)",
                template="plotly_dark",
                height=340,
                margin=dict(t=35, l=10, r=10, b=10),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown("#### 📜 Verified 30-Day Trade-by-Trade Execution Log (5-Star Signals Only)")
            st.dataframe(bt_df, use_container_width=True, hide_index=True, height=450)
        else:
            st.warning("No historical 15m data returned for backtest window.")

    # ==========================================
    # TAB 4: CONGRESS & CORPORATE INSIDER TRACKER
    # ==========================================
    with tab_congress:
        st.subheader("🏛️ U.S. Congress (STOCK Act) & SEC Form 4 Corporate Insider Tracker")
        st.caption(
            "Tracks disclosed stock purchases/sales by U.S. Senators & House Representatives alongside real-time SEC Form 4 C-Suite Insider buying."
        )

        filter_by_active = st.checkbox(
            f"🔍 Filter Filings Strictly for {active_ticker} (Uncheck to see All Market-Wide Filings)",
            value=False,
        )
        target_sym = active_ticker if filter_by_active else None

        col_cong, col_ins = st.columns(2)

        with col_cong:
            st.markdown("#### 🏛️ Latest U.S. Congress Trades (STOCK Act Disclosures)")
            cong_df = fetch_congress_trades(target_sym)
            if not cong_df.empty:
                st.dataframe(cong_df, use_container_width=True, hide_index=True, height=450)
            else:
                st.info("No recent Congressional filings matched this filter.")

        with col_ins:
            st.markdown("#### 👔 Latest SEC Form 4 Corporate Insider Buys (CEOs / CFOs / Directors)")
            ins_df = fetch_sec_insider_trades(target_sym)
            if not ins_df.empty:
                st.dataframe(ins_df, use_container_width=True, hide_index=True, height=450)
            else:
                st.info("No recent SEC Form 4 insider filings found for this filter.")

except Exception as e:
    st.error(f"Error loading live feed for {active_ticker}: {e}")

if live_stream_on:
    time.sleep(30)
    st.rerun()
