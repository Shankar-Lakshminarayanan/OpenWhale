import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import os
import re
import sqlite3
import threading
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OpenWhale | Multi-Ticker Live Stream, $100k+ Sweeps & 2026 Backtester",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .ow-glossary-bar {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 14px;
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
        font-size: 0.82rem;
        font-weight: 600;
        cursor: help;
        border-bottom: 1px dashed #38bdf8;
    }
    .ow-pill .ow-tip {
        visibility: hidden;
        width: 280px;
        background-color: #0f172a;
        color: #f8fafc;
        text-align: left;
        border-radius: 8px;
        padding: 10px 12px;
        position: absolute;
        z-index: 9999;
        top: 130%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.2s;
        border: 1px solid #38bdf8;
        box-shadow: 0 10px 25px rgba(0,0,0,0.75);
        font-weight: 400;
        font-size: 0.79rem;
        line-height: 1.4;
    }
    .ow-pill:hover .ow-tip {
        visibility: visible;
        opacity: 1;
    }
    .star-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px solid #eab308;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hover_glossary_banner():
    st.markdown(
        """
        <div class="ow-glossary-bar">
            <span style="color:#94a3b8; font-size:0.84rem; font-weight:600;">💡 Hover any term for Plain-English meaning:</span>
            <div class="ow-pill">Price Floor (Put Wall)<span class="ow-tip"><b>Price Floor / Put Wall:</b> Strike where dealers mechanically buy shares to hedge options. Acts like a trampoline safety net.</span></div>
            <div class="ow-pill">Price Ceiling (Call Wall)<span class="ow-tip"><b>Price Ceiling / Call Wall:</b> Strike where dealers mechanically sell shares to hedge options. Take profits slightly below here.</span></div>
            <div class="ow-pill">Danger Line (Zero Gamma)<span class="ow-tip"><b>Danger Line (Flip Level):</b> Above = calm dip-buying market. Below = fast, volatile swings.</span></div>
            <div class="ow-pill">Fair Price (VWAP)<span class="ow-tip"><b>Fair Value Price (VWAP):</b> Volume-weighted average price paid today. Above = Buyers winning.</span></div>
            <div class="ow-pill">Urgency (Vol/OI)<span class="ow-tip"><b>Sweep Urgency (Vol/OI):</b> Today's contract volume divided by existing Open Interest. Higher numbers mean aggressive brand-new institutional positioning.</span></div>
            <div class="ow-pill">$100 Trade Accumulated Profit<span class="ow-tip"><b>$100 Per-Trade Sizing:</b> Every time a 5-Star setup triggers, you invest $100. The profit or loss from each $100 trade is added to your Running Accumulated Profit total.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 1. UNION STOCK UNIVERSE (TOP 50 MCAP + MAG 7 + ETFS + FLOW + BUZZ + CONGRESS)
# ==========================================
TOP_50_MARKET_CAP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "LLY",
    "WMT", "JPM", "V", "XOM", "MA", "ORCL", "UNH", "COST", "PG", "HD",
    "JNJ", "NFLX", "ABBV", "BAC", "CRM", "KO", "CVX", "MRK", "AMD", "PEP",
    "TMO", "CSCO", "LIN", "WFC", "ADBE", "ACN", "MCD", "ABT", "IBM", "GE",
    "CAT", "PM", "TXN", "ISRG", "QCOM", "INTU", "VZ", "AMGN", "NOW", "GS"
]

POPULAR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "XLV", "XLI",
    "XLY", "XLP", "XLU", "ARKK", "GLD", "SLV", "TLT", "HYG", "IBIT", "EEM",
    "SOXL", "TQQQ", "SQQQ", "SPXL", "XBI", "KRE", "GDX"
]

OPTIONS_FLOW_LEADERS = [
    "PLTR", "COIN", "MSTR", "SMCI", "ARM", "MU", "INTC", "SNOW", "UBER", "SHOP",
    "ABNB", "BA", "DIS", "PYPL", "SQ", "PFE", "MRNA", "PANW", "CRWD", "DELL"
]

SOCIAL_MEDIA_BUZZ = [
    "RDDT", "GME", "AMC", "SOFI", "HOOD", "RIVN", "LCID", "DJT", "MARA", "RIOT",
    "CLSK", "IONQ", "RKLB", "ASTS", "LUNR", "HIMS", "CVNA", "UPST", "AFRM", "CELH"
]

OCC_REGEX = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")
MIN_SWEEP_USD = 100000


# ==========================================
# 2. CONGRESS & INSIDER SCRAPER
# ==========================================
@st.cache_data(ttl=1800)
def fetch_congress_trades(ticker_filter=None):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    urls = [
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    ]
    records = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=7)
            if r.status_code == 200:
                for item in r.json()[:350]:
                    tkr = str(item.get("ticker", "")).upper().strip().replace("--", "")
                    if not tkr or tkr == "N/A" or len(tkr) > 5 or not tkr.isalpha():
                        continue
                    records.append(
                        {
                            "Filing Date": item.get("disclosure_date") or item.get("transaction_date", ""),
                            "Trade Date": item.get("transaction_date", ""),
                            "Politician": item.get("representative") or item.get("senator", "Member of Congress"),
                            "Chamber": "House" if "representative" in item else "Senate",
                            "Stock Symbol": tkr,
                            "Action": item.get("type", "Purchase"),
                            "Amount Range": item.get("amount", "$15,001 - $50,000"),
                            "Company / Asset": str(item.get("asset_description", ""))[:42],
                        }
                    )
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Filing Date", ascending=False).reset_index(drop=True)
        if ticker_filter:
            sub = df[df["Stock Symbol"] == ticker_filter.upper().strip()]
            if not sub.empty:
                return sub.head(40)
    return df.head(60)


@st.cache_data(ttl=1800)
def fetch_sec_insider_trades(ticker_filter=None):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if ticker_filter:
        url = f"http://openinsider.com/screener?s={ticker_filter.upper().strip()}&fd=180&xp=1&xs=1&cnt=40&page=1"
    else:
        url = "http://openinsider.com/screener?s=&fd=30&xp=1&vl=100&cnt=50&page=1"
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        tables = pd.read_html(resp.text)
        for tbl in tables:
            cols = [str(c).replace("\xa0", " ").strip() for c in tbl.columns]
            if "Filing Date" in cols and "Trade Type" in cols:
                tbl.columns = cols
                keep = [
                    c for c in [
                        "Filing Date", "Trade Date", "Ticker", "Insider Name",
                        "Title", "Trade Type", "Price", "Qty", "Owned", "ΔOwn", "Value"
                    ] if c in tbl.columns
                ]
                return tbl[keep].head(45)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=1800)
def build_master_universe():
    cong_df = fetch_congress_trades()
    ins_df = fetch_sec_insider_trades()

    cong_tickers = cong_df["Stock Symbol"].dropna().unique().tolist()[:15] if not cong_df.empty else []
    ins_tickers = (
        [str(x).strip().upper() for x in ins_df["Ticker"].dropna().unique().tolist()[:15]]
        if (not ins_df.empty and "Ticker" in ins_df.columns)
        else []
    )
    fallback_filings = ["NVDA", "MSFT", "GOOGL", "AAPL", "PANW", "LMT", "RTX", "GD", "JPM", "UNH"]

    union_list = list(
        dict.fromkeys(
            POPULAR_ETFS
            + TOP_50_MARKET_CAP
            + OPTIONS_FLOW_LEADERS
            + SOCIAL_MEDIA_BUZZ
            + cong_tickers
            + ins_tickers
            + fallback_filings
        )
    )
    return [t for t in union_list if isinstance(t, str) and 1 <= len(t) <= 5]


# ==========================================
# 3. AUTOMATIC ZERO-CLICK SQLITE LOGGING & SILENT GITHUB SYNC
# ==========================================
DB_PATH = "openwhale_signals.db"
CSV_BACKUP_PATH = "openwhale_signals_log.csv"


def init_sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_history (
            signal_date TEXT,
            ticker TEXT,
            first_seen_time TEXT,
            last_seen_time TEXT,
            times_confirmed INTEGER,
            peak_stars INTEGER,
            current_stars INTEGER,
            bias TEXT,
            strength_trend TEXT,
            duration_mins INTEGER,
            entry_price REAL,
            latest_spot REAL,
            stop_loss REAL,
            target_price REAL,
            rr_ratio REAL,
            call_flow_pct REAL,
            PRIMARY KEY (signal_date, ticker)
        )
        """
    )
    conn.commit()
    conn.close()


def _silent_github_push_worker():
    try:
        gh_repo = st.secrets.get("GITHUB_REPO", "") if "GITHUB_REPO" in st.secrets else os.environ.get("GITHUB_REPO", "")
        gh_token = st.secrets.get("GITHUB_TOKEN", "") if "GITHUB_TOKEN" in st.secrets else os.environ.get("GITHUB_TOKEN", "")
        if not (gh_repo and gh_token):
            return

        for fpath in [DB_PATH, CSV_BACKUP_PATH]:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode("utf-8")
            api_url = f"https://api.github.com/repos/{gh_repo}/contents/{fpath}"
            headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
            r_get = requests.get(api_url, headers=headers, timeout=8)
            sha = r_get.json().get("sha") if r_get.status_code == 200 else None
            payload = {
                "message": f"Auto-log 5-Star setup ({fpath}) [{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
                "content": content_b64,
                "branch": "main",
            }
            if sha:
                payload["sha"] = sha
            requests.put(api_url, headers=headers, json=payload, timeout=10)
    except Exception:
        pass


def auto_log_5star_to_db(eval_dict):
    if eval_dict["stars"] < 5:
        return
    init_sqlite_db()
    today_str = dt.date.today().strftime("%Y-%m-%d")
    now_str = dt.datetime.now().strftime("%H:%M:%S")
    tkr = eval_dict["ticker"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT first_seen_time, times_confirmed, peak_stars FROM signal_history WHERE signal_date=? AND ticker=?",
        (today_str, tkr),
    )
    row = cur.fetchone()
    is_new_5star = False

    if row is None:
        is_new_5star = True
        cur.execute(
            "INSERT INTO signal_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                today_str, tkr, now_str, now_str, 1,
                eval_dict["stars"], eval_dict["stars"], eval_dict["bias"],
                eval_dict["strength_trend"], eval_dict["duration_mins"],
                eval_dict["spot"], eval_dict["spot"], eval_dict["stop"],
                eval_dict["target"], eval_dict["rr"], eval_dict["call_pct"],
            ),
        )
    else:
        _, count_conf, peak_s = row
        cur.execute(
            """
            UPDATE signal_history
            SET last_seen_time=?, times_confirmed=?, peak_stars=?, current_stars=?,
                bias=?, strength_trend=?, duration_mins=?, latest_spot=?,
                stop_loss=?, target_price=?, rr_ratio=?, call_flow_pct=?
            WHERE signal_date=? AND ticker=?
            """,
            (
                now_str, int(count_conf) + 1, max(int(peak_s), eval_dict["stars"]),
                eval_dict["stars"], eval_dict["bias"], eval_dict["strength_trend"],
                eval_dict["duration_mins"], eval_dict["spot"], eval_dict["stop"],
                eval_dict["target"], eval_dict["rr"], eval_dict["call_pct"],
                today_str, tkr,
            ),
        )
    conn.commit()
    df_log = pd.read_sql_query("SELECT * FROM signal_history ORDER BY signal_date DESC, last_seen_time DESC", conn)
    conn.close()
    df_log.to_csv(CSV_BACKUP_PATH, index=False)

    if is_new_5star:
        threading.Thread(target=_silent_github_push_worker, daemon=True).start()


def read_signals_from_db():
    init_sqlite_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM signal_history ORDER BY signal_date DESC, last_seen_time DESC", conn)
    conn.close()
    return df


# ==========================================
# 4. MULTI-TICKER STREAMING ENGINE (30s REFRESH) & CBOE SWEEPS
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = np.where(loss == 0, 100, gain / loss)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def calc_bs_gamma(S, K, T, r, sigma):
    T = np.maximum(T, 4 / (365 * 24))
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


@st.cache_data(ttl=25)
def fetch_batch_live_quotes(tickers):
    raw = yf.download(
        tickers, period="5d", interval="5m", prepost=True, group_by="ticker", threads=True, progress=False
    )
    spot_map = {}

    for sym in tickers:
        try:
            df = raw[sym].dropna(subset=["Close"]).copy() if sym in raw else pd.DataFrame()
            if len(df) < 15:
                continue

            df["RSI"] = compute_rsi(df["Close"], 14)
            df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

            df["Date"] = df.index.date
            df["TypVol"] = ((df["High"] + df["Low"] + df["Close"]) / 3.0) * df["Volume"]
            cum_tv = df.groupby("Date")["TypVol"].cumsum()
            cum_v = df.groupby("Date")["Volume"].cumsum()
            df["VWAP"] = np.where(cum_v > 0, cum_tv / cum_v, df["Close"])

            spot = float(df["Close"].iloc[-1])
            smoothed_spot = float(df["Close"].tail(3).mean())

            unique_dates = sorted(df["Date"].unique())
            if len(unique_dates) >= 2:
                prev_day_df = df[df["Date"] == unique_dates[-2]]
                prev_close = float(prev_day_df["Close"].iloc[-1])
            else:
                prev_close = float(df["Close"].iloc[0])

            change_pct = ((spot - prev_close) / prev_close) * 100.0
            vwap = float(df["VWAP"].iloc[-1])
            rsi = float(df["RSI"].iloc[-1]) if not np.isnan(df["RSI"].iloc[-1]) else 50.0
            ema9 = float(df["EMA9"].iloc[-1])
            ema21 = float(df["EMA21"].iloc[-1])

            above_vwap_now = smoothed_spot >= vwap
            consecutive_bars = 0
            for _, r in df.iloc[::-1].iterrows():
                if (float(r["Close"]) >= float(r["VWAP"])) == above_vwap_now:
                    consecutive_bars += 1
                else:
                    break
            duration_mins = consecutive_bars * 5
            duration_label = (
                f"{duration_mins // 60}h {duration_mins % 60}m" if duration_mins >= 60 else f"{duration_mins}m"
            )

            spread_now = abs(ema9 - ema21) / max(spot, 1) * 1000
            spread_prev = (
                abs(float(df["EMA9"].iloc[-7]) - float(df["EMA21"].iloc[-7])) / max(spot, 1) * 1000
                if len(df) >= 7
                else spread_now
            )

            if duration_mins <= 15:
                strength_trend = "⚡ Newly Triggered"
            elif spread_now > spread_prev * 1.06:
                strength_trend = "🔥 Gaining Strength"
            elif spread_now < spread_prev * 0.92:
                strength_trend = "❄️ Cooling Off"
            else:
                strength_trend = "💪 Holding Steady"

            spot_map[sym] = {
                "ticker": sym,
                "spot": round(spot, 2),
                "smoothed_spot": round(smoothed_spot, 2),
                "change_pct": round(change_pct, 2),
                "vwap": round(vwap, 2),
                "rsi": round(rsi, 1),
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "duration_mins": duration_mins,
                "duration_label": duration_label,
                "strength_trend": strength_trend,
                "intraday_df": df.tail(75),
            }
        except Exception:
            continue

    return spot_map


@st.cache_data(ttl=30)
def fetch_cboe_options_for_ticker(ticker_symbol, spot_override=None):
    clean_sym = ticker_symbol.upper().strip().replace("^", "")
    index_tickers = {"SPX", "VIX", "NDX", "RUT", "DJX", "XSP"}
    cboe_sym = f"_{clean_sym}" if clean_sym in index_tickers else clean_sym

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_sym}.json"
    resp = requests.get(url, headers=headers, timeout=8)
    if resp.status_code != 200:
        raise ValueError(f"CBOE unavailable for {ticker_symbol}")

    payload = resp.json().get("data", {})
    quote_timestamp = payload.get("timestamp") or dt.datetime.now().strftime("%H:%M:%S")
    spot = float(spot_override or payload.get("current_price") or payload.get("close") or 0)
    raw_options = payload.get("options", [])
    if not raw_options or spot <= 0:
        raise ValueError("Empty options chain")

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
        if strike < spot * 0.84 or strike > spot * 1.16:
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
            exec_side = "🔥 Ask-Side Sweep"
        elif bid > 0 and last_p <= bid * 1.02:
            exec_side = "❄️ Bid-Side Block"
        else:
            exec_side = "⚖️ Mid-Market"

        if dte == 0:
            horizon = "⚡ 0DTE Scalp"
        elif dte <= 5:
            horizon = f"🏃 Weekly ({dte}d)"
        elif dte <= 21:
            horizon = f"📅 Swing ({dte}d)"
        else:
            horizon = f"🏛️ Macro ({dte}d)"

        rows.append(
            {
                "Ticker": clean_sym,
                "Spot": round(spot, 2),
                "Contract": sym,
                "Type": "🟢 CALL" if cp == "C" else "🔴 PUT",
                "raw_type": "call" if cp == "C" else "put",
                "Strike": strike,
                "Expiration": exp_str,
                "DTE": dte,
                "Timeframe_Horizon": horizon,
                "Last_Price": round(mark, 2),
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
        raise ValueError("No valid contracts")

    nearest_exps = sorted(df["Expiration"].unique())[:4]
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

    sweeps_100k = (
        df[(df["Volume"] >= 150) & (df["Vol_OI_Ratio"] >= 1.25) & (df["Premium_USD"] >= MIN_SWEEP_USD)]
        .sort_values(["Vol_OI_Ratio", "Premium_USD"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(float(zero_gamma), 2),
        "call_pct": round(call_pct, 1),
        "sweeps_df": sweeps_100k,
    }


def evaluate_star_rating(spot_info, opt_info):
    spot = spot_info["spot"]
    anchor_p = spot_info["smoothed_spot"]
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

    is_bullish = (anchor_p >= vwap and call_pct >= 50.0) or (call_pct >= 60.0)
    stars = 0

    if is_bullish:
        bias = "LONG"
        if anchor_p >= vwap * 0.999 and ema_bull:
            stars += 1
        if anchor_p >= zero_gamma * 0.999:
            stars += 1
        if call_pct >= 56.0:
            stars += 1
        if call_sweeps >= 1:
            stars += 1
        target = call_wall if call_wall > spot * 1.005 else round(spot * 1.02, 2)
        stop = round(max(put_wall, min(vwap, zero_gamma) * 0.994, spot * 0.988), 2)
        if stop >= spot:
            stop = round(spot * 0.99, 2)
        rr = round((target - spot) / max(spot - stop, 0.01), 2)
        if 32.0 <= rsi <= 69.0 and rr >= 1.4:
            stars += 1
    else:
        bias = "SHORT"
        put_pct = 100.0 - call_pct
        if anchor_p < vwap * 1.001 and not ema_bull:
            stars += 1
        if anchor_p < zero_gamma * 1.001:
            stars += 1
        if put_pct >= 56.0:
            stars += 1
        if put_sweeps >= 1:
            stars += 1
        target = put_wall if put_wall < spot * 0.995 else round(spot * 0.98, 2)
        stop = round(min(call_wall, max(vwap, zero_gamma) * 1.006, spot * 1.012), 2)
        if stop <= spot:
            stop = round(spot * 1.01, 2)
        rr = round((spot - target) / max(stop - spot, 0.01), 2)
        if 31.0 <= rsi <= 68.0 and rr >= 1.4:
            stars += 1

    star_str = "⭐" * stars + "☆" * (5 - stars)
    verdict = (
        "🟢 5-STAR LONG"
        if (stars == 5 and bias == "LONG")
        else (
            "🔴 5-STAR SHORT"
            if (stars == 5 and bias == "SHORT")
            else ("🟢 4-STAR LONG" if (stars == 4 and bias == "LONG") else ("🔴 4-STAR SHORT" if (stars == 4 and bias == "SHORT") else f"🟡 {stars}-STAR WAIT"))
        )
    )

    res = {
        "ticker": spot_info["ticker"],
        "stars": stars,
        "star_str": star_str,
        "bias": bias,
        "verdict_label": verdict,
        "strength_trend": spot_info["strength_trend"],
        "duration_label": spot_info["duration_label"],
        "duration_mins": spot_info["duration_mins"],
        "spot": spot,
        "change_pct": spot_info["change_pct"],
        "vwap": vwap,
        "zero_gamma": zero_gamma,
        "put_wall": put_wall,
        "call_wall": call_wall,
        "entry": round(spot, 2),
        "stop": stop,
        "target": target,
        "rr": rr,
        "rsi": rsi,
        "call_pct": call_pct,
        "sweeps_summary": f"{call_sweeps}C / {put_sweeps}P",
    }

    if stars == 5:
        auto_log_5star_to_db(res)
    return res


@st.cache_data(ttl=25)
def run_multi_ticker_grid_and_sweeps(universe_tickers):
    spot_map = fetch_batch_live_quotes(universe_tickers)
    active_syms = list(spot_map.keys())

    grid_rows = []
    all_sweeps = []

    def _worker(sym):
        s_info = spot_map[sym]
        try:
            o_info = fetch_cboe_options_for_ticker(sym, spot_override=s_info["spot"])
            eval_res = evaluate_star_rating(s_info, o_info)
            return eval_res, o_info["sweeps_df"]
        except Exception:
            fallback_opt = {
                "call_wall": round(s_info["spot"] * 1.02, 2),
                "put_wall": round(s_info["spot"] * 0.98, 2),
                "zero_gamma": s_info["vwap"],
                "call_pct": 50.0,
                "sweeps_df": pd.DataFrame(),
            }
            return evaluate_star_rating(s_info, fallback_opt), pd.DataFrame()

    with ThreadPoolExecutor(max_workers=14) as ex:
        futures = {ex.submit(_worker, sym): sym for sym in active_syms}
        for f in as_completed(futures):
            ev_res, sw_df = f.result()
            grid_rows.append(ev_res)
            if not sw_df.empty:
                all_sweeps.append(sw_df)

    grid_df = (
        pd.DataFrame(grid_rows)
        .sort_values(["stars", "duration_mins", "rr"], ascending=[False, False, False])
        .reset_index(drop=True)
        if grid_rows
        else pd.DataFrame()
    )

    if all_sweeps:
        master_sweeps_df = (
            pd.concat(all_sweeps, ignore_index=True)
            .sort_values(["Vol_OI_Ratio", "Premium_USD"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        master_sweeps_df = pd.DataFrame()

    return grid_df, master_sweeps_df, spot_map


# ==========================================
# 5. YTD 2026 BACKTEST ENGINE ($100 PER 5-STAR TRADE ACCUMULATION)
# ==========================================
@st.cache_data(ttl=900)
def run_ytd_2026_5star_backtest(tickers):
    """
    Backtests every 5-Star setup since 2026-01-01 across the tracked universe.
    Every time a 5-Star trade triggers, $100 is invested in that setup, and the
    dollar returns are accumulated trade-by-trade chronologically.
    """
    raw = yf.download(
        tickers, start="2025-12-01", interval="1d", prepost=False, group_by="ticker", threads=True, progress=False
    )

    trades = []
    for sym in tickers:
        try:
            df = raw[sym].dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy() if sym in raw else pd.DataFrame()
            if len(df) < 25:
                continue

            df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
            df["RSI"] = compute_rsi(df["Close"], 14)
            df["Vol_MA20"] = df["Volume"].rolling(20).mean()
            df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
            df["Typical_VWAP"] = (df["High"] + df["Low"] + df["Close"]) / 3.0

            df_2026 = df[df.index >= "2026-01-01"].copy()
            if len(df_2026) < 5:
                continue

            i = 1
            while i < len(df_2026) - 3:
                row = df_2026.iloc[i]
                prev_row = df_2026.iloc[i - 1]
                sig_date = df_2026.index[i].strftime("%Y-%m-%d")

                if pd.isna(row["ATR"]) or pd.isna(row["Vol_MA20"]) or row["ATR"] <= 0:
                    i += 1
                    continue

                vol_surge = float(row["Volume"] / max(row["Vol_MA20"], 1))
                atr = float(row["ATR"])
                entry = float(row["Close"])
                rsi = float(row["RSI"]) if not np.isnan(row["RSI"]) else 50.0

                is_5star_long = (
                    entry > float(row["Typical_VWAP"])
                    and float(row["EMA9"]) > float(row["EMA21"])
                    and float(row["Close"]) > float(prev_row["High"])
                    and 46.0 <= rsi <= 68.0
                    and vol_surge >= 1.35
                )

                is_5star_short = (
                    entry < float(row["Typical_VWAP"])
                    and float(row["EMA9"]) < float(row["EMA21"])
                    and float(row["Close"]) < float(prev_row["Low"])
                    and 32.0 <= rsi <= 54.0
                    and vol_surge >= 1.35
                )

                if not (is_5star_long or is_5star_short):
                    i += 1
                    continue

                setup_dir = "🟢 5-STAR LONG" if is_5star_long else "🔴 5-STAR SHORT"
                stop_dist = max(atr * 1.0, entry * 0.012)
                target_dist = stop_dist * 2.0

                stop_price = entry - stop_dist if is_5star_long else entry + stop_dist
                target_price = entry + target_dist if is_5star_long else entry - target_dist

                forward_slice = df_2026.iloc[i + 1 : min(i + 6, len(df_2026))]
                exit_price = float(forward_slice["Close"].iloc[-1])
                exit_date = forward_slice.index[-1].strftime("%Y-%m-%d")
                status = "⏱️ Time Exit (5d)"

                for f_dt, f_bar in forward_slice.iterrows():
                    if is_5star_long:
                        if float(f_bar["Low"]) <= stop_price:
                            exit_price = stop_price
                            exit_date = f_dt.strftime("%Y-%m-%d")
                            status = "❌ FAILURE (Stop Hit)"
                            break
                        if float(f_bar["High"]) >= target_price:
                            exit_price = target_price
                            exit_date = f_dt.strftime("%Y-%m-%d")
                            status = "✅ SUCCESS (Target Hit)"
                            break
                    else:
                        if float(f_bar["High"]) >= stop_price:
                            exit_price = stop_price
                            exit_date = f_dt.strftime("%Y-%m-%d")
                            status = "❌ FAILURE (Stop Hit)"
                            break
                        if float(f_bar["Low"]) <= target_price:
                            exit_price = target_price
                            exit_date = f_dt.strftime("%Y-%m-%d")
                            status = "✅ SUCCESS (Target Hit)"
                            break

                roi_pct = (
                    ((exit_price - entry) / entry) * 100.0
                    if is_5star_long
                    else ((entry - exit_price) / entry) * 100.0
                )
                if status == "⏱️ Time Exit (5d)":
                    status = "✅ SUCCESS (Time Gain)" if roi_pct > 0 else "❌ FAILURE (Time Loss)"

                risk_pct = (stop_dist / entry) * 100.0
                dollar_risk_on_100 = round(100.0 * (risk_pct / 100.0), 2)
                dollar_return_on_100 = round(100.0 * (roi_pct / 100.0), 2)
                final_value_100 = round(100.0 + dollar_return_on_100, 2)

                trades.append(
                    {
                        "Signal Date": sig_date,
                        "Exit Date": exit_date,
                        "Stock Symbol": sym,
                        "5-Star Setup": setup_dir,
                        "Trade Outcome": status,
                        "Entry Price ($)": round(entry, 2),
                        "Stop-Loss ($)": round(stop_price, 2),
                        "Target Price ($)": round(target_price, 2),
                        "Exit Price ($)": round(exit_price, 2),
                        "Max Risk on $100 ($)": dollar_risk_on_100,
                        "Risk (%)": round(risk_pct, 2),
                        "Profit / Loss on $100 ($)": dollar_return_on_100,
                        "$100 Trade Final Value ($)": final_value_100,
                        "Trade ROI (%)": round(roi_pct, 2),
                    }
                )
                i += 3
        except Exception:
            continue

    bt_df = pd.DataFrame(trades)
    if not bt_df.empty:
        # Sort chronologically (oldest to newest) first to compute running Accumulated Profit
        bt_df = bt_df.sort_values(["Signal Date", "Stock Symbol"], ascending=[True, True]).reset_index(drop=True)
        bt_df["Trade #"] = bt_df.index + 1
        bt_df["Accumulated Total Profit ($)"] = bt_df["Profit / Loss on $100 ($)"].cumsum().round(2)

        # Reorder columns cleanly with Trade # first and Accumulated Total Profit ($) at the end
        ordered_cols = [
            "Trade #",
            "Signal Date",
            "Exit Date",
            "Stock Symbol",
            "5-Star Setup",
            "Trade Outcome",
            "Entry Price ($)",
            "Stop-Loss ($)",
            "Target Price ($)",
            "Exit Price ($)",
            "Max Risk on $100 ($)",
            "Risk (%)",
            "Profit / Loss on $100 ($)",
            "$100 Trade Final Value ($)",
            "Trade ROI (%)",
            "Accumulated Total Profit ($)",
        ]
        bt_df = bt_df[ordered_cols]
    return bt_df


# ==========================================
# 6. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🐋 OpenWhale Terminal")
st.sidebar.caption("Auto-Logging 5-Star Setups • $100k+ Sweeps • 30s Live Stream")

master_universe = build_master_universe()

use_plain_english = st.sidebar.toggle(
    "🗣️ Use Everyday Plain-English Terminology",
    value=True,
    help="Switches between beginner-friendly labels and institutional quant labels.",
)

live_stream_on = st.sidebar.toggle(
    "⚡ Live Stream Quotes Every 30s",
    value=True,
    help="Automatically refreshes spot prices, multi-ticker grid, and $100k+ sweeps every 30 seconds.",
)

scan_batch_size = st.sidebar.slider(
    "📡 Active Grid Size (from Union Universe):",
    min_value=24,
    max_value=len(master_universe),
    value=min(48, len(master_universe)),
    step=8,
    help=f"Total Union Universe contains {len(master_universe)} stocks (Top 50 Market Cap + Mag 7 + ETFs + Options Flow + Social Buzz + Congress/Insider).",
)

chart_ticker = st.sidebar.selectbox(
    "📈 Focus Ticker for Candlestick Chart (Shown on Tab 1):",
    options=master_universe,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.success(
    f"✅ **Auto-DB Logging Active:** Any 5-Star setup across all {len(master_universe)} union tickers is automatically logged to `openwhale_signals.db`."
)

# ==========================================
# 7. MAIN DASHBOARD EXECUTION
# ==========================================
st.title("🐋 OpenWhale: Multi-Ticker Live Grid, $100k+ Sweeps & 2026 Backtester")
render_hover_glossary_banner()

active_universe = master_universe[:scan_batch_size]
if chart_ticker not in active_universe:
    active_universe = [chart_ticker] + active_universe

with st.spinner(f"Streaming live quotes & scanning $100k+ institutional sweeps across {len(active_universe)} tickers..."):
    grid_df, sweeps_100k_df, spot_map = run_multi_ticker_grid_and_sweeps(active_universe)

five_star_count = len(grid_df[grid_df["stars"] == 5]) if not grid_df.empty else 0
four_star_count = len(grid_df[grid_df["stars"] == 4]) if not grid_df.empty else 0
total_100k_sweeps = len(sweeps_100k_df) if not sweeps_100k_df.empty else 0
db_logged_df = read_signals_from_db()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🌐 Union Universe Size", f"{len(master_universe)} Tickers", f"Streaming {len(active_universe)} Live")
m2.metric("⭐⭐⭐⭐⭐ Active 5-Star Setups", f"{five_star_count} Live", "Auto-Logged to DB")
m3.metric("⭐⭐⭐⭐ Active 4-Star Setups", f"{four_star_count} Live", "High Confluence")
m4.metric("🚨 $100k+ Institutional Sweeps", f"{total_100k_sweeps} Sweeps", "Sorted by Urgency")
m5.metric("💾 Total DB Logged 5-Star Trades", f"{len(db_logged_df)} Saved", "Zero-Click Auto-Save")

tab_grid, tab_sweeps, tab_backtest, tab_db, tab_congress = st.tabs(
    [
        f"📡 1. Live Candlestick Chart ({chart_ticker}) & Multi-Ticker Grid",
        "🚨 2. All $100k+ Institutional Sweeps (Sorted by Urgency)",
        "📊 3. 2026 YTD 5-Star Backtest ($100 Per Trade Accumulated)",
        "💾 4. Auto-Logged 5-Star SQLite Database",
        "🏛️ 5. Congress & Corporate Insider Filings",
    ]
)

# ==========================================
# TAB 1: FOCUS CANDLESTICK CHART + MULTI-TICKER LIVE STREAM GRID
# ==========================================
with tab_grid:
    # 1A. Render the Live Candlestick Chart for the Sidebar-Selected Focus Ticker
    if chart_ticker in spot_map and not grid_df.empty:
        focus_spot = spot_map[chart_ticker]
        focus_row_df = grid_df[grid_df["ticker"] == chart_ticker]
        focus_eval = focus_row_df.iloc[0].to_dict() if not focus_row_df.empty else grid_df.iloc[0].to_dict()

        lbl_vwap = "Fair Value Price (VWAP)" if use_plain_english else "Session VWAP"
        lbl_floor = "Price Floor (Safety Net)" if use_plain_english else "Put Wall (-GEX)"
        lbl_ceil = "Price Ceiling (Target)" if use_plain_english else "Call Wall (+GEX)"

        st.subheader(f"📈 Live 5-Minute Candlestick Stream & Institutional Walls: {chart_ticker}")
        c_chart, c_ticket = st.columns([0.65, 0.35])

        with c_chart:
            intra_df = focus_spot["intraday_df"]
            fig_candle = go.Figure()
            fig_candle.add_trace(
                go.Candlestick(
                    x=intra_df.index,
                    open=intra_df["Open"],
                    high=intra_df["High"],
                    low=intra_df["Low"],
                    close=intra_df["Close"],
                    name=f"{chart_ticker} Price",
                )
            )
            fig_candle.add_trace(
                go.Scatter(
                    x=intra_df.index,
                    y=intra_df["VWAP"],
                    mode="lines",
                    name=lbl_vwap,
                    line=dict(color="#38bdf8", width=2),
                )
            )
            fig_candle.add_hline(
                y=focus_eval["call_wall"],
                line_dash="dash",
                line_color="#22c55e",
                annotation_text=f"{lbl_ceil}: ${focus_eval['call_wall']}",
            )
            fig_candle.add_hline(
                y=focus_eval["put_wall"],
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"{lbl_floor}: ${focus_eval['put_wall']}",
            )
            fig_candle.add_hline(
                y=focus_eval["zero_gamma"],
                line_dash="dot",
                line_color="#eab308",
                annotation_text=f"Danger Line: ${focus_eval['zero_gamma']}",
            )
            fig_candle.update_layout(
                template="plotly_dark",
                height=390,
                xaxis_rangeslider_visible=False,
                margin=dict(t=15, l=10, r=10, b=10),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_candle, use_container_width=True)

        with c_ticket:
            st.markdown(
                f"""
                <div class="star-card">
                    <h4 style="margin:0; color:#f8fafc;">🎯 {chart_ticker} Focus Trade Ticket</h4>
                    <p style="margin:6px 0; font-size:1.05rem; color:#eab308; font-weight:700;">
                        {focus_eval['star_str']} — {focus_eval['verdict_label']}
                    </p>
                    <p style="margin:4px 0; color:#38bdf8; font-weight:600;">
                        {focus_eval['strength_trend']} ({focus_eval['duration_label']})
                    </p>
                    <hr style="border-color:#334155; margin:8px 0;">
                    <p style="margin:4px 0; color:#e2e8f0;">• <b>Live Traded Price:</b> ${focus_eval['spot']:,.2f} ({focus_eval['change_pct']:+.2f}%)</p>
                    <p style="margin:4px 0; color:#e2e8f0;">• <b>{lbl_vwap}:</b> ${focus_eval['vwap']:,.2f}</p>
                    <p style="margin:4px 0; color:#e2e8f0;">• <b>{lbl_floor}:</b> ${focus_eval['put_wall']:,.2f}</p>
                    <p style="margin:4px 0; color:#e2e8f0;">• <b>{lbl_ceil}:</b> ${focus_eval['call_wall']:,.2f}</p>
                    <p style="margin:4px 0; color:#f87171;">• <b>Hard Stop-Loss:</b> ${focus_eval['stop']:,.2f}</p>
                    <p style="margin:4px 0; color:#4ade80;">• <b>Profit Target:</b> ${focus_eval['target']:,.2f}</p>
                    <p style="margin:4px 0; color:#cbd5e1;">• <b>Smart Money Call Flow:</b> {focus_eval['call_pct']:.1f}% ({focus_eval['sweeps_summary']} Sweeps)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader(f"📡 Multi-Ticker Live Quote & 5-Star Confluence Grid ({len(grid_df)} Tickers Streaming)")

    if not grid_df.empty:
        top_tiles = grid_df.head(12)
        for row_idx in range(0, len(top_tiles), 6):
            cols = st.columns(6)
            for col_idx, (_, item) in enumerate(top_tiles.iloc[row_idx : row_idx + 6].iterrows()):
                with cols[col_idx]:
                    st.metric(
                        label=f"{item['ticker']} ({item['star_str']})",
                        value=f"${item['spot']:,.2f}",
                        delta=f"{item['change_pct']:+.2f}% | {item['duration_label']}",
                    )

        st.markdown("#### 📋 Full Multi-Ticker Live Streaming Table")
        grid_col_map = (
            {
                "ticker": "Stock Symbol",
                "star_str": "Star Rating",
                "verdict_label": "Action Verdict",
                "spot": "Live Price ($)",
                "change_pct": "Today Move (%)",
                "strength_trend": "Signal Strength",
                "duration_label": "Duration Held",
                "vwap": "Fair Price VWAP ($)",
                "put_wall": "Price Floor ($)",
                "call_wall": "Price Ceiling ($)",
                "stop": "Stop-Loss ($)",
                "target": "Profit Target ($)",
                "rr": "Reward:Risk",
                "call_pct": "Smart Money Call (%)",
                "sweeps_summary": "$100k+ Sweeps (Calls/Puts)",
            }
            if use_plain_english
            else {
                "ticker": "Ticker",
                "star_str": "Rating",
                "verdict_label": "Signal",
                "spot": "Spot ($)",
                "change_pct": "Chg (%)",
                "strength_trend": "Momentum",
                "duration_label": "VWAP Duration",
                "vwap": "VWAP ($)",
                "put_wall": "Put Wall ($)",
                "call_wall": "Call Wall ($)",
                "stop": "Stop ($)",
                "target": "Target ($)",
                "rr": "R:R",
                "call_pct": "Call Flow (%)",
                "sweeps_summary": "Sweeps >=100k",
            }
        )
        st.dataframe(
            grid_df[list(grid_col_map.keys())].rename(columns=grid_col_map),
            use_container_width=True,
            hide_index=True,
            height=480,
        )

# ==========================================
# TAB 2: ALL $100K+ SWEEPS ACROSS ALL TICKERS (SORTED BY URGENCY)
# ==========================================
with tab_sweeps:
    st.subheader("🚨 Multi-Ticker Institutional Sweep Tape (Minimum $100,000 • Sorted by Urgency)")
    st.caption(
        "Aggregates all institutional sweeps >= $100,000 across the entire tracked stock & ETF universe, ranked highest-to-lowest by Urgency (`Volume / Open Interest`)."
    )

    if not sweeps_100k_df.empty:
        disp_sw = sweeps_100k_df.copy()
        disp_sw["Money Spent ($)"] = disp_sw["Premium_USD"].apply(lambda x: f"${x:,.0f}")
        disp_sw["Strike Price ($)"] = disp_sw["Strike"].apply(lambda x: f"${x:g}")
        disp_sw["Live Price ($)"] = disp_sw["Spot"].apply(lambda x: f"${x:,.2f}")
        disp_sw["Urgency (Vol/OI)"] = disp_sw["Vol_OI_Ratio"].apply(lambda x: f"🔥 {x:.2f}x OI")

        sw_rename = {
            "Ticker": "Stock Symbol",
            "Live Price ($)": "Live Price ($)",
            "Type": "Option Direction",
            "Strike Price ($)": "Target Strike ($)",
            "Expiration": "Expiry Date",
            "Timeframe_Horizon": "Trade Timeframe",
            "Urgency (Vol/OI)": "Urgency (Volume vs OI)",
            "Money Spent ($)": "Institutional Money Spent ($)",
            "Execution_Side": "Buyer Aggression",
            "Volume": "Contracts Traded Today",
            "Open_Interest": "Existing Open Contracts",
            "IV": "Implied Volatility (%)",
            "Trade_Time": "Last Print Time",
        }
        st.dataframe(
            disp_sw[list(sw_rename.keys())].rename(columns=sw_rename),
            use_container_width=True,
            hide_index=True,
            height=580,
        )
    else:
        st.info("No institutional sweeps >= $100,000 detected in the current batch—increase Active Grid Size in the sidebar.")

# ==========================================
# TAB 3: YTD 2026 BACKTESTER ($100 PER 5-STAR TRADE ACCUMULATION)
# ==========================================
with tab_backtest:
    st.subheader("📊 2026 YTD 5-Star Backtest (Putting $100 into Every 5-Star Setup Since 01/01/2026)")
    st.markdown(
        """
        **How This Backtest Works:**
        * Every time a **5-Star Setup** triggers on any tracked stock since **January 1, 2026**, you invest **$100** into that trade.
        * The **Profit or Loss ($)** from each `$100` trade is added to your **Accumulated Total Profit ($)** column so you can see exactly how your money grows trade after trade.
        """
    )

    with st.spinner("Running YTD 2026 5-Star Backtest (01/01/2026 to Present) across tracked universe..."):
        bt_2026_df = run_ytd_2026_5star_backtest(master_universe[:45])

    if not bt_2026_df.empty:
        total_t = len(bt_2026_df)
        succ_t = len(bt_2026_df[bt_2026_df["Trade Outcome"].str.startswith("✅")])
        fail_t = total_t - succ_t
        win_rate = (succ_t / total_t) * 100.0

        total_accumulated_profit = bt_2026_df["Profit / Loss on $100 ($)"].sum()
        avg_roi_pct = bt_2026_df["Trade ROI (%)"].mean()
        avg_risk_dlr = bt_2026_df["Max Risk on $100 ($)"].mean()

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Total 5-Star Trades (2026)", f"{total_t} Trades", "$100 Put in Each Trade")
        b2.metric("Success Win Rate", f"{win_rate:.1f}%", f"{succ_t} Wins / {fail_t} Losses")
        b3.metric("💰 Accumulated Total Profit", f"${total_accumulated_profit:+,.2f}", "Sum of All $100 Trades")
        b4.metric("📈 Avg Return (ROI) per Trade", f"{avg_roi_pct:+.2f}%", f"${avg_roi_pct:+.2f} per $100 Trade")
        b5.metric("🛡️ Avg Stop-Loss Risk per $100", f"${avg_risk_dlr:.2f}", f"{avg_risk_dlr:.2f}% Max Risk if Stopped")

        # Clear Two-Layer Chart: Individual $100 Trade Profit/Loss Bars + Accumulated Total Profit Line
        fig_bt = make_subplots(specs=[[{"secondary_y": True}]])

        bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in bt_2026_df["Profit / Loss on $100 ($)"]]
        hover_labels = (
            "<b>Trade #"
            + bt_2026_df["Trade #"].astype(str)
            + " ("
            + bt_2026_df["Signal Date"]
            + ") — "
            + bt_2026_df["Stock Symbol"]
            + "</b><br>"
            + "Setup: "
            + bt_2026_df["5-Star Setup"]
            + "<br>"
            + "Outcome: "
            + bt_2026_df["Trade Outcome"]
            + "<br>"
            + "This $100 Trade P&L: $"
            + bt_2026_df["Profit / Loss on $100 ($)"].map(lambda x: f"{x:+.2f}")
            + " ("
            + bt_2026_df["Trade ROI (%)"].map(lambda x: f"{x:+.2f}%")
            + ")<br>"
            + "<b>Running Accumulated Profit: $"
            + bt_2026_df["Accumulated Total Profit ($)"].map(lambda x: f"{x:+,.2f}")
            + "</b>"
        )

        # Bar trace: Single $100 trade P&L
        fig_bt.add_trace(
            go.Bar(
                x=bt_2026_df["Trade #"],
                y=bt_2026_df["Profit / Loss on $100 ($)"],
                name="Single $100 Trade Profit/Loss ($)",
                marker_color=bar_colors,
                opacity=0.55,
                hovertext=hover_labels,
                hoverinfo="text",
            ),
            secondary_y=False,
        )

        # Line trace: Running Accumulated Profit ($)
        fig_bt.add_trace(
            go.Scatter(
                x=bt_2026_df["Trade #"],
                y=bt_2026_df["Accumulated Total Profit ($)"],
                mode="lines+markers",
                name="Accumulated Total Profit ($)",
                line=dict(color="#eab308", width=3.5),
                marker=dict(size=6, color="#eab308"),
                hovertext=hover_labels,
                hoverinfo="text",
            ),
            secondary_y=True,
        )

        fig_bt.update_layout(
            title="💰 Accumulated Profit Curve Since 01/01/2026 (Investing $100 in Every 5-Star Setup)",
            xaxis_title="Chronological 5-Star Trade Number (Trade #1 to Present)",
            template="plotly_dark",
            height=400,
            margin=dict(t=40, l=10, r=10, b=10),
            legend=dict(orientation="h", y=1.11),
        )
        fig_bt.update_yaxes(title_text="Single $100 Trade P&L ($)", secondary_y=False)
        fig_bt.update_yaxes(title_text="Accumulated Total Profit ($)", secondary_y=True)
        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("#### 📜 Trade-by-Trade 2026 YTD Log (Newest Trades First • Clean Headers)")
        # Show newest trades at top of table while keeping Trade # and Accumulated Total Profit intact
        st.dataframe(
            bt_2026_df.iloc[::-1].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=480,
        )

# ==========================================
# TAB 4: AUTO-LOGGED 5-STAR SQLITE DATABASE
# ==========================================
with tab_db:
    st.subheader("💾 Zero-Click Auto-Logged 5-Star SQLite Database (`openwhale_signals.db`)")
    st.caption(
        "Every time any ticker hits a 5-Star rating (even for 1 minute), it is automatically saved here and evaluated on a $100 position size."
    )
    if not db_logged_df.empty:
        db_show = db_logged_df.copy()
        db_show["Trade ROI (%)"] = np.where(
            db_show["bias"] == "LONG",
            ((db_show["latest_spot"] - db_show["entry_price"]) / db_show["entry_price"]) * 100.0,
            ((db_show["entry_price"] - db_show["latest_spot"]) / db_show["entry_price"]) * 100.0,
        ).round(2)
        db_show["Max Risk (%)"] = (
            np.abs(db_show["entry_price"] - db_show["stop_loss"]) / db_show["entry_price"] * 100.0
        ).round(2)
        db_show["Profit / Loss on $100 ($)"] = (100.0 * db_show["Trade ROI (%)"] / 100.0).round(2)
        db_show["Max Risk on $100 ($)"] = (100.0 * db_show["Max Risk (%)"] / 100.0).round(2)
        db_show["Accumulated Total Profit ($)"] = db_show["Profit / Loss on $100 ($)"].cumsum().round(2)

        db_clean_headers = {
            "signal_date": "Signal Date",
            "ticker": "Stock Symbol",
            "first_seen_time": "First Logged Time",
            "last_seen_time": "Latest Update Time",
            "times_confirmed": "Times Confirmed",
            "bias": "Direction",
            "strength_trend": "Signal Strength",
            "duration_mins": "Duration Held (Mins)",
            "entry_price": "First Entry Price ($)",
            "latest_spot": "Current Price ($)",
            "stop_loss": "Stop-Loss ($)",
            "target_price": "Target Price ($)",
            "Max Risk on $100 ($)": "Max Risk on $100 ($)",
            "Profit / Loss on $100 ($)": "Profit / Loss on $100 ($)",
            "Trade ROI (%)": "Trade ROI (%)",
            "Accumulated Total Profit ($)": "Accumulated Total Profit ($)",
        }
        st.dataframe(
            db_show[list(db_clean_headers.keys())].rename(columns=db_clean_headers),
            use_container_width=True,
            hide_index=True,
            height=450,
        )
    else:
        st.info("Waiting for the first 5-Star setup of the session—will log automatically as soon as one triggers.")

# ==========================================
# TAB 5: CONGRESS & CORPORATE INSIDER TRACKER
# ==========================================
with tab_congress:
    st.subheader("🏛️ U.S. Congress (STOCK Act) & SEC Form 4 Corporate Insider Filings")
    col_cong, col_ins = st.columns(2)
    with col_cong:
        st.markdown("#### 🏛️ Latest U.S. Congress Trades")
        cong_df = fetch_congress_trades()
        if not cong_df.empty:
            st.dataframe(cong_df, use_container_width=True, hide_index=True, height=450)
    with col_ins:
        st.markdown("#### 👔 Latest SEC Form 4 Corporate Insider Buys")
        ins_df = fetch_sec_insider_trades()
        if not ins_df.empty:
            st.dataframe(ins_df, use_container_width=True, hide_index=True, height=450)

# Stream quotes every 30 seconds automatically
if live_stream_on:
    time.sleep(30)
    st.rerun()
