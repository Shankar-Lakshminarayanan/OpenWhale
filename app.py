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
    .health-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
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
# 1. EXTENDED UNION STOCK UNIVERSE
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
    "SOXL", "TQQQ", "SQQQ", "SPXL", "XBI", "KRE", "GDX", "XLB", "XLC", "VTI"
]

EXTENDED_SP_NASDAQ = [
    "MS", "SPGI", "AXP", "BLK", "C", "SCHW", "CB", "PGR", "PYPL", "COF",
    "USB", "PNC", "TFC", "CME", "ICE", "MCO", "TMUS", "CMCSA", "CHTR", "DIS",
    "BKNG", "MAR", "CMG", "MCD", "LOW", "TJX", "ORLY", "AZO", "ROST", "NKE",
    "SBUX", "TGT", "F", "GM", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    "OXY", "HAL", "DE", "LMT", "RTX", "HON", "UNP", "BA", "UPS", "FDX",
    "ADP", "ETN", "ITW", "GD", "NOC", "WM", "SHW", "FCX", "NEM", "NEE",
    "SO", "DUK", "AMAT", "MU", "LRCX", "ADI", "KLAC", "PANW", "SNPS", "CDNS",
    "CRWD", "PLTR", "UBER", "ABNB", "SHOP", "SNOW", "ARM", "SMCI", "DELL", "MRVL"
]

OPTIONS_FLOW_LEADERS = [
    "COIN", "MSTR", "INTC", "SQ", "PFE", "MRNA", "GILD", "REGN", "VRTX", "ISRG",
    "LLY", "BMY", "CVS", "HUM", "CI", "BIIB", "GEHC", "DHR", "ABT", "MDLZ"
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
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                for item in r.json()[:300]:
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
        resp = requests.get(url, headers=headers, timeout=7)
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

    cong_tickers = cong_df["Stock Symbol"].dropna().unique().tolist()[:20] if not cong_df.empty else []
    ins_tickers = (
        [str(x).strip().upper() for x in ins_df["Ticker"].dropna().unique().tolist()[:20]]
        if (not ins_df.empty and "Ticker" in ins_df.columns)
        else []
    )
    fallback_filings = ["NVDA", "MSFT", "GOOGL", "AAPL", "PANW", "LMT", "RTX", "GD", "JPM", "UNH"]

    union_list = list(
        dict.fromkeys(
            POPULAR_ETFS
            + TOP_50_MARKET_CAP
            + EXTENDED_SP_NASDAQ
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
# 4. MULTI-TICKER STREAMING ENGINE & CBOE SWEEPS
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
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for sym in tickers:
        try:
            if is_multi:
                df = raw[sym].dropna(subset=["Close"]).copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                df = raw.dropna(subset=["Close"]).copy() if not raw.empty else pd.DataFrame()

            if len(df) < 10:
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

            change_pct = ((spot - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
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
            elif spread_now > spread_prev * 1.05:
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
                "Vol_OI_Ratio": round(
