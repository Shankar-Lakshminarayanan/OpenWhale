import datetime as dt
import io
import re
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
    page_title="OpenWhale | Institutional Flow & GEX",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 1. BLACK-SCHOLES GREEKS FALLBACK ENGINE
# ==========================================
def calc_bs_greeks(S, K, T, r, sigma, option_type="call"):
    T = np.maximum(T, 4 / (365 * 24))
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
    return gamma, delta


# Regex to parse OCC Option Symbols (e.g. SPY261106C00580000)
OCC_REGEX = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")


def fetch_cboe_chain(ticker_symbol, num_expiries=4, risk_free_rate=0.042):
    """
    Primary Cloud-Safe Fetcher: Pulls full options chain from CBOE's public CDN.
    Supports both stocks/ETFs (SPY, QQQ, NVDA) and indices (SPX, VIX, NDX).
    """
    clean_sym = ticker_symbol.upper().strip().replace("^", "")
    index_tickers = {"SPX", "VIX", "NDX", "RUT", "DJX", "XSP"}
    cboe_sym = f"_{clean_sym}" if clean_sym in index_tickers else clean_sym

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_sym}.json"
    resp = requests.get(url, headers=headers, timeout=20)

    if resp.status_code != 200 and not cboe_sym.startswith("_"):
        # Retry with underscore prefix in case it's an index symbol
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{clean_sym}.json"
        resp = requests.get(url, headers=headers, timeout=20)

    resp.raise_for_status()
    payload = resp.json().get("data", {})

    spot = float(
        payload.get("current_price")
        or payload.get("close")
        or payload.get("bid")
        or 0
    )
    raw_options = payload.get("options", [])
    if not raw_options or spot <= 0:
        raise ValueError(f"CBOE returned empty options list for {ticker_symbol}")

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

        last_price = float(opt.get("last_trade_price") or 0.0)
        bid = float(opt.get("bid") or 0.0)
        ask = float(opt.get("ask") or 0.0)
        mark_price = last_price if last_price > 0 else ((bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0)

        parsed_rows.append(
            {
                "contractSymbol": sym,
                "type": "call" if cp == "C" else "put",
                "expiration": exp_str,
                "strike": strike,
                "lastPrice": round(mark_price, 2),
                "volume": float(opt.get("volume") or 0.0),
                "openInterest": float(opt.get("open_interest") or 0.0),
                "impliedVolatility": float(opt.get("iv") or 0.0),
                "cboe_gamma": float(opt.get("gamma") or 0.0),
                "cboe_delta": float(opt.get("delta") or 0.0),
            }
        )

    df = pd.DataFrame(parsed_rows)
    if df.empty:
        raise ValueError("No valid near-the-money contracts found in CBOE feed.")

    # Filter to nearest N expiration dates selected by the user
    valid_expiries = sorted(df["expiration"].unique())[:num_expiries]
    df = df[df["expiration"].isin(valid_expiries)].copy()

    now = dt.datetime.now()
    exp_dates = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = np.maximum((exp_dates - now).dt.total_seconds() / (365.0 * 24 * 3600), 1 / (365 * 24))

    # Compute Black-Scholes Greeks as fallback when CBOE gamma is 0 (e.g., over the weekend)
    for opt_type in ["call", "put"]:
        mask = df["type"] == opt_type
        if mask.any():
            bs_gamma, bs_delta = calc_bs_greeks(
                spot,
                df.loc[mask, "strike"].values,
                df.loc[mask, "T"].values,
                risk_free_rate,
                df.loc[mask, "impliedVolatility"].values,
                opt_type,
            )
            df.loc[mask, "gamma"] = np.where(
                df.loc[mask, "cboe_gamma"] > 0, df.loc[mask, "cboe_gamma"], bs_gamma
            )
            df.loc[mask, "delta"] = np.where(
                df.loc[mask, "cboe_delta"] != 0, df.loc[mask, "cboe_delta"], bs_delta
            )

    return spot, df


@st.cache_data(ttl=60)
def fetch_options_and_gex(ticker_symbol, num_expiries=4, risk_free_rate=0.042):
    # 1. Fetch options chain via CBOE CDN (Primary)
    spot, full_df = fetch_cboe_chain(ticker_symbol, num_expiries, risk_free_rate)

    # 2. Fetch 30-day Volume Profile Shelves via yfinance (with safe fallback)
    top_volume_shelves = []
    try:
        yf_sym = "^SPX" if ticker_symbol.upper() == "SPX" else ticker_symbol
        hist = yf.Ticker(yf_sym).history(period="30d", interval="1h")
        if not hist.empty:
            hist["price_bin"] = hist["Close"].round(0)
            vol_profile = (
                hist.groupby("price_bin")["Volume"]
                .sum()
                .reset_index()
                .sort_values("Volume", ascending=False)
            )
            top_volume_shelves = vol_profile.head(3)["price_bin"].tolist()
    except Exception:
        top_volume_shelves = [round(spot, 0)]

    # 3. Compute GEX & Whale Flow Metrics
    sign = np.where(full_df["type"] == "call", 1.0, -1.0)
    full_df["gex_dollars"] = (
        full_df["gamma"] * full_df["openInterest"] * 100 * (spot**2) * 0.01 * sign
    )
    full_df["est_premium"] = full_df["volume"] * full_df["lastPrice"] * 100
    full_df["vol_oi_ratio"] = np.where(
        full_df["openInterest"] > 0, full_df["volume"] / full_df["openInterest"], 0
    )

    gex_by_strike = (
        full_df.groupby(["strike", "type"])["gex_dollars"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["call", "put"]:
        if col not in gex_by_strike:
            gex_by_strike[col] = 0.0
    gex_by_strike["net_gex"] = gex_by_strike["call"] + gex_by_strike["put"]

    call_wall = gex_by_strike.loc[gex_by_strike["call"].idxmax(), "strike"]
    put_wall = gex_by_strike.loc[gex_by_strike["put"].idxmin(), "strike"]

    # Zero Gamma Flip Interpolation
    near_spot = gex_by_strike[
        (gex_by_strike["strike"] >= spot * 0.93)
        & (gex_by_strike["strike"] <= spot * 1.07)
    ].sort_values("strike")
    zero_gamma = spot
    signs = np.sign(near_spot["net_gex"].values)
    strikes = near_spot["strike"].values
    net_vals = near_spot["net_gex"].values
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and (net_vals[i + 1] - net_vals[i]) != 0:
            zero_gamma = strikes[i] - net_vals[i] * (
                strikes[i + 1] - strikes[i]
            ) / (net_vals[i + 1] - net_vals[i])
            break

    call_prem = full_df[full_df["type"] == "call"]["est_premium"].sum()
    put_prem = full_df[full_df["type"] == "put"]["est_premium"].sum()
    total_net_gex = gex_by_strike["net_gex"].sum()

    summary = {
        "spot": round(spot, 2),
        "net_gex_m": round(total_net_gex / 1e6, 2),
        "regime": "POSITIVE GAMMA (Mean-Reverting)"
        if total_net_gex >= 0
        else "NEGATIVE GAMMA (Volatile / Trend)",
        "call_wall": call_wall,
        "put_wall": put_wall,
        "zero_gamma": round(float(zero_gamma), 2),
        "call_prem_m": round(call_prem / 1e6, 2),
        "put_prem_m": round(put_prem / 1e6, 2),
        "vol_shelves": top_volume_shelves,
    }
    return summary, gex_by_strike, full_df


# ==========================================
# 2. FINRA DARK POOL SHORT VOLUME SCRAPER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_finra_darkpool_history(ticker_symbol, lookback_days=12):
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
                    short_vol = float(row["ShortVolume"].iloc[0])
                    short_exempt = float(row["ShortExemptVolume"].iloc[0])
                    total_vol = float(row["TotalVolume"].iloc[0])
                    dp_short_pct = (
                        ((short_vol + short_exempt) / total_vol) * 100
                        if total_vol > 0
                        else 0
                    )
                    records.append(
                        {
                            "Date": check_date.strftime("%Y-%m-%d"),
                            "OffExchange_Short_Vol": int(short_vol + short_exempt),
                            "OffExchange_Total_Vol": int(total_vol),
                            "DarkPool_Short_Pct": round(dp_short_pct, 2),
                        }
                    )
            if len(records) >= 7:
                break
        except Exception:
            continue

    return pd.DataFrame(records).sort_values("Date") if records else pd.DataFrame()


# ==========================================
# 3. STREAMLIT SIDEBAR & UI CONTROLS
# ==========================================
st.sidebar.title("📡 OpenWhale Controls")
ticker = st.sidebar.text_input("Ticker Symbol (e.g. SPY, QQQ, SPX, NVDA)", value="SPY").upper().strip()
num_expiries = st.sidebar.slider("Expirations to Aggregate", 1, 12, 4)
min_vol_oi = st.sidebar.slider("Min Vol/OI Ratio (Whale Filter)", 0.5, 5.0, 1.25, 0.25)
min_premium = st.sidebar.number_input("Min Trade Premium ($)", value=25000, step=25000)
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (60s)", value=False)

st.title(f"📡 {ticker} Institutional Flow, GEX & Dark Pool Terminal")

try:
    with st.spinner(f"Pulling CBOE Options Chain & scanning FINRA TRF for {ticker}..."):
        summary, gex_df, full_chain = fetch_options_and_gex(ticker, num_expiries)
        finra_df = fetch_finra_darkpool_history(ticker)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Spot Price", f"${summary['spot']}")
    c2.metric("Zero Gamma Flip", f"${summary['zero_gamma']}")
    c3.metric("Call Wall (Resistance)", f"${summary['call_wall']}")
    c4.metric("Put Wall (Support)", f"${summary['put_wall']}")
    c5.metric("Net Dealer GEX", f"${summary['net_gex_m']}M")
    c6.metric("Call vs Put Flow", f"${summary['call_prem_m']}M / ${summary['put_prem_m']}M")

    shelves_str = ", ".join([f"${x}" for x in summary["vol_shelves"]]) if summary["vol_shelves"] else "N/A"
    st.info(
        f"**Current Dealer Regime:** {summary['regime']}  |  "
        f"**30-Day High-Volume Institutional Shelves:** {shelves_str}"
    )

    col_left, col_right = st.columns([0.62, 0.38])

    with col_left:
        spot = summary["spot"]
        plot_df = gex_df[
            (gex_df["strike"] >= spot * 0.92) & (gex_df["strike"] <= spot * 1.08)
        ]
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in plot_df["net_gex"]]

        fig_gex = go.Figure()
        fig_gex.add_trace(
            go.Bar(
                x=plot_df["strike"],
                y=plot_df["net_gex"] / 1e6,
                marker_color=colors,
                name="Net GEX ($M)",
            )
        )
        fig_gex.add_vline(
            x=spot, line_dash="dash", line_color="#3b82f6", annotation_text=f"Spot: ${spot}"
        )
        fig_gex.add_vline(
            x=summary["zero_gamma"],
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Zero Gamma: ${summary['zero_gamma']}",
        )
        fig_gex.update_layout(
            title=f"{ticker} Net Gamma Exposure ($M per 1% Move)",
            template="plotly_dark",
            height=420,
            xaxis_title="Strike Price",
            yaxis_title="Net GEX ($ Millions)",
        )
        st.plotly_chart(fig_gex, use_container_width=True)

    with col_right:
        if not finra_df.empty:
            fig_dp = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dp.add_trace(
                go.Bar(
                    x=finra_df["Date"],
                    y=finra_df["OffExchange_Total_Vol"],
                    name="FINRA Off-Exchange Vol",
                    marker_color="#6366f1",
                    opacity=0.6,
                ),
                secondary_y=False,
            )
            fig_dp.add_trace(
                go.Scatter(
                    x=finra_df["Date"],
                    y=finra_df["DarkPool_Short_Pct"],
                    name="Short Vol %",
                    mode="lines+markers",
                    line=dict(color="#f43f5e", width=3),
                ),
                secondary_y=True,
            )
            fig_dp.add_hline(y=50, line_dash="dot", line_color="#94a3b8", secondary_y=True)
            fig_dp.update_layout(
                title=f"{ticker} FINRA RegSHO Off-Exchange Short %",
                template="plotly_dark",
                height=420,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_dp, use_container_width=True)
        else:
            st.warning("No recent FINRA RegSHO records matched this symbol (indices like SPX do not trade on Dark Pools).")

    st.subheader("🔥 Live Unusual Options Flow (New Positioning: Vol > OI)")
    whales = full_chain[
        (full_chain["volume"] >= 100)
        & (full_chain["vol_oi_ratio"] >= min_vol_oi)
        & (full_chain["est_premium"] >= min_premium)
    ].sort_values("est_premium", ascending=False)

    display_cols = [
        "contractSymbol",
        "type",
        "expiration",
        "strike",
        "lastPrice",
        "volume",
        "openInterest",
        "vol_oi_ratio",
        "impliedVolatility",
        "est_premium",
    ]
    st.dataframe(
        whales[display_cols].head(25),
        use_container_width=True,
        hide_index=True,
    )

except Exception as e:
    st.error(f"Error loading data for {ticker}: {e}")

if auto_refresh:
    time.sleep(60)
    st.rerun()
