import datetime as dt
import io
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
    page_title="OpenWhale - Dealer GEX Terminal",
    page_icon="📡", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 1. BLACK-SCHOLES GREEKS & GEX ENGINE
# ==========================================
def calc_bs_greeks(S, K, T, r, sigma, option_type="call"):
    T = np.maximum(T, 4 / (365 * 24))
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
    return gamma, delta


@st.cache_data(ttl=60)
def fetch_options_and_gex(ticker_symbol, num_expiries=4, risk_free_rate=0.042):
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="30d", interval="1h")
    if hist.empty:
        raise ValueError(f"Could not fetch price history for {ticker_symbol}")
    spot = float(hist["Close"].iloc[-1])

    # Approximate Institutional Price Shelves (High Volume Nodes over 30 days)
    hist["price_bin"] = hist["Close"].round(0)
    vol_profile = (
        hist.groupby("price_bin")["Volume"]
        .sum()
        .reset_index()
        .sort_values("Volume", ascending=False)
    )
    top_volume_shelves = vol_profile.head(3)["price_bin"].tolist()

    expirations = tk.options[:num_expiries]
    now = dt.datetime.now()
    all_contracts = []

    for exp_str in expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d") + dt.timedelta(hours=16)
        T = max((exp_date - now).total_seconds() / (365.0 * 24 * 3600), 1 / (365 * 24))
        chain = tk.option_chain(exp_str)

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["type"] = opt_type
            df["expiration"] = exp_str
            df["T"] = T
            df["volume"] = df["volume"].fillna(0)
            df["openInterest"] = df["openInterest"].fillna(0)
            df["impliedVolatility"] = df["impliedVolatility"].fillna(0)
            df = df[(df["strike"] >= spot * 0.80) & (df["strike"] <= spot * 1.20)]

            gamma, delta = calc_bs_greeks(
                spot,
                df["strike"].values,
                df["T"].values,
                risk_free_rate,
                df["impliedVolatility"].values,
                opt_type,
            )
            df["gamma"] = gamma
            df["delta"] = delta

            sign = 1.0 if opt_type == "call" else -1.0
            df["gex_dollars"] = (
                df["gamma"] * df["openInterest"] * 100 * (spot**2) * 0.01 * sign
            )
            df["est_premium"] = df["volume"] * df["lastPrice"] * 100
            df["vol_oi_ratio"] = np.where(
                df["openInterest"] > 0, df["volume"] / df["openInterest"], 0
            )
            all_contracts.append(df)

    full_df = pd.concat(all_contracts, ignore_index=True)

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
    """
    Pulls daily off-exchange (TRF/ADF) Short vs Total Volume directly from
    FINRA's public RegSHO consolidated text files (CNMSshvolYYYYMMDD.txt).
    """
    records = []
    today = dt.date.today()
    headers = {"User-Agent": "Mozilla/5.0"}

    for offset in range(1, lookback_days + 5):
        check_date = today - dt.timedelta(days=offset)
        if check_date.weekday() >= 5:  # Skip weekends
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
st.sidebar.title("🐋 Whale & GEX Controls")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper().strip()
num_expiries = st.sidebar.slider("Expirations to Aggregate", 1, 8, 4)
min_vol_oi = st.sidebar.slider("Min Vol/OI Ratio (Whale Filter)", 1.0, 5.0, 1.5, 0.25)
min_premium = st.sidebar.number_input("Min Trade Premium ($)", value=50000, step=25000)
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh (60s)", value=False)

st.title(f"📊 {ticker} Institutional Flow, GEX & Dark Pool Terminal")

try:
    with st.spinner(f"Computing Black-Scholes Greeks & scanning FINRA TRF for {ticker}..."):
        summary, gex_df, full_chain = fetch_options_and_gex(ticker, num_expiries)
        finra_df = fetch_finra_darkpool_history(ticker)

    # Top Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Spot Price", f"${summary['spot']}")
    c2.metric("Zero Gamma Flip", f"${summary['zero_gamma']}")
    c3.metric("Call Wall (Resistance)", f"${summary['call_wall']}")
    c4.metric("Put Wall (Support)", f"${summary['put_wall']}")
    c5.metric("Net Dealer GEX", f"${summary['net_gex_m']}M")
    c6.metric(
        "Call vs Put Flow",
        f"${summary['call_prem_m']}M / ${summary['put_prem_m']}M",
    )

    st.info(
        f"**Current Dealer Regime:** {summary['regime']}  |  "
        f"**30-Day High-Volume Institutional Shelves:** {', '.join([f'${x}' for x in summary['vol_shelves']])}"
    )

    # ==========================================
    # 4. CHARTS: GEX PROFILE & FINRA DARK POOL
    # ==========================================
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
            fig_dp.add_hline(
                y=50, line_dash="dot", line_color="#94a3b8", secondary_y=True
            )
            fig_dp.update_layout(
                title=f"{ticker} FINRA RegSHO Off-Exchange Short %",
                template="plotly_dark",
                height=420,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_dp, use_container_width=True)
        else:
            st.warning("No recent FINRA RegSHO records matched this symbol.")

    # ==========================================
    # 5. UNUSUAL WHALES FLOW TABLE
    # ==========================================
    st.subheader("🔥 Live Unusual Options Flow (New Positioning: Vol > OI)")
    whales = full_chain[
        (full_chain["volume"] >= 200)
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
