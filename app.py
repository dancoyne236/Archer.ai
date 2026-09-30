"""Streamlit dashboard for market regimes.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regime import fetch_prices, fit_regimes

COLORS = {"Calm": "#2e7d32", "Normal": "#f9a825",
          "Elevated": "#ef6c00", "Turbulent": "#c62828"}
PRESETS = {
    "GLD (gold ETF)": "GLD",
    "GC=F (gold futures)": "GC=F",
    "IAU (gold ETF)": "IAU",
    "SLV (silver ETF)": "SLV",
    "SPY (S&P 500)": "SPY",
}

st.set_page_config(page_title="Gold Regime Tracker", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def load_prices(ticker, start):
    return fetch_prices(ticker, start)


@st.cache_resource(show_spinner="Fitting regime model…")
def load_model(ticker, start, n_states):
    return fit_regimes(load_prices(ticker, start), n_states=n_states)


with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Asset", [*PRESETS, "Other…"])
    ticker = (st.text_input("Yahoo Finance ticker", "GLD").strip().upper()
              if choice == "Other…" else PRESETS[choice])
    start_year = st.slider("History from", 2005, 2024, 2010)
    n_states = st.radio("Number of regimes", [2, 3, 4], index=1, horizontal=True)

try:
    rm = load_model(ticker, f"{start_year}-01-01", n_states)
except Exception as e:
    st.error(f"Couldn't load {ticker}: {e}")
    st.stop()

s = rm.summary()
st.title(f"{ticker} market regime")
st.caption(f"Data through {s['as_of']} · last close {s['price']:,.2f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current regime", s["regime"])
c2.metric("Model confidence", f"{s['confidence']:.0%}")
c3.metric("Days in this regime", s["days_in_regime"])
c4.metric("Typical length", f"~{s['expected_duration']:.0f} days")

stats = rm.regime_stats()
vol = stats.loc[s["regime"], "Annualised volatility (%)"]
st.info(
    f"**{s['regime']}** has historically meant roughly **{vol:.0f}% annualised "
    f"volatility**, i.e. a typical daily swing of about ±{vol / 252 ** 0.5:.1f}%. "
    "Regimes describe how bumpy the ride is, not which way it's heading."
)

# Price chart coloured by regime
df = rm.df
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode="lines",
                         line=dict(color="lightgray", width=1),
                         name="Price", hoverinfo="skip"))
for name in rm.names:
    sub = df[df["Label"] == name]
    fig.add_trace(go.Scatter(x=sub.index, y=sub["Price"], mode="markers",
                             marker=dict(size=3, color=COLORS.get(name)), name=name))
fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                  legend=dict(orientation="h", y=1.05), hovermode="x")
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Outlook")
    st.caption("Chance of being in each regime this many trading days from now.")
    st.dataframe(rm.outlook().style.format("{:.0%}"), width="stretch")
with right:
    st.subheader("How each regime has behaved")
    st.dataframe(
        stats.style.format({
            "Share of days": "{:.0%}",
            "Avg daily move (%)": "{:+.2f}",
            "Annualised volatility (%)": "{:.1f}",
            "Worst day (%)": "{:+.1f}",
            "Best day (%)": "{:+.1f}",
            "Typical length (days)": "{:.0f}",
        }),
        width="stretch",
    )

with st.expander("How this works"):
    st.markdown(
        "A hidden Markov model groups daily returns into regimes with different "
        "volatility, then learns how often the market switches between them. "
        "Today's regime uses only data up to today. Historical colouring uses the "
        "full sample, so past regimes look cleaner than they would have in real time."
    )
    st.write("Transition probabilities (row = today, column = tomorrow):")
    st.dataframe(
        pd.DataFrame(rm.transmat, index=rm.names, columns=rm.names)
        .style.format("{:.1%}"),
        width="stretch",
    )

st.caption("Educational tool, not financial advice.")
