"""Archer.ai: a Streamlit dashboard for the arrows and the archer.

Run with:  streamlit run app.py

Everything except Jev's decision is free and needs no key. Visitors who want
Jev's call paste their own OpenRouter or TypeSafe key. The app never reads a
key from the environment or from Streamlit secrets, and never passes one to a
cached function, because both are shared by every visitor of a hosted app.
"""

import backtest  # noqa: F401  (sets single-threaded BLAS before numpy loads)

import re
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from archer import build_agent, jev_model, judge
from arrows.real_yield import RealYield, fetch_real_yield
from backtest import scorecard, track_record_text, walk_forward
from decision_backtest import compare, run_decisions, simulate, strategy_weights
from quiver import briefing, default_arrows, draw_all
from regime import fetch_prices, fit_regimes

st.set_page_config(page_title="Archer.ai", layout="wide")

TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,15}$")
# Solarized accents, matching .streamlit/config.toml
REGIME_COLORS = {"Calm": "#859900", "Normal": "#b58900",
                 "Elevated": "#cb4b16", "Turbulent": "#dc322f"}
PRICE_LINE = "#93a1a1"
STANCE_ICON = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}
VOL_ICON = {"low": "🟢", "normal": "🟡", "high": "🔴"}
ARROW_TITLES = {
    "hmm_regime": "Volatility regime (HMM)",
    "garch": "Volatility forecast (GARCH)",
    "trend": "Trend",
    "mean_reversion": "Mean reversion",
    "real_yield": "Real yields (gold only)",
}
PROVIDERS = {"OpenRouter": "openrouter", "TypeSafe": "typesafe"}

HORIZON_OPTIONS = [None, 5, 10, 20, 60, 120, 250]
HORIZON_LABELS = {None: "5–60 days by model", 5: "about a week", 10: "about two weeks",
                  20: "about a month", 60: "about a quarter", 120: "about six months",
                  250: "about a year"}
STEP_OPTIONS = [5, 10, 20, 60]
STEP_LABELS = {5: "weekly", 10: "every two weeks", 20: "monthly", 60: "quarterly"}
HORIZON_HELP = (
    "**Horizon** is how far ahead each model's call is judged. A 20-day horizon asks: "
    "was the call right about the next month? By default each model uses its own "
    "(mean reversion 5 days, trend and volatility 20, real yields 60). Volatility "
    "forecasts change with the horizon; trend and mean-reversion signals are computed "
    "the same way and are simply judged over a different period."
)
STEP_HELP = (
    "**Step size** is how often the backtest stops to refit the models and make a call, "
    "and how often Jev revisits its decision. Smaller steps give more calls to judge "
    "but take longer to run. When the horizon is longer than the step, neighbouring "
    "calls overlap, and the statistics count them for less."
)
KEY_LINKS = {"OpenRouter": "https://openrouter.ai/keys", "TypeSafe": "https://docs.typesafe.ai"}


# ---- Cached computations. None of these ever receive an API key. ----

@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def load_prices(ticker, start):
    return fetch_prices(ticker, start)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_real_yield():
    return fetch_real_yield()


def arrows_for(ticker, horizon):
    """Default arrows, with real yields fetched once and shared."""
    return [RealYield(real_yield=load_real_yield(), horizon_days=a.horizon_days)
            if isinstance(a, RealYield) else a
            for a in default_arrows(ticker, horizon)]


@st.cache_data(ttl=3600, show_spinner="Running the models…")
def load_signals(ticker, start, horizon):
    return draw_all(load_prices(ticker, start), arrows_for(ticker, horizon))


@st.cache_resource(show_spinner="Fitting regime model…")
def load_regimes(ticker, start, n_states):
    return fit_regimes(load_prices(ticker, start), n_states=n_states)


@st.cache_data(show_spinner=False, max_entries=20)
def load_records(ticker, start, day, horizon, step, _progress=None):
    """Walk-forward records. `day` keys the cache so it refreshes daily."""
    prices = load_prices(ticker, start)
    return walk_forward(prices, arrows_for(ticker, horizon), step=step, on_progress=_progress)


def redact(message, secret):
    return message.replace(secret, "***") if secret else message


# ---- Sidebar: the visitor's own key, and settings. ----

with st.sidebar:
    st.header("Jev (optional)")
    provider = st.radio("Your key is from", list(PROVIDERS), horizontal=True)
    api_key = st.text_input("API key", type="password",
                            placeholder="sk-or-…" if provider == "OpenRouter" else "")
    st.caption(
        f"Only needed for Jev's decision. Your key is sent to {provider} with your own "
        f"requests in this session, and isn't stored, logged or shared. This app has no "
        f"key of its own. [Get a key]({KEY_LINKS[provider]})"
    )
    st.header("Backtest settings")
    with st.popover("What do these mean?", icon=":material/help:", width="stretch"):
        st.markdown(HORIZON_HELP)
        st.markdown(STEP_HELP)
    horizon_choice = st.select_slider(
        "Horizon (trading days ahead)", options=HORIZON_OPTIONS, value=None,
        format_func=lambda h: "Each model's default" if h is None else f"{h} days",
        help=HORIZON_HELP)
    step = st.select_slider(
        "Step size (trading days between checks)", options=STEP_OPTIONS, value=20,
        format_func=lambda d: f"{d} days", help=STEP_HELP)
    st.caption(f"Horizon: {HORIZON_LABELS.get(horizon_choice, '')} · "
               f"step: {STEP_LABELS.get(step, '')}")

    with st.expander("Advanced"):
        start_year = st.slider("History from", 2005, 2022, 2010)
        n_states = st.radio("Regimes in the HMM view", [2, 3, 4], index=1, horizontal=True)
        jev_version = st.text_input("Jev version", "jev-latest")
        min_conf = st.slider("Escalate below this confidence", 0.5, 0.95, 0.7, 0.05)

start = f"{start_year}-01-01"
api_key = api_key.strip()
via = PROVIDERS[provider]

# ---- Ticker ----

st.title("Archer.ai")
st.caption("Five quantitative models read one asset; an optional decision model weighs them. "
           "Educational, not financial advice.")

with st.form("ticker_form", border=False):
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    raw = c1.text_input("Ticker (Yahoo Finance symbol)", "GLD", max_chars=15)
    c2.form_submit_button("Analyze", width="stretch")
ticker = raw.strip().upper()
if not TICKER_RE.match(ticker):
    st.error("Enter a ticker like GLD, SPY, GC=F or BTC-USD.")
    st.stop()

try:
    prices = load_prices(ticker, start)
    signals = load_signals(ticker, start, horizon_choice)
except Exception as e:
    st.error(f"Couldn't load {ticker}: {e}")
    st.stop()
if not signals:
    st.error(f"None of the models could run on {ticker}. It may have too little history.")
    st.stop()

records_key = ("records", ticker, start, date.today(), horizon_choice, step)
records = st.session_state.get(records_key)
card = scorecard(records, min_calls=20) if records is not None else None

today_tab, regime_tab, record_tab = st.tabs(["Today", "Regimes", "Track record"])

# ---- Today ----

with today_tab:
    as_of = max(s.as_of for s in signals)
    st.subheader(f"{ticker} · {prices.iloc[-1]:,.2f}")
    st.caption(f"Data through {as_of}")

    text = briefing(ticker, signals, card)
    jev_key = ("jev", ticker, start, str(as_of), jev_version, horizon_choice, card is not None)

    with st.container(border=True):
        st.markdown("#### Jev's call")
        if not api_key:
            st.info("Paste your OpenRouter or TypeSafe key in the sidebar to have Jev "
                    "weigh these models and make a call.")
        else:
            if card is None:
                st.caption("Tip: run the backtest on the Track record tab first, so Jev "
                           "knows which models have earned trust.")
            if st.button("Ask Jev", type="primary"):
                try:
                    with st.spinner("Asking Jev…"):
                        agent = build_agent(jev_model(jev_version, api_key=api_key, via=via))
                        st.session_state[jev_key] = judge(agent, ticker, text, min_conf)
                except Exception as e:
                    st.error(f"Jev request failed: {redact(f'{type(e).__name__}: {e}', api_key)}")
            call = st.session_state.get(jev_key)
            if call:
                c = call.confidence
                a, b, d = st.columns(3)
                a.metric("Action", "Escalate" if call.escalate else call.decision.action.title(),
                         f"{c.get('action', 0):.0%} confident", delta_color="off")
                b.metric("Risk", call.decision.risk.title(),
                         f"{c.get('risk', 0):.0%} confident", delta_color="off")
                d.metric("Signals conflict", "Yes" if call.decision.signals_conflict else "No",
                         f"{c.get('signals_conflict', 0):.0%} confident", delta_color="off")
                if call.escalate:
                    st.warning(f"Jev leans '{call.decision.action}' but is below your "
                               f"{call.min_confidence:.0%} bar, so treat this as undecided.")
        with st.expander("Exactly what Jev reads"):
            st.code(text, language=None, wrap_lines=True)

    st.markdown("#### The models")
    cols = st.columns(2)
    for i, s in enumerate(signals):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{ARROW_TITLES.get(s.model, s.model)}** · {s.horizon_days}-day view")
            tags = []
            if s.stance:
                tags.append(f"{STANCE_ICON[s.stance]} {s.stance}")
            if s.vol_level:
                tags.append(f"{VOL_ICON[s.vol_level]} {s.vol_level} volatility")
            tags.append(f"strength {s.strength:.0%}")
            st.markdown(" · ".join(tags))
            st.write(s.rationale)
            if card and s.model in card:
                st.caption(f"Track record: {track_record_text(card[s.model])}")

# ---- Regimes ----

with regime_tab:
    rm = load_regimes(ticker, start, n_states)
    s = rm.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current regime", s["regime"])
    c2.metric("Model confidence", f"{s['confidence']:.0%}")
    c3.metric("Days in this regime", s["days_in_regime"])
    c4.metric("Typical length", f"~{s['expected_duration']:.0f} days")

    stats = rm.regime_stats()
    df = rm.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode="lines",
                             line=dict(color=PRICE_LINE, width=1), name="Price", hoverinfo="skip"))
    for name in rm.names:
        sub = df[df["Label"] == name]
        fig.add_trace(go.Scatter(x=sub.index, y=sub["Price"], mode="markers",
                                 marker=dict(size=3, color=REGIME_COLORS.get(name)), name=name))
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.05), hovermode="x")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Outlook** · chance of each regime this many trading days ahead")
        st.dataframe(rm.outlook().style.format("{:.0%}"), width="stretch")
    with right:
        st.markdown("**How each regime has behaved**")
        st.dataframe(stats.style.format({
            "Share of days": "{:.0%}", "Avg daily move (%)": "{:+.2f}",
            "Annualised volatility (%)": "{:.1f}", "Worst day (%)": "{:+.1f}",
            "Best day (%)": "{:+.1f}", "Typical length (days)": "{:.0f}",
        }), width="stretch")
    st.caption("Today's regime uses only data up to today. The colouring of past days uses "
               "the full history, so it looks cleaner than it would have in real time.")

# ---- Track record ----

with record_tab:
    st.markdown(
        f"Every {step} trading days since the history starts (after a 3-year warm-up), each model is "
        "refit on prices up to that day only, and its call is checked against what happened next. "
        "A model is only called better or worse than its simple baseline when the difference is "
        "statistically clear."
    )
    if records is None:
        n_dates = max((len(prices) - 3 * 252) // step, 0)
        minutes = max(round(n_dates * len(signals) * 0.12 / 60), 1)
        if st.button(f"Run the backtest ({n_dates} checks, about {minutes} min)", type="primary"):
            bar = st.progress(0.0, text="Replaying history…")
            records = load_records(ticker, start, date.today(), horizon_choice, step,
                                   _progress=lambda f: bar.progress(f, text="Replaying history…"))
            bar.empty()
            st.session_state[records_key] = records
            st.rerun()
        st.stop()

    rows = []
    for model, entry in card.items():
        kind = "direction" if "direction" in entry else "volatility"
        stats = entry[kind]
        rows.append({
            "Model": ARROW_TITLES.get(model, model),
            "Predicts": kind,
            "Calls": stats["calls"],
            "Verdict vs baseline": stats["verdict"].replace(" from", "").replace(" than", ""),
            "Detail": (f"right {stats['hit_rate']:.0%} vs {stats['always_bullish_hit_rate']:.0%} "
                       "for always-bullish" if kind == "direction" else
                       f"error {stats['vol_mae']:.1f} vs {stats['naive_vol_mae']:.1f} pts for "
                       "'next month = last month'"),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    st.markdown("#### Position-sizing strategies")
    st.caption(f"Starting at 75% in the asset, rest in cash earning 0%. Positions are reviewed "
               f"every {step} trading days; Jev's strategy moves 25 points per increase or "
               f"reduce call.")
    decisions_key = ("decisions", ticker, start, date.today(), horizon_choice, step,
                     jev_version, min_conf)
    decisions = st.session_state.get(decisions_key)

    if api_key and decisions is None:
        n_calls = records["date"].nunique()
        if st.button(f"Backtest Jev's decisions ({n_calls} calls on your key)"):
            try:
                with st.spinner(f"Asking Jev {n_calls} times…"):
                    agent = build_agent(jev_model(jev_version, api_key=api_key, via=via))
                    st.session_state[decisions_key] = run_decisions(ticker, records, agent, min_conf)
                st.rerun()
            except Exception as e:
                st.error(f"Jev request failed: {redact(f'{type(e).__name__}: {e}', api_key)}")
    elif not api_key:
        st.caption("Add your key in the sidebar to backtest Jev's decisions against these.")

    st.dataframe(compare(prices, records, decisions).style.format({
        "Annual return": "{:.1%}", "Annual volatility": "{:.1%}", "Return / volatility": "{:.2f}",
        "Max drawdown": "{:.1%}", "Average weight": "{:.0%}", "Trades": "{:.0f}",
    }), width="stretch")

    fig = go.Figure()
    for name, w in strategy_weights(records, decisions).items():
        equity = (1 + simulate(prices, w)).cumprod()
        fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name=name))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Growth of $1",
                      legend=dict(orientation="h", y=1.05), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    if decisions is not None:
        st.caption(f"Jev's actions: {decisions['action'].value_counts().to_dict()}, "
                   f"escalated {decisions['escalate'].mean():.0%} of the time.")
    st.caption("Past performance says little about the future. Not financial advice.")
