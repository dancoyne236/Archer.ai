# Archer.ai

[![tests](https://github.com/dancoyne236/Archer.ai/actions/workflows/tests.yml/badge.svg)](https://github.com/dancoyne236/Archer.ai/actions/workflows/tests.yml)

A quiver of five quantitative models (the "arrows") reads one asset, and the archer, [TypeSafe's Jev](https://typesafe.ai) decision model, weighs them into a typed call: **increase, hold or reduce**, with a confidence. Every model is scored walk-forward against a simple baseline, so you can see which ones have actually earned trust.

Built around gold, but it works with any Yahoo Finance ticker.

> Educational project. Not financial advice.

## What it does

| Arrow | Question it answers | Horizon |
|---|---|---|
| HMM regime | Which volatility regime is the market in, and how long will it last? | 20 days |
| GARCH(1,1) | How volatile will the next month be? | 20 days |
| Trend | Is the price trending up or down? (12-1 momentum, 200-day average, 50/200 cross) | 20 days |
| Mean reversion | Has the price stretched too far, too fast? | 5 days |
| Real yields | Is gold cheap or rich relative to 10-year TIPS yields? (gold tickers only) | 60 days |

Each arrow returns the same typed `Signal` (see [`arrows/base.py`](arrows/base.py)): a categorical stance or volatility level, a strength, and a plain-English rationale. The categories matter: Jev reasons over text and typed options and is unreliable at arithmetic, so each model does its own maths and hands over conclusions.

**Track records.** [`backtest.py`](backtest.py) refits every arrow on data up to each date, checks its call against what happened next, and only calls a model better or worse than its baseline when the difference is statistically clear (adjusted for overlapping horizons). Those verdicts go into the briefing Jev reads.

**Decision backtest.** [`decision_backtest.py`](decision_backtest.py) replays history, showing Jev only what it would have known on each date (including track records built from already-resolved calls), and compares its position changes with buy-and-hold and a GARCH volatility-targeting rule.

## Run the app

```bash
git clone https://github.com/dancoyne236/Archer.ai.git
cd Archer.ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter a ticker and press **Analyze**. Everything except Jev's call is free and needs no key: prices come from Yahoo Finance and real yields from FRED.

### Your API key

Jev's decision is optional. To get it, paste your own key in the sidebar:

- **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys) (Jev is routed through OpenRouter's TypeSafe integration), or
- **TypeSafe**: [docs.typesafe.ai](https://docs.typesafe.ai)

How the app treats your key:

- It is sent only to the provider you chose, only for requests you trigger, and lives only in your browser session.
- It is never stored, logged, cached or shared with other visitors.
- **The app never reads a key from its own environment or secrets**, so a hosted copy can't spend its owner's credits, and a visitor can't spend anyone else's. This is enforced by [`tests/test_app.py`](tests/test_app.py).

Jev costs about $0.04 per million input tokens with free output. One call reads about 1,000 tokens, and a full decision backtest makes about 170 calls.

## Command line

```bash
python markov.py GLD                  # regime report and chart
python quiver.py GLD                  # the briefing Jev reads (add --json for typed signals)
python backtest.py GLD                # score each arrow, ~2 minutes
export OPENROUTER_API_KEY=...         # or TYPESAFE_API_KEY
python archer.py GLD --show-briefing  # one Jev call
python decision_backtest.py GLD       # Jev vs buy-and-hold vs vol targeting
python decision_backtest.py GLD --baselines-only   # no key needed
```

The command-line tools do read keys from the environment, since they run on your own machine.

## Findings so far (GLD, 2013–2026)

- **Only GARCH beat its baseline**, with volatility forecasts about 9% more accurate than assuming next month looks like last month.
- No directional model (trend, mean reversion, real yields) showed a statistically reliable edge.
- A simple GARCH volatility-targeting rule beat buy-and-hold on return per unit of volatility (0.50 against 0.41), with a similar max drawdown.

That is the bar for Jev: if it can't beat volatility targeting, it isn't adding anything.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests run offline on synthetic data. They cover each arrow, the scoring statistics, no look-ahead in either backtest, the Jev mapping (against a stand-in decision model that speaks the same protocol), and the app's key handling.

## Project layout

```
app.py                 Streamlit dashboard
arrows/                the five models and the shared Signal contract
regime.py              HMM regime model (used by the app and the HMM arrow)
quiver.py              run all arrows, render the briefing
backtest.py            walk-forward scoring of each arrow
archer.py              Jev integration: briefing in, typed Decision out
decision_backtest.py   replay Jev's decisions against baselines
markov.py              original command-line regime report
tests/
```

## License

MIT
