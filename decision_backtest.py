"""Backtest the archer's decisions against simple position-sizing rules.

At each scoring date Jev sees the briefing it would have seen then: the
arrows fitted on data up to that day, with track records built only from
calls whose outcomes were already known. Its actions move a position in
steps, and the resulting portfolio is compared with:

- buy and hold: a constant weight in the asset
- vol target: scale the weight down when GARCH (the arrow with a proven
  volatility edge) forecasts high volatility

Money not in the asset sits in cash earning nothing, which flatters no one:
it understates every strategy that holds cash, the archer's included.

Usage:
    python decision_backtest.py --baselines-only     # no API key needed
    python decision_backtest.py GLD                  # needs TYPESAFE_API_KEY or OPENROUTER_API_KEY
"""

import backtest  # noqa: F401  (sets single-threaded BLAS before numpy loads)

import argparse
import logging

import numpy as np
import pandas as pd

from archer import MISSING_KEY, build_agent, has_jev_key, jev_model, judge
from arrows.real_yield import RealYield, fetch_real_yield
from arrows.base import TRADING_DAYS
from backtest import RESULTS_DIR, scorecard, walk_forward
from quiver import briefing, default_arrows
from regime import fetch_prices

log = logging.getLogger(__name__)

START_WEIGHT = 0.75
WEIGHT_STEP = 0.25
MIN_TRACK_RECORD = 20  # resolved calls before an arrow's track record is shown


def run_decisions(ticker, records, agent, min_confidence=0.7):
    """One archer call per scoring date, using only what was known that day."""
    rows = []
    for date, group in records.groupby("date"):
        card = scorecard(records, known_by=date, min_calls=MIN_TRACK_RECORD)
        text = briefing(ticker, list(group["signal"]), card)
        call = judge(agent, ticker, text, min_confidence)
        rows.append({
            "date": date,
            "action": call.decision.action,
            "confidence": call.confidence.get("action", np.nan),
            "escalate": call.escalate,
            "risk": call.decision.risk,
            "signals_conflict": call.decision.signals_conflict,
        })
        log.info("%s: %s", date.date(), call.decision.action)
    return pd.DataFrame(rows).set_index("date")


def weights_from_actions(decisions, start=START_WEIGHT, step=WEIGHT_STEP):
    """Move the weight a step per call; escalated calls are treated as hold."""
    w, out = start, []
    for action, escalate in zip(decisions["action"], decisions["escalate"]):
        if not escalate:
            w += {"increase": step, "reduce": -step}.get(action, 0.0)
            w = min(max(w, 0.0), 1.0)
        out.append(w)
    return pd.Series(out, index=decisions.index, name="weight")


def vol_target_weights(records, start=START_WEIGHT):
    """Hold less when GARCH forecasts above-typical volatility. The 'typical'
    level is the median of forecasts made so far, so there's no look-ahead."""
    fc = records.loc[records["model"] == "garch"].set_index("date")["vol_forecast"]
    typical = fc.expanding().median()
    return (start * typical / fc).clip(0, 1).rename("weight")


def simulate(prices, weights):
    """Daily portfolio returns, holding each weight from the day after it is set."""
    rets = prices.pct_change()
    w = weights.reindex(prices.index).ffill().shift(1)
    daily = (w * rets).loc[weights.index[0]:].iloc[1:]
    return daily.fillna(0.0)


def summarize(daily, weights):
    equity = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    ann = equity.iloc[-1] ** (1 / years) - 1
    return {
        "Annual return": ann,
        "Annual volatility": vol,
        "Return / volatility": ann / vol if vol else np.nan,
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Average weight": weights.mean(),
        "Trades": int((weights.diff().abs() > 1e-9).sum()),
    }


def strategy_weights(records, decisions=None):
    """Position weights per strategy: baselines, plus the archer when decisions are given."""
    dates = records["date"].drop_duplicates().sort_values()
    strategies = {
        "Buy and hold": pd.Series(START_WEIGHT, index=dates),
        "Vol target (GARCH)": vol_target_weights(records),
    }
    if decisions is not None:
        strategies["Archer (Jev)"] = weights_from_actions(decisions)
    return strategies


def compare(prices, records, decisions=None):
    """Summary table, one row per strategy."""
    return pd.DataFrame({name: summarize(simulate(prices, w), w)
                         for name, w in strategy_weights(records, decisions).items()}).T


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="GLD")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--model", default="jev-latest", help="Jev version, e.g. jev-1.13")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    use_archer = not args.baselines_only
    if use_archer and not has_jev_key():
        parser.error(f"{MISSING_KEY}, or pass --baselines-only")

    prices = fetch_prices(args.ticker, args.start)
    arrows = [RealYield(real_yield=fetch_real_yield()) if isinstance(a, RealYield) else a
              for a in default_arrows(args.ticker)]
    records = walk_forward(prices, arrows, step=args.step)

    decisions = None
    if use_archer:
        decisions = run_decisions(args.ticker, records, build_agent(jev_model(args.model)), args.min_confidence)
        RESULTS_DIR.mkdir(exist_ok=True)
        decisions.to_csv(RESULTS_DIR / f"{args.ticker.upper()}_decisions.csv")
        print("Actions:", decisions["action"].value_counts().to_dict(),
              f"| escalated {decisions['escalate'].mean():.0%}")

    table = compare(prices, records, decisions)
    first = records["date"].min().date()
    print(f"\n{args.ticker}, decisions every {args.step} trading days from {first}\n")
    print(table.to_string(formatters={
        "Annual return": "{:.1%}".format, "Annual volatility": "{:.1%}".format,
        "Return / volatility": "{:.2f}".format, "Max drawdown": "{:.1%}".format,
        "Average weight": "{:.0%}".format, "Trades": "{:.0f}".format,
    }))
    print("\nCash earns 0%. Past performance says little about the future. Not financial advice.")


if __name__ == "__main__":
    main()
