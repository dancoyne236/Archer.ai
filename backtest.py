"""Walk-forward scoring for the arrows.

At each scoring date every arrow is refit on prices up to that day only, its
Signal is recorded, and later compared with what actually happened over its
horizon. The result is a track record per arrow, judged against simple
baselines, plus a one-line verdict a decision model can read.

Usage:
    python backtest.py              # GLD, score every 20 trading days
    python backtest.py SPY --step 10
"""

import os

# Thousands of small model fits run faster without BLAS thread contention.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from arrows.base import TRADING_DAYS, daily_returns
from arrows.real_yield import RealYield, fetch_real_yield
from quiver import default_arrows
from regime import fetch_prices

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def walk_forward(prices, arrows, warmup=3 * TRADING_DAYS, step=20, on_progress=None):
    """Refit each arrow at every `step`-th day after `warmup` and record its signal
    alongside what happened next. `resolved` is the day the outcome became known;
    the Signal itself is kept in `signal` so briefings can be rebuilt later.
    `on_progress`, if given, is called with the fraction done after each date."""
    rets = daily_returns(prices).reindex(prices.index)
    rows = []
    dates = range(warmup, len(prices), step)
    for n, i in enumerate(dates, 1):
        history = prices.iloc[: i + 1]
        for arrow in arrows:
            try:
                sig = arrow.fit(history).signal()
            except Exception:
                log.exception("Arrow %s failed at %s", arrow.name, prices.index[i].date())
                continue
            h = sig.horizon_days
            row = {
                "model": sig.model, "date": prices.index[i], "horizon": h,
                "stance": sig.stance, "direction": sig.direction,
                "vol_level": sig.vol_level, "vol_forecast": sig.vol_forecast,
                "strength": sig.strength, "step": step,
                # Naive benchmark: next month looks like last month.
                "naive_vol": rets.iloc[i - 19: i + 1].std() * np.sqrt(TRADING_DAYS),
                "fwd_return": np.nan, "fwd_vol": np.nan, "resolved": pd.NaT,
                "signal": sig,
            }
            if i + h < len(prices):
                row["resolved"] = prices.index[i + h]
                row["fwd_return"] = (prices.iloc[i + h] / prices.iloc[i] - 1) * 100
                row["fwd_vol"] = rets.iloc[i + 1: i + h + 1].std() * np.sqrt(TRADING_DAYS)
            rows.append(row)
        log.info("Scored %s", prices.index[i].date())
        if on_progress:
            on_progress(n / len(dates))
    return pd.DataFrame(rows)


MIN_BUCKET = 30  # fewest calls before a strength bucket is quoted to the agent


def _verdict(z):
    """Word the comparison with a baseline, requiring about 2 standard errors."""
    return "better than" if z >= 2 else "worse than" if z <= -2 else "not reliably different from"


def _effective_n(df, n):
    """Overlapping horizons share outcomes, so they count for less."""
    return n * min(1.0, df["step"].iloc[0] / df["horizon"].iloc[0])


def _strength_bucket(s):
    return pd.cut(s, [0, 1 / 3, 2 / 3, 1.0001], right=False, labels=["weak", "moderate", "strong"])


def score_direction(df):
    df = df.dropna(subset=["fwd_return", "direction"])
    sided = df[df["stance"] != "neutral"]
    correct = np.sign(sided["fwd_return"]) == np.where(sided["stance"] == "bullish", 1, -1)
    up_rate = float((df["fwd_return"] > 0).mean())
    hit = float(correct.mean()) if len(sided) else np.nan
    buckets = correct.groupby(_strength_bucket(sided["strength"]), observed=True)
    by_stance = df.groupby("stance")["fwd_return"].mean()
    # Hit rate vs. the always-bullish baseline, as a z-score.
    def z_vs_up_rate(rate, n):
        se = np.sqrt(up_rate * (1 - up_rate) / max(_effective_n(df, n), 1))
        return (rate - up_rate) / se if n else np.nan

    z = z_vs_up_rate(hit, len(sided))
    strong = buckets.get_group("strong") if "strong" in buckets.groups else pd.Series(dtype=bool)
    return {
        "calls": len(df),
        "sided_calls": len(sided),
        "hit_rate": hit,
        "always_bullish_hit_rate": up_rate,
        "rank_corr": float(df["direction"].corr(df["fwd_return"], method="spearman")),
        "z_vs_baseline": float(z),
        **{f"hit_rate_{k}": float(v) for k, v in buckets.mean().items()},
        **{f"n_{k}": int(v) for k, v in buckets.size().items()},
        "z_strong_vs_baseline": float(z_vs_up_rate(strong.mean(), len(strong))) if len(strong) else np.nan,
        **{f"avg_fwd_return_{k}": float(v) for k, v in by_stance.items()},
        "verdict": _verdict(z) if len(sided) else "untested",
    }


def score_vol(df):
    df = df.dropna(subset=["fwd_vol", "vol_forecast", "naive_vol"])
    mae = float((df["vol_forecast"] - df["fwd_vol"]).abs().mean())
    naive_mae = float((df["naive_vol"] - df["fwd_vol"]).abs().mean())
    by_level = df.groupby("vol_level")["fwd_vol"].mean()
    skill = 1 - mae / naive_mae
    # Paired test: is the model's error reliably smaller than the naive one?
    gain = (df["naive_vol"] - df["fwd_vol"]).abs() - (df["vol_forecast"] - df["fwd_vol"]).abs()
    z = gain.mean() / (gain.std() / np.sqrt(_effective_n(df, len(df))))
    return {
        "calls": len(df),
        "vol_corr": float(df["vol_forecast"].corr(df["fwd_vol"])),
        "vol_mae": mae,
        "naive_vol_mae": naive_mae,
        "vol_skill": skill,
        "z_vs_baseline": float(z),
        **{f"realised_vol_when_{k}": float(v) for k, v in by_level.items()},
        "verdict": _verdict(z),
    }


def scorecard(records, known_by=None, min_calls=1):
    """Summary stats per arrow, keyed by model name.

    With `known_by`, only calls whose outcome was known by that date count, so
    a track record quoted at that date uses no future information.
    """
    if known_by is not None:
        records = records[records["resolved"] <= known_by]
    card = {}
    for model, df in records.groupby("model"):
        if df["fwd_return"].notna().sum() < min_calls:
            continue
        entry = {"horizon": int(df["horizon"].iloc[0]),
                 "since": str(df["date"].min().date())}
        if df["direction"].notna().any():
            entry["direction"] = score_direction(df)
        if df["vol_forecast"].notna().any():
            entry["volatility"] = score_vol(df)
        card[model] = entry
    return card


def track_record_text(entry):
    """Plain-English track record. Comparisons are pre-computed into words
    because the decision model shouldn't be asked to do arithmetic."""
    parts = []
    if d := entry.get("direction"):
        parts.append(
            f"Direction calls since {entry['since']}: right {d['hit_rate']:.0%} of the time when it took a side, "
            f"{d['verdict']} simply assuming the price rises ({d['always_bullish_hit_rate']:.0%})."
        )
        if d.get("n_strong", 0) >= MIN_BUCKET and d.get("z_strong_vs_baseline", 0) >= 2:
            parts.append(f"Its strong signals were reliably better: right {d['hit_rate_strong']:.0%} of the time.")
    if v := entry.get("volatility"):
        parts.append(
            f"Volatility forecasts since {entry['since']} were {v['verdict']} assuming next month "
            f"looks like last month."
        )
    return " ".join(parts)


def save_scorecard(ticker, card):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{ticker.upper()}_scorecard.json"
    path.write_text(json.dumps(card, indent=2))
    return path


def load_scorecard(ticker):
    path = RESULTS_DIR / f"{ticker.upper()}_scorecard.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="GLD")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--step", type=int, default=20, help="trading days between scoring dates")
    parser.add_argument("--horizon", type=int, default=None,
                        help="trading days ahead each call is judged (default: each arrow's own)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    prices = fetch_prices(args.ticker, args.start)
    arrows = default_arrows(args.ticker, args.horizon)
    # Fetch real yields once instead of on every refit.
    arrows = [RealYield(real_yield=fetch_real_yield()) if isinstance(a, RealYield) else a
              for a in arrows]

    records = walk_forward(prices, arrows, step=args.step)
    RESULTS_DIR.mkdir(exist_ok=True)
    records.drop(columns="signal").to_csv(RESULTS_DIR / f"{args.ticker.upper()}_signals.csv", index=False)
    card = scorecard(records)
    path = save_scorecard(args.ticker, card)

    for model, entry in card.items():
        print(f"\n== {model} (horizon {entry['horizon']}d) ==")
        for kind in ("direction", "volatility"):
            if kind in entry:
                print(pd.Series(entry[kind]).to_string())
        print("->", track_record_text(entry))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
