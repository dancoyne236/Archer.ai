"""Scoring checks: a perfect forecaster should beat its baseline, a coin flip
shouldn't, and the walk-forward loop must never see future prices."""

import numpy as np
import pandas as pd

from arrows import Signal, Trend
from backtest import scorecard, track_record_text, walk_forward
from tests.test_arrows import make_prices


def _records(direction, fwd_return, n=400, vol_forecast=None, fwd_vol=None):
    return pd.DataFrame({
        "model": "m", "date": pd.bdate_range("2015-01-01", periods=n), "horizon": 20, "step": 20,
        "stance": np.where(direction > 0, "bullish", "bearish"), "direction": direction,
        "vol_level": None, "vol_forecast": vol_forecast, "strength": np.abs(direction),
        "naive_vol": 15.0, "fwd_return": fwd_return, "fwd_vol": fwd_vol,
    })


def test_perfect_direction_beats_baseline():
    rng = np.random.default_rng(0)
    fwd = rng.normal(0, 2, 400)
    card = scorecard(_records(np.sign(fwd) * 0.9, fwd))
    assert card["m"]["direction"]["hit_rate"] == 1.0
    assert card["m"]["direction"]["verdict"] == "better than"


def test_coin_flip_is_not_reliably_better():
    rng = np.random.default_rng(1)
    card = scorecard(_records(rng.choice([-0.9, 0.9], 400), rng.normal(0, 2, 400)))
    assert card["m"]["direction"]["verdict"] == "not reliably different from"


def test_accurate_vol_forecast_beats_naive():
    rng = np.random.default_rng(2)
    actual = rng.uniform(5, 30, 400)
    df = _records(np.zeros(400), np.zeros(400), vol_forecast=actual + rng.normal(0, 0.5, 400),
                  fwd_vol=actual)
    df["direction"] = np.nan
    entry = scorecard(df)["m"]
    assert entry["volatility"]["verdict"] == "better than"
    assert "better than assuming next month" in track_record_text(entry)


def test_walk_forward_never_sees_the_future():
    prices = make_prices()
    seen = []

    class Spy:
        name = "spy"
        def fit(self, history):
            seen.append(history.index[-1])
            self.as_of = history.index[-1]
            return self
        def signal(self):
            return Signal(model=self.name, as_of=self.as_of.date(), horizon_days=20,
                          stance="bullish", direction=0.5, strength=0.5, rationale="r")

    records = walk_forward(prices, [Spy()], warmup=300, step=50)
    assert list(records["date"]) == seen
    # Outcome is measured strictly after the signal date.
    first = records.iloc[0]
    i = prices.index.get_loc(first["date"])
    assert np.isclose(first["fwd_return"], (prices.iloc[i + 20] / prices.iloc[i] - 1) * 100)


def test_walk_forward_runs_real_arrow():
    records = walk_forward(make_prices(), [Trend()], warmup=300, step=100)
    assert len(records) == 7 and records["fwd_return"].notna().sum() == 7
