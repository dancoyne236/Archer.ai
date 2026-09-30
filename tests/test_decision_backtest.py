"""Decision backtest: portfolio maths, no look-ahead in the briefing, and the
archer loop end to end against the stand-in decision model."""

import numpy as np
import pandas as pd
import pytest

from archer import build_agent
from arrows import Garch, Trend
from backtest import scorecard, walk_forward
from decision_backtest import (
    compare, run_decisions, simulate, vol_target_weights, weights_from_actions,
)
from tests.test_archer import FakeJev
from tests.test_arrows import make_prices


@pytest.fixture(scope="module")
def setup():
    prices = make_prices(drift=0.0003)
    records = walk_forward(prices, [Garch(), Trend()], warmup=300, step=40)
    return prices, records


def test_weights_step_and_clamp():
    idx = pd.bdate_range("2020-01-01", periods=6)
    d = pd.DataFrame({"action": ["increase", "increase", "reduce", "reduce", "reduce", "reduce"],
                      "escalate": [False, False, False, True, False, False]}, index=idx)
    assert list(weights_from_actions(d)) == [1.0, 1.0, 0.75, 0.75, 0.5, 0.25]


def test_simulate_applies_weight_from_next_day():
    prices = pd.Series([100, 110, 121, 121], index=pd.bdate_range("2020-01-01", periods=4))
    weights = pd.Series([0.5], index=prices.index[:1])
    daily = simulate(prices, weights)
    assert list(daily.round(6)) == [0.05, 0.05, 0.0]


def test_vol_target_has_no_look_ahead(setup):
    _, records = setup
    w = vol_target_weights(records)
    # Changing a later forecast must not move an earlier weight.
    tweaked = records.copy()
    last = tweaked.index[tweaked["model"] == "garch"][-1]
    tweaked.loc[last, "vol_forecast"] *= 10
    w2 = vol_target_weights(tweaked)
    assert w.iloc[:-1].equals(w2.iloc[:-1])
    assert w.between(0, 1).all()


def test_point_in_time_scorecard_ignores_unresolved_calls(setup):
    _, records = setup
    early = records["date"].sort_values().iloc[len(records) // 2]
    card = scorecard(records, known_by=early)
    counted = records[records["resolved"] <= early]
    for model, entry in card.items():
        assert sum(counted["model"] == model) >= 1
    assert all(pd.Timestamp(e["since"]) <= early for e in card.values())


def test_archer_that_always_holds_matches_buy_and_hold(setup):
    prices, records = setup
    decisions = run_decisions("TEST", records, build_agent(FakeJev({"action": "hold"}, 0.9)))
    table = compare(prices, records, decisions)
    assert table.loc["Archer (Jev)", "Annual return"] == pytest.approx(
        table.loc["Buy and hold", "Annual return"])
    assert table.loc["Archer (Jev)", "Trades"] == 0


def test_one_call_per_date_and_no_track_record_before_enough_history(setup):
    _, records = setup
    jev = FakeJev({"action": "reduce"}, 0.9)
    run_decisions("TEST", records, build_agent(jev))
    dates = sorted(records["date"].unique())
    # With few resolved calls, early briefings carry no track record at all.
    assert "Track record" not in str(jev.requests[0].state)
    assert len(jev.requests) == len(dates)
