"""Offline checks on synthetic prices: each arrow returns a valid Signal and
points the right way on cases with an obvious answer."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from arrows import Garch, HMMRegime, MeanReversion, RealYield, Signal, Trend
from quiver import briefing, draw_all

DAYS = 1000


def make_prices(drift=0.0, vol=0.01, seed=0, n=DAYS):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx, name="Price")


@pytest.mark.parametrize("arrow", [HMMRegime(), Garch(), Trend(), MeanReversion()],
                         ids=lambda a: a.name)
def test_arrow_returns_valid_signal(arrow):
    prices = make_prices()
    sig = arrow.fit(prices).signal()
    assert isinstance(sig, Signal)
    assert sig.model == arrow.name
    assert sig.as_of == prices.index[-1].date()
    assert sig.rationale
    assert sig.stance is not None or sig.vol_level is not None


def test_trend_follows_direction():
    assert Trend().fit(make_prices(drift=0.001, vol=0.005)).signal().stance == "bullish"
    assert Trend().fit(make_prices(drift=-0.001, vol=0.005)).signal().stance == "bearish"


def test_mean_reversion_after_sharp_drop():
    prices = make_prices(vol=0.005)
    prices.iloc[-5:] *= np.linspace(0.97, 0.90, 5)
    assert MeanReversion().fit(prices).signal().stance == "bullish"


def test_garch_flags_volatility_spike():
    calm = make_prices(vol=0.005, n=DAYS - 30)
    wild = make_prices(vol=0.04, n=30, seed=1)
    wild = wild / wild.iloc[0] * calm.iloc[-1]
    wild.index = pd.bdate_range(calm.index[-1] + pd.offsets.BDay(), periods=30)
    assert Garch().fit(pd.concat([calm, wild])).signal().vol_level == "high"


def _real_yield_case(sign):
    """Gold driven by real yields with the given sign, then pushed 10% rich."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=DAYS)
    ry = pd.Series(np.cumsum(rng.normal(0, 0.03, DAYS)), index=idx)
    logp = 5 + sign * 0.2 * ry + rng.normal(0, 0.01, DAYS)
    logp.iloc[-1] += 0.10
    return pd.Series(np.exp(logp), index=idx), ry


def test_real_yield_flags_expensive_gold():
    prices, ry = _real_yield_case(sign=-1)
    sig = RealYield(real_yield=ry).fit(prices).signal()
    assert sig.stance == "bearish"
    assert "expensive" in sig.rationale


def test_real_yield_abstains_when_relationship_inverts():
    prices, ry = _real_yield_case(sign=+1)
    sig = RealYield(real_yield=ry).fit(prices).signal()
    assert sig.direction == 0
    assert "cannot say" in sig.rationale


def test_signal_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        Signal(model="x", as_of="2026-01-01", horizon_days=5, direction=1.5,
               strength=0.5, rationale="r")
    with pytest.raises(ValidationError):
        Signal(model="x", as_of="2026-01-01", horizon_days=5, stance="moon",
               strength=0.5, rationale="r")


def test_draw_all_survives_a_broken_arrow():
    class Broken:
        name = "broken"
        def fit(self, prices):
            raise RuntimeError("boom")

    signals = draw_all(make_prices(), [Broken(), Trend()])
    assert [s.model for s in signals] == ["trend"]
    assert briefing("TEST", signals).count("\n") == 1
