"""Trend arrow: time-series momentum. Is the asset going up or down?"""

import numpy as np

from .base import Signal, TRADING_DAYS, daily_returns, pct, stance_from


class Trend:
    name = "trend"

    def __init__(self, horizon_days=20):
        self.horizon_days = horizon_days

    def fit(self, prices):
        if len(prices) < TRADING_DAYS + 1:
            raise ValueError("Trend needs at least a year of prices")
        self.prices = prices
        return self

    def signal(self):
        p = self.prices
        vol = daily_returns(p).iloc[-TRADING_DAYS:].std() * np.sqrt(TRADING_DAYS) / 100
        price = p.iloc[-1]
        sma50, sma200 = p.iloc[-50:].mean(), p.iloc[-200:].mean()

        # Classic 12-1 momentum: past year's return, skipping the latest month.
        mom_12_1 = p.iloc[-21] / p.iloc[-TRADING_DAYS] - 1
        vs_200 = price / sma200 - 1
        cross = sma50 / sma200 - 1

        # Scale each by volatility so a 10% move means less in a wild market.
        z = np.array([mom_12_1 / vol,
                      vs_200 / (vol * np.sqrt(200 / TRADING_DAYS)),
                      cross / (vol * np.sqrt(100 / TRADING_DAYS))])
        direction = float(np.tanh(z.mean()))
        agree = int((np.sign(z) == np.sign(direction)).sum())

        return Signal(
            model=self.name,
            as_of=p.index[-1].date(),
            horizon_days=self.horizon_days,
            stance=stance_from(direction),
            direction=direction,
            strength=abs(direction),
            rationale=(
                f"Over the past year excluding the latest month the price is {pct(mom_12_1 * 100)}. "
                f"It sits {pct(vs_200 * 100)} versus its 200-day average, and the 50-day average is "
                f"{'above' if cross > 0 else 'below'} the 200-day. {agree} of 3 trend measures point the same way."
            ),
            diagnostics={
                "mom_12_1": float(mom_12_1),
                "vs_sma200": float(vs_200),
                "sma50_vs_sma200": float(cross),
                "agree": agree,
            },
        )
