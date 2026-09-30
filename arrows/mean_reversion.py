"""Mean-reversion arrow: has the price stretched too far, too fast?"""

import numpy as np

from .base import Signal, pct, stance_from


def rsi(prices, window=14):
    """Wilder's relative strength index, 0-100."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


class MeanReversion:
    name = "mean_reversion"

    def __init__(self, window=20, horizon_days=5):
        self.window = window
        self.horizon_days = horizon_days

    def fit(self, prices):
        if len(prices) < self.window + 1:
            raise ValueError(f"Mean reversion needs at least {self.window + 1} prices")
        self.prices = prices
        return self

    def signal(self):
        logp = np.log(self.prices)
        recent = logp.iloc[-self.window:]
        z = float((logp.iloc[-1] - recent.mean()) / recent.std())
        r = float(rsi(self.prices).iloc[-1])
        gap = (np.exp(logp.iloc[-1] - recent.mean()) - 1) * 100

        # Stretched up -> expect a pullback (bearish), and vice versa.
        direction = float(-np.tanh(z / 2))
        state = ("overbought" if z > 1.5 or r > 70 else
                 "oversold" if z < -1.5 or r < 30 else "not stretched")

        return Signal(
            model=self.name,
            as_of=self.prices.index[-1].date(),
            horizon_days=self.horizon_days,
            stance=stance_from(direction, threshold=0.5),
            direction=direction,
            strength=abs(direction),
            rationale=(
                f"The price is {pct(gap)} from its {self.window}-day average, {abs(z):.1f} typical moves "
                f"{'above' if z > 0 else 'below'} it, with RSI at {r:.0f}. Short-term it looks {state}."
            ),
            diagnostics={"zscore": z, "rsi": r},
        )
