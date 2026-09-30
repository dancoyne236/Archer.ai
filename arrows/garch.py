"""GARCH(1,1) arrow: how volatile will the next few weeks be?"""

import numpy as np
from arch import arch_model

from .base import Signal, TRADING_DAYS, daily_returns, vol_level_from


class Garch:
    name = "garch"

    def __init__(self, horizon_days=20):
        self.horizon_days = horizon_days

    def fit(self, prices):
        self.returns = daily_returns(prices)
        self.res = arch_model(self.returns, mean="Constant", vol="GARCH",
                              p=1, q=1, dist="t").fit(disp="off")
        return self

    def signal(self):
        h = self.horizon_days
        ann = np.sqrt(TRADING_DAYS)
        fc = self.res.forecast(horizon=h, reindex=False).variance.iloc[-1].to_numpy()
        vol_forecast = float(np.sqrt(fc.mean()) * ann)

        history = self.res.conditional_volatility * ann
        current = float(history.iloc[-1])
        percentile = float((history < vol_forecast).mean())
        level = vol_level_from(percentile)

        # Long-run level the model reverts to; tells the agent which way vol is drifting.
        omega, alpha, beta = (self.res.params[k] for k in ("omega", "alpha[1]", "beta[1]"))
        persistence = alpha + beta
        long_run = float(np.sqrt(omega / (1 - persistence)) * ann) if persistence < 1 else float("nan")
        drift = "rising" if vol_forecast > current * 1.05 else "falling" if vol_forecast < current * 0.95 else "steady"

        return Signal(
            model=self.name,
            as_of=self.returns.index[-1].date(),
            horizon_days=h,
            vol_level=level,
            vol_forecast=vol_forecast,
            strength=abs(percentile - 0.5) * 2,
            rationale=(
                f"Volatility is forecast at about {vol_forecast:.0f}% annualised over the next {h} trading days, "
                f"{level} by this asset's own history (higher than {percentile:.0%} of past days). "
                f"It is {drift} from today's {current:.0f}%, and the long-run norm is about {long_run:.0f}%."
            ),
            diagnostics={
                "current_vol": current,
                "long_run_vol": long_run,
                "percentile": percentile,
                "persistence": float(persistence),
            },
        )
