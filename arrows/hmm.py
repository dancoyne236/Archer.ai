"""HMM regime arrow: which volatility state are we in?"""

import numpy as np

from regime import fit_regimes

from .base import Signal, TRADING_DAYS, strength_word

# Map regime names onto the shared volatility buckets.
VOL_BUCKET = {"Calm": "low", "Normal": "normal", "Elevated": "high", "Turbulent": "high"}


class HMMRegime:
    name = "hmm_regime"

    def __init__(self, n_states=3, horizon_days=20):
        self.n_states = n_states
        self.horizon_days = horizon_days

    def fit(self, prices):
        self.rm = fit_regimes(prices, n_states=self.n_states)
        return self

    def signal(self):
        rm, h = self.rm, self.horizon_days
        s = rm.summary()
        stats = rm.regime_stats()
        daily_var = (stats["Annualised volatility (%)"].to_numpy() ** 2) / TRADING_DAYS

        # Average variance over the horizon, weighting each regime by how
        # likely we are to be in it on each future day.
        p, total = rm.current, 0.0
        for _ in range(h):
            p = p @ rm.transmat
            total += p @ daily_var
        vol_forecast = float(np.sqrt(total / h * TRADING_DAYS))

        persist = float(rm.outlook((h,)).iloc[0][s["regime"]])
        return Signal(
            model=self.name,
            as_of=s["as_of"],
            horizon_days=h,
            vol_level=VOL_BUCKET.get(s["regime"], "normal"),
            vol_forecast=vol_forecast,
            strength=s["confidence"],
            rationale=(
                f"The market is in a {s['regime'].lower()} regime "
                f"({strength_word(s['confidence'])} conviction), which has historically meant about "
                f"{stats.loc[s['regime'], 'Annualised volatility (%)']:.0f}% annualised volatility. "
                f"It has lasted {s['days_in_regime']} trading days so far; spells like this typically run "
                f"about {s['expected_duration']:.0f}. The chance it is still in place in {h} trading days "
                f"is {persist:.0%}."
            ),
            diagnostics={
                "confidence": s["confidence"],
                "days_in_regime": s["days_in_regime"],
                "expected_duration": s["expected_duration"],
                "persist_prob": persist,
            },
        )
