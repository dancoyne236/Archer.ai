"""Real-yield arrow: is gold cheap or rich relative to real interest rates?

Gold pays no income, so it has historically moved inversely to the 10-year
TIPS yield (FRED series DFII10). This arrow fits that relationship over a
rolling window and reports how far the price sits from what yields imply,
scaled down when the relationship has been weak.
"""

import numpy as np
import pandas as pd

from .base import Signal, TRADING_DAYS, pct, stance_from

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_real_yield(series="DFII10"):
    """Daily 10-year real yield, %, from FRED (no API key needed)."""
    df = pd.read_csv(FRED_URL.format(series=series), index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename("real_yield")


class RealYield:
    name = "real_yield"

    def __init__(self, real_yield=None, window_years=3, horizon_days=60):
        self.real_yield = real_yield  # inject for backtests to avoid refetching
        self.window = window_years * TRADING_DAYS
        self.horizon_days = horizon_days

    def fit(self, prices):
        ry = self.real_yield if self.real_yield is not None else fetch_real_yield()
        df = pd.concat([np.log(prices).rename("logp"), ry.rename("real_yield").reindex(prices.index).ffill()],
                       axis=1).dropna().iloc[-self.window:]
        if len(df) < TRADING_DAYS:
            raise ValueError("Real-yield model needs at least a year of overlapping data")

        slope, intercept = np.polyfit(df["real_yield"], df["logp"], 1)
        fitted = intercept + slope * df["real_yield"]
        resid = df["logp"] - fitted
        self.df, self.slope = df, float(slope)
        self.r2 = float(1 - resid.var() / df["logp"].var())
        self.resid_z = float(resid.iloc[-1] / resid.std())
        self.gap = float(np.exp(resid.iloc[-1]) - 1) * 100
        return self

    def signal(self):
        df = self.df
        ry_now = float(df["real_yield"].iloc[-1])
        ry_chg = ry_now - float(df["real_yield"].iloc[-21])

        # Only trust the valuation if the fit has the expected sign (higher
        # real yields -> lower gold) and explains a meaningful share of moves.
        weight = self.r2 if self.slope < 0 else 0.0
        direction = float(-np.tanh(self.resid_z / 2) * weight)

        years = len(df) // TRADING_DAYS
        if self.slope >= 0:
            view = (f"Over the past {years} years gold has not moved inversely to real yields as it "
                    "usually does, so this model cannot say whether gold is cheap or expensive right now.")
        else:
            quality = "strong" if self.r2 >= 0.6 else "moderate" if self.r2 >= 0.3 else "weak"
            view = (f"Gold is {pct(self.gap)} {'above' if self.gap > 0 else 'below'} the level real yields "
                    f"imply, so it looks {'expensive' if self.gap > 0 else 'cheap'} on this measure. "
                    f"The relationship has been {quality} over the past {years} years.")

        return Signal(
            model=self.name,
            as_of=df.index[-1].date(),
            horizon_days=self.horizon_days,
            stance=stance_from(direction),
            direction=direction,
            strength=abs(direction),
            rationale=(
                f"The 10-year real yield is {ry_now:.2f}% ({ry_chg:+.2f} points over the past month). {view}"
            ),
            diagnostics={
                "real_yield": ry_now,
                "real_yield_chg_1m": ry_chg,
                "slope": self.slope,
                "r2": self.r2,
                "resid_z": self.resid_z,
            },
        )
