"""Shared contract for every model ("arrow") in the quiver.

Each arrow does its own numeric work and reports a Signal. Signals carry
categorical fields (stance, vol_level) alongside the numbers because the
decision model that consumes them (TypeSafe's Jev) reasons over text and
typed options, and is unreliable at arithmetic. The numbers stay available
for the backtest harness and for debugging.
"""

from datetime import date
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Stance = Literal["bullish", "neutral", "bearish"]
VolLevel = Literal["low", "normal", "high"]

TRADING_DAYS = 252


class Signal(BaseModel, frozen=True):
    model: str = Field(description="Which arrow produced this signal")
    as_of: date = Field(description="Last date of data the signal used")
    horizon_days: int = Field(description="Trading days the view applies to")
    stance: Stance | None = Field(None, description="Directional view; None if the model doesn't call direction")
    direction: float | None = Field(None, ge=-1, le=1, description="-1 bearish .. +1 bullish")
    vol_level: VolLevel | None = Field(None, description="Volatility view; None if the model doesn't call volatility")
    vol_forecast: float | None = Field(None, ge=0, description="Forecast annualised volatility, %")
    strength: float = Field(ge=0, le=1, description="How strongly the model leans. Not a calibrated probability.")
    rationale: str = Field(description="One or two plain-English sentences, no maths required to follow")
    diagnostics: dict[str, float] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Render for a text-reading decision model."""
        parts = [f"[{self.model}] horizon {self.horizon_days} trading days"]
        if self.stance:
            parts.append(f"stance {self.stance}")
        if self.vol_level:
            parts.append(f"volatility {self.vol_level}")
        parts.append(f"strength {strength_word(self.strength)}")
        return ", ".join(parts) + f". {self.rationale}"


class Arrow(Protocol):
    name: str

    def fit(self, prices: pd.Series) -> "Arrow": ...
    def signal(self) -> Signal: ...


def strength_word(x: float) -> str:
    return "strong" if x >= 0.66 else "moderate" if x >= 0.33 else "weak"


def stance_from(direction: float, threshold: float = 0.25) -> Stance:
    if direction >= threshold:
        return "bullish"
    if direction <= -threshold:
        return "bearish"
    return "neutral"


def vol_level_from(percentile: float) -> VolLevel:
    """Bucket a volatility reading by where it sits in its own history."""
    if percentile < 1 / 3:
        return "low"
    if percentile > 2 / 3:
        return "high"
    return "normal"


def daily_returns(prices: pd.Series) -> pd.Series:
    """Daily % returns."""
    return prices.pct_change().mul(100).dropna()


def realised_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualised volatility, %."""
    return daily_returns(prices).rolling(window).std() * np.sqrt(TRADING_DAYS)


def pct(x: float) -> str:
    return f"{x:+.1f}%"
