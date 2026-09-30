"""Market regime detection with a Gaussian hidden Markov model.

Fits an HMM to daily % returns, orders the hidden states from lowest to
highest volatility so they get stable, human-readable names, and turns the
fitted model into plain answers: which regime we're in today, how long it
typically lasts, and how likely it is to persist.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

TRADING_DAYS = 252

REGIME_NAMES = {
    2: ["Calm", "Turbulent"],
    3: ["Calm", "Normal", "Turbulent"],
    4: ["Calm", "Normal", "Elevated", "Turbulent"],
}


def regime_names(n_states):
    return REGIME_NAMES.get(n_states, [f"Regime {i + 1}" for i in range(n_states)])


def fetch_prices(ticker="GLD", start="2010-01-01"):
    """Daily adjusted closing prices from Yahoo Finance, up to today."""
    hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    prices = hist["Close"].rename("Price")
    prices.index = prices.index.tz_localize(None).normalize()
    return prices


@dataclass
class RegimeModel:
    model: GaussianHMM
    names: list          # regime names, ordered calm -> turbulent
    transmat: np.ndarray  # transition matrix in that same order
    df: pd.DataFrame     # Price, Return, Regime, Label, and one P(<name>) column per regime

    @property
    def current(self):
        """Today's regime probabilities (filtered, so no look-ahead)."""
        return self.df[[f"P({n})" for n in self.names]].iloc[-1].to_numpy()

    def summary(self):
        """Plain-language snapshot of where things stand today."""
        probs = self.current
        idx = int(np.argmax(probs))
        labels = self.df["Label"]
        # Length of the current run of the same regime.
        run_breaks = labels.ne(labels.shift()).cumsum()
        days_in = int((run_breaks == run_breaks.iloc[-1]).sum())
        return {
            "as_of": self.df.index[-1].date(),
            "price": float(self.df["Price"].iloc[-1]),
            "regime": self.names[idx],
            "confidence": float(probs[idx]),
            "days_in_regime": days_in,
            "expected_duration": expected_duration(self.transmat)[idx],
        }

    def outlook(self, horizons=(5, 20, 60)):
        """Probability of being in each regime N trading days from now."""
        rows = {}
        for h in horizons:
            rows[f"{h} days"] = self.current @ np.linalg.matrix_power(self.transmat, h)
        return pd.DataFrame(rows, index=self.names).T

    def regime_stats(self):
        """How each regime has behaved historically."""
        durations = expected_duration(self.transmat)
        rows = []
        for i, name in enumerate(self.names):
            r = self.df.loc[self.df["Regime"] == i, "Return"]
            rows.append({
                "Regime": name,
                "Share of days": len(r) / len(self.df),
                "Avg daily move (%)": r.mean(),
                "Annualised volatility (%)": r.std() * np.sqrt(TRADING_DAYS),
                "Worst day (%)": r.min(),
                "Best day (%)": r.max(),
                "Typical length (days)": durations[i],
            })
        return pd.DataFrame(rows).set_index("Regime")


def expected_duration(transmat):
    """Average run length of each state: 1 / (1 - P(stay))."""
    stay = np.clip(np.diag(transmat), 0, 1 - 1e-9)
    return 1 / (1 - stay)


def fit_regimes(prices, n_states=3, n_init=5, seed=42):
    """Fit an HMM to daily returns and return a RegimeModel.

    Tries several random starts and keeps the best fit, since HMM training
    can land in poor local optima.
    """
    returns = prices.pct_change().mul(100).dropna()  # percent, so the HMM fits reliably
    X = returns.to_numpy().reshape(-1, 1)

    best, best_score = None, -np.inf
    for i in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=seed + i)
        m.fit(X)
        score = m.score(X)
        if score > best_score:
            best, best_score = m, score

    # Raw state numbers are arbitrary; rank them by volatility instead.
    order = np.argsort(best.covars_.reshape(n_states, -1)[:, 0])
    names = regime_names(n_states)

    # The last row of the forward-backward posterior equals the filtered
    # probability, so "today" uses no future data. Earlier rows are smoothed.
    probs = best.predict_proba(X)[:, order]

    df = pd.DataFrame({"Price": prices.loc[returns.index], "Return": returns})
    df["Regime"] = probs.argmax(axis=1)
    df["Label"] = [names[r] for r in df["Regime"]]
    for i, name in enumerate(names):
        df[f"P({name})"] = probs[:, i]

    transmat = best.transmat_[np.ix_(order, order)]
    return RegimeModel(model=best, names=names, transmat=transmat, df=df)
