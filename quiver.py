"""Run every arrow on one asset and collect their Signals.

`briefing()` renders the signals as the text state a decision model like
Jev reads; `draw_all()` returns the typed Signals for code.

Usage:
    python quiver.py            # GLD briefing
    python quiver.py GC=F --json
"""

import argparse
import logging

from arrows import Garch, HMMRegime, MeanReversion, RealYield, Signal, Trend
from regime import fetch_prices

log = logging.getLogger(__name__)

GOLD_TICKERS = {"GLD", "IAU", "GC=F", "GLDM", "SGOL"}


def default_arrows(ticker):
    arrows = [HMMRegime(), Garch(), Trend(), MeanReversion()]
    if ticker.upper() in GOLD_TICKERS:
        arrows.append(RealYield())  # the real-yield link is specific to gold
    return arrows


def draw_all(prices, arrows) -> list[Signal]:
    """Fit each arrow and collect its signal. One broken arrow doesn't sink the rest."""
    signals = []
    for arrow in arrows:
        try:
            signals.append(arrow.fit(prices).signal())
        except Exception:
            log.exception("Arrow %s failed", arrow.name)
    return signals


def briefing(ticker, signals: list[Signal]) -> str:
    """Text state for a decision model: one line per arrow."""
    as_of = max(s.as_of for s in signals)
    lines = [f"Asset: {ticker}. Data as of {as_of}. {len(signals)} model signals follow."]
    lines += [s.to_text() for s in signals]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="GLD")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--json", action="store_true", help="print typed signals as JSON")
    args = parser.parse_args()

    prices = fetch_prices(args.ticker, args.start)
    signals = draw_all(prices, default_arrows(args.ticker))
    if args.json:
        for s in signals:
            print(s.model_dump_json(indent=2))
    else:
        print(briefing(args.ticker, signals))


if __name__ == "__main__":
    main()
