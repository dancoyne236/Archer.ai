"""The archer: TypeSafe's Jev reads the quiver's briefing and makes a typed call.

Jev answers each field of `Decision` as a separate question, with a
calibrated confidence. When it isn't confident enough about the action, the
call is marked for escalation (to a person or a language model) instead of
being acted on.

Setup:
    pip install "pydantic-ai-slim[typesafe]"
    export TYPESAFE_API_KEY=...
    python backtest.py GLD      # optional but recommended: adds track records

Usage:
    python archer.py            # GLD
    python archer.py GC=F --min-confidence 0.8
"""

import argparse
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from backtest import load_scorecard
from quiver import briefing, default_arrows, draw_all
from regime import fetch_prices

Action = Literal["increase", "hold", "reduce"]

INSTRUCTIONS = """\
You advise a long-term investor who already holds this asset. The text is a
briefing from several quantitative models: each line gives one model's view,
the horizon it applies to, and when available its track record against a
simple baseline. Trust a model only as far as its track record supports:
a model described as "not reliably different from" its baseline has not shown
an edge. Volatility views are about risk, not about direction."""


class Decision(BaseModel):
    """A position decision for the asset in the briefing, for the next month."""

    action: Action = Field(description=(
        "Taking each model's track record into account, should the investor increase, hold, "
        "or reduce their position over the next month?"))
    risk: Literal["low", "normal", "high"] = Field(description=(
        "How risky is holding this asset over the next month, judging from the volatility views?"))
    signals_conflict: bool = Field(description=(
        "Do the directional views in the briefing point in different directions?"))


@dataclass(frozen=True)
class ArcherCall:
    ticker: str
    decision: Decision
    confidence: dict[str, float]  # per field, from Jev
    min_confidence: float
    briefing: str

    @property
    def escalate(self) -> bool:
        return self.confidence.get("action", 0.0) < self.min_confidence

    def __str__(self):
        c = self.confidence
        verdict = (f"ESCALATE: Jev leans '{self.decision.action}' but is below the "
                   f"{self.min_confidence:.0%} confidence bar" if self.escalate
                   else f"Action: {self.decision.action.upper()}")
        return (f"{self.ticker}: {verdict} (confidence {c.get('action', 0):.0%})\n"
                f"Risk: {self.decision.risk} ({c.get('risk', 0):.0%})\n"
                f"Signals conflict: {'yes' if self.decision.signals_conflict else 'no'} "
                f"({c.get('signals_conflict', 0):.0%})")


def build_agent(model: Model | str = "typesafe:jev-latest") -> Agent:
    return Agent(model, output_type=Decision, instructions=INSTRUCTIONS)


def judge(agent: Agent, ticker: str, text: str, min_confidence=0.7) -> ArcherCall:
    """Have the archer make a call on a briefing."""
    result = agent.run_sync(text)
    details = result.response.provider_details or {}
    return ArcherCall(ticker=ticker, decision=result.output,
                      confidence=dict(details.get("confidence", {})),
                      min_confidence=min_confidence, briefing=text)


def decide(ticker="GLD", agent: Agent | None = None, prices=None,
           min_confidence=0.7) -> ArcherCall:
    """Run every arrow on `ticker` and have the archer make a call."""
    prices = fetch_prices(ticker) if prices is None else prices
    text = briefing(ticker, draw_all(prices, default_arrows(ticker)), load_scorecard(ticker))
    return judge(agent or build_agent(), ticker, text, min_confidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="GLD")
    parser.add_argument("--model", default="typesafe:jev-latest")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--show-briefing", action="store_true")
    args = parser.parse_args()

    if args.model.startswith("typesafe:") and not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("set TYPESAFE_API_KEY first (see https://docs.typesafe.ai)")

    call = decide(args.ticker, build_agent(args.model), min_confidence=args.min_confidence)
    if args.show_briefing:
        print(call.briefing, "\n")
    print(call)
    print("\nNot financial advice.")


if __name__ == "__main__":
    main()
