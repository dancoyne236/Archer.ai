"""The archer against a stand-in decision model, so no API key is needed.

FakeJev speaks the same Decisions protocol Jev does, so these tests exercise
the real Pydantic AI mapping from `Decision` to questions and back."""

import pytest
from pydantic_ai.models.decision import (
    ChoiceAnswer, ChoiceQuestion, DecisionModel, DecisionResponse, NoulAnswer, NoulQuestion,
)

from archer import build_agent, decide
from tests.test_arrows import make_prices


class FakeJev(DecisionModel):
    """Answers every question with fixed picks and confidences."""

    def __init__(self, picks, confidence):
        super().__init__()
        self.picks, self.conf = picks, confidence
        self.requests = []

    model_name = "fake-jev"
    system = "fake"
    base_url = None

    async def decide(self, request, model_settings):
        self.requests.append(request)
        answers = {}
        for name, q in request.questions.items():
            field = next((f for f in self.picks if f in name), None)
            if isinstance(q, ChoiceQuestion):
                pick = self.picks.get(field, next(iter(q.criteria)))
                others = (1 - self.conf) / (len(q.criteria) - 1)
                answers[name] = ChoiceAnswer(
                    choice=pick, confidence=self.conf,
                    probabilities={o: self.conf if o == pick else others for o in q.criteria})
            elif isinstance(q, NoulQuestion):
                yes = self.picks.get(field, False)
                answers[name] = NoulAnswer(noul=0.95 if yes else 0.05)
            else:
                raise AssertionError(f"Unexpected question type {q}")
        return DecisionResponse(answers=answers, model_name=self.model_name)


@pytest.fixture
def prices():
    return make_prices(drift=0.0005)


def test_confident_call_is_actionable(prices):
    jev = FakeJev({"action": "reduce", "risk": "high", "signals_conflict": True}, confidence=0.9)
    call = decide("TEST", build_agent(jev), prices=prices)

    assert call.decision.action == "reduce"
    assert call.decision.risk == "high"
    assert call.decision.signals_conflict is True
    assert call.confidence["action"] == pytest.approx(0.9)
    assert not call.escalate
    assert "REDUCE" in str(call)


def test_unsure_call_escalates(prices):
    jev = FakeJev({"action": "increase"}, confidence=0.5)
    call = decide("TEST", build_agent(jev), prices=prices, min_confidence=0.7)
    assert call.escalate
    assert "ESCALATE" in str(call)


def test_jev_gets_the_briefing_and_typed_questions(prices):
    jev = FakeJev({"action": "hold"}, confidence=0.9)
    call = decide("TEST", build_agent(jev), prices=prices)

    kinds = {type(q) for q in jev.requests[0].questions.values()}
    assert kinds == {ChoiceQuestion, NoulQuestion}
    action_q = next(q for n, q in jev.requests[0].questions.items() if "action" in n)
    assert set(action_q.criteria) == {"increase", "hold", "reduce"}
    # The briefing text reaches the model as the state it judges.
    assert "[trend]" in str(jev.requests[0].state)
    assert call.briefing.startswith("Asset: TEST")
