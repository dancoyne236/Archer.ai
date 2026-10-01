"""The dashboard must only ever use a key the visitor pasted, never one from
the server's environment. Runs offline on synthetic prices."""

import pytest
from streamlit.testing.v1 import AppTest

import archer
import arrows.real_yield
import regime
from tests.test_archer import FakeJev
from tests.test_arrows import make_prices


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SERVER")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-SERVER")
    monkeypatch.setattr(regime, "fetch_prices", lambda ticker, start: make_prices(drift=0.0003))
    monkeypatch.setattr(arrows.real_yield, "fetch_real_yield",
                        lambda: make_prices().rename("real_yield") / 50)
    keys = []

    def fake_jev_model(name="jev-latest", api_key=None, via=None):
        keys.append(api_key)
        return FakeJev({"action": "hold"}, 0.9)

    monkeypatch.setattr(archer, "jev_model", fake_jev_model)
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    return at, keys


def test_no_pasted_key_means_no_jev_even_with_server_keys(app):
    at, keys = app
    assert not at.exception
    assert not any(b.label == "Ask Jev" for b in at.button)
    assert keys == []


def test_jev_uses_only_the_visitors_key(app):
    at, keys = app
    at.sidebar.text_input[0].set_value("sk-or-VISITOR").run()
    next(b for b in at.button if b.label == "Ask Jev").click().run()
    assert not at.exception
    assert keys == ["sk-or-VISITOR"]
    assert [m.value for m in at.metric][0] == "Hold"


def test_bad_ticker_is_rejected(app):
    at, _ = app
    box = next(t for t in at.text_input if t.label.startswith("Ticker"))
    box.set_value("rm -rf /")
    at.button[0].click().run()
    assert "Enter a ticker" in at.error[0].value
