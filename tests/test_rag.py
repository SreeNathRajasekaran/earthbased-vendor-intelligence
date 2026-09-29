import pandas as pd
import pytest

from src.evaluation import GROUNDING_TESTS
from src.rag import Evidence, VendorCopilot, check_grounding, detect_intent


@pytest.fixture(scope="module")
def copilot(system):
    return VendorCopilot(system, use_llm=False)


@pytest.mark.parametrize("case", GROUNDING_TESTS, ids=[c["question"][:40] for c in GROUNDING_TESTS])
def test_grounding_suite(copilot, case):
    r = copilot.ask(case["question"])
    assert r.intent == case["intent"]
    assert r.sufficient == case["sufficient"]
    assert r.grounded, r.grounding_issues
    if r.sufficient:
        assert not r.evidence.empty
        assert "caveat" in r.answer.lower()
    else:
        assert "insufficient evidence" in r.answer.lower()


def test_evidence_ids_exist(copilot, system):
    r = copilot.ask("Which vendors should we prioritise?")
    assert set(r.evidence["vendor_id"]) <= set(system.profile["vendor_id"])


def test_check_catches_invented_vendor():
    ev = Evidence(pd.DataFrame({"vendor_id": ["V001"], "x": [0.5]}), [], True, ["V001"])
    assert any("V050" in i for i in check_grounding("V050 looks strongest.", ev))


def test_check_catches_invented_number():
    ev = Evidence(pd.DataFrame({"vendor_id": ["V001"], "gmv": [12000.0]}), [], True, ["V001"])
    assert check_grounding("V001 has GMV of ₹12,000.", ev) == []
    assert check_grounding("V001 has GMV of ₹98,765.", ev)


def test_check_flags_certainty():
    ev = Evidence(pd.DataFrame(), [], True, [])
    assert check_grounding("This vendor is guaranteed to perform.", ev)


def test_runs_without_api_key(system, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cp = VendorCopilot(system)
    assert cp.use_llm is False
    assert cp.ask("Which categories need more vendor coverage?").mode == "deterministic"


def test_intent_router():
    assert detect_intent("Find vendors similar to V008.") == "similar_vendors"
    assert detect_intent("What was EarthBased's actual revenue?") == "out_of_scope"
