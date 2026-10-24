from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from journey_agent.container import Container
from journey_agent.main import create_app
from journey_agent.persistence.repository import PageRecord
from journey_agent.welcome import brand_label, company_candidates, greeting_ok, validate
from tests.fakes import ScriptedLLM

LOAN_PAGE = """<html><head>
<title>Apply for Instant Pocket Loan Online | Fast Disbursal | Poonawalla Fincorp</title></head>
<body><img class="site-logo" alt="Poonawalla logo" src="/logo.svg">
<img alt="cibil-check-logo" src="/cibil.png">
<h1>Get Instant Loan up to ₹15 lakhs</h1>
<footer><p class="copyrightMsg__5_rIz">© Poonawalla Fincorp 2023. All rights reserved.</p></footer>
</body></html>"""


def test_brand_label_handles_subdomains_and_country_tlds() -> None:
    assert (
        brand_label("https://instant-pocket-loan.poonawallafincorp.com/?x=1") == "poonawallafincorp"
    )
    assert brand_label("https://www.amazon.in/") == "amazon"
    assert brand_label("https://shop.example.co.uk/") == "example"


def test_company_candidates_prefer_brand_over_headline() -> None:
    cands = company_candidates(LOAN_PAGE, "https://instant-pocket-loan.poonawallafincorp.com/")
    assert cands[0] == "Poonawalla Fincorp"
    assert not any("copyrightMsg" in c or "rIz" in c for c in cands)  # class names aren't text
    assert "Apply for Instant Pocket Loan Online" not in cands  # headline, not a brand


def test_company_candidates_strip_legal_suffix_and_copyright_word() -> None:
    html = (
        "<title>Personal Banking | HDFC Bank</title><footer>Copyright HDFC Bank Ltd. 2026</footer>"
    )
    assert company_candidates(html, "https://www.hdfc.bank.in/")[0] == "HDFC Bank"


GOOD = json.dumps(
    {
        "company_name": "Poonawalla Fincorp",
        "greeting": "Hi! I'm the virtual assistant for Poonawalla Fincorp. I can help you apply for an Instant Pocket Loan.",
    }
)


@pytest.mark.parametrize(
    ("company", "greeting"),
    [
        ("Acme Loans", "Hi! I'm the virtual assistant for Acme Loans."),  # invented company
        ("Poonawalla Fincorp", "Hi! I'm the assistant for Poonawalla Fincorp. Rates from 9.99%."),
        ("Poonawalla Fincorp", "Hi! Poonawalla Fincorp here: see https://evil.example"),
        ("Poonawalla Fincorp", "Hi! Poonawalla Fincorp guarantees instant approval."),
        ("Poonawalla Fincorp", "Hi! Please share your PAN so Poonawalla Fincorp can help."),
        ("Poonawalla Fincorp", "Hi! I can help you with loans."),  # doesn't name the company
    ],
)
def test_validate_rejects_unsafe_greetings(company: str, greeting: str) -> None:
    raw = json.dumps({"company_name": company, "greeting": greeting})
    assert validate(raw, ["Poonawalla Fincorp"], "Poonawalla Fincorp loan page") is None


def test_validate_accepts_json_wrapped_in_prose() -> None:
    raw = f"Sure, here it is:\n```json\n{GOOD}\n```"
    company, greeting = validate(raw, ["Poonawalla Fincorp"], "")  # type: ignore[misc]
    assert company == "Poonawalla Fincorp" and greeting.startswith("Hi!")


async def test_welcome_is_generated_once_and_cached(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.answer = GOOD.replace("Poonawalla Fincorp", "Acme")
    first = await container.welcome.welcome(page)
    assert first.generated_by == "llm" and first.company_name == "Acme"
    calls = len(llm.calls)
    again = await container.welcome.welcome(await container.repo.get_page(page.page_id) or page)
    assert again.message == first.message and len(llm.calls) == calls  # no second LLM call


async def test_llm_down_gives_template_then_retries(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.fail = True
    w = await container.welcome.welcome(page)
    assert w.generated_by == "template" and w.company_name in w.message
    stored = await container.repo.get_page(page.page_id)
    assert (
        stored is not None
        and stored.company_name == w.company_name
        and stored.welcome_message is None
    )

    llm.fail, llm.answer = False, GOOD.replace("Poonawalla Fincorp", "Acme")
    assert (await container.welcome.welcome(stored)).generated_by == "llm"  # model is back


async def test_stale_cached_greeting_is_regenerated(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    # Written under older, looser rules: a promise the assistant can't make.
    await container.repo.set_welcome(
        page.page_id, "Acme", "Welcome to Acme! We guarantee the fastest loans."
    )
    llm.answer = GOOD.replace("Poonawalla Fincorp", "Acme")
    w = await container.welcome.welcome(await container.repo.get_page(page.page_id) or page)
    assert "guarantee" not in w.message and w.generated_by == "llm"


def test_welcome_endpoint_and_company_in_canned_replies(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.answer = GOOD.replace("Poonawalla Fincorp", "Acme")
    with TestClient(create_app(container.settings, container)) as client:
        body = client.get(f"/api/pages/{page.page_id}/welcome").json()
        assert body == {
            "company_name": "Acme",
            "message": json.loads(llm.answer)["greeting"],
            "generated_by": "llm",
        }
        resp = client.post(
            "/api/chat",
            json={"page_id": page.page_id, "session_id": "session-0009", "message": "hello"},
        )
        assert "**Acme**" in resp.text  # small talk names the company, not the page title


def test_greeting_ok_matches_validate_rules() -> None:
    assert greeting_ok("Acme", "Hi! I'm the virtual assistant for Acme. Ask me about this page.")
    assert not greeting_ok("Acme", "Hi! Acme offers the best loans.")
