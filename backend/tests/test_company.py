from __future__ import annotations

from fastapi.testclient import TestClient

from journey_agent.company import (
    CompanyIdentificationError,
    brand_label,
    company_candidates,
    identify_company,
    is_grounded,
    parse_company,
)
from journey_agent.container import Container
from journey_agent.main import create_app
from journey_agent.persistence.repository import PageRecord
from tests.fakes import ScriptedLLM

LOAN_PAGE = """<html><head>
<title>Apply for Instant Pocket Loan Online | Fast Disbursal | Poonawalla Fincorp</title></head>
<body><img class="site-logo" alt="Poonawalla logo" src="/logo.svg">
<img alt="cibil-check-logo" src="/cibil.png">
<h1>Get Instant Loan up to ₹15 lakhs</h1>
<footer><p class="copyrightMsg__5_rIz">© Poonawalla Fincorp 2023. All rights reserved.</p></footer>
</body></html>"""
URL = "https://instant-pocket-loan.poonawallafincorp.com/"


def test_brand_label_handles_subdomains_and_country_tlds() -> None:
    assert brand_label(URL + "?x=1") == "poonawallafincorp"
    assert brand_label("https://www.amazon.in/") == "amazon"
    assert brand_label("https://shop.example.co.uk/") == "example"


def test_candidates_prefer_brand_over_headline_and_class_names() -> None:
    cands = company_candidates(LOAN_PAGE, URL)
    assert cands[0] == "Poonawalla Fincorp"
    assert not any("copyrightMsg" in c or "rIz" in c for c in cands)
    assert "Apply for Instant Pocket Loan Online" not in cands


def test_candidates_strip_legal_suffix_and_copyright_word() -> None:
    html = (
        "<title>Personal Banking | HDFC Bank</title><footer>Copyright HDFC Bank Ltd. 2026</footer>"
    )
    assert company_candidates(html, "https://www.hdfc.bank.in/")[0] == "HDFC Bank"


def test_parse_company_normalises_legal_suffix() -> None:
    assert parse_company('Here: {"company_name": "HDFC Bank Ltd."}') == "HDFC Bank"
    assert parse_company("not json") == ""


def test_grounding_rejects_invented_names() -> None:
    assert is_grounded("Poonawalla Fincorp", ["Poonawalla Fincorp"], "")
    assert is_grounded("PFL", [], "… policy of PFL and …")
    assert not is_grounded("Acme Finance", ["Poonawalla Fincorp"], "a loan page")


async def test_identify_uses_llm_answer() -> None:
    llm = ScriptedLLM()
    llm.company = "Poonawalla Fincorp"
    name = await identify_company(
        llm, url=URL, title="t", description="", excerpt="", html=LOAN_PAGE
    )
    assert name == "Poonawalla Fincorp" and len(llm.side_calls) == 1


async def test_identify_retries_multiple_choice_then_fails() -> None:
    llm = ScriptedLLM()
    llm.company = "Totally Invented Corp"  # not on the page, both attempts
    try:
        await identify_company(llm, url=URL, title="t", description="", excerpt="", html=LOAN_PAGE)
    except CompanyIdentificationError as exc:
        assert "Couldn't identify" in str(exc)
    else:
        raise AssertionError("an ungrounded name must not be accepted")
    assert len(llm.side_calls) == 2  # free answer, then multiple choice
    assert "Answer with exactly one of" in llm.side_calls[1][1].content


def test_ingestion_fails_without_model_and_saves_nothing(
    container: Container, page_html: str, llm: ScriptedLLM
) -> None:
    llm.fail = True
    with TestClient(create_app(container.settings, container)) as client:
        resp = client.post(
            "/api/pages/upload", files={"file": ("page.html", page_html, "text/html")}
        )
        assert resp.status_code == 503 and resp.json()["code"] == "llm_unavailable"
        assert "model server" in resp.json()["detail"]
        assert client.get("/api/pages").json() == []  # nothing half-ingested


async def test_ingested_page_has_llm_company(page: PageRecord) -> None:
    assert page.company_name == "Acme Loans"


async def test_older_pages_get_identified_on_first_use(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    await container.repo.set_company(page.page_id, "")  # as if ingested before it was required
    stale = await container.repo.get_page(page.page_id)
    assert stale is not None and not stale.company_name
    llm.side_calls.clear()
    fresh = await container.ingestion.ensure_company(stale)
    assert fresh.company_name == "Acme Loans" and len(llm.side_calls) == 1
    stored = await container.repo.get_page(page.page_id)
    assert stored is not None and stored.company_name == "Acme Loans"
