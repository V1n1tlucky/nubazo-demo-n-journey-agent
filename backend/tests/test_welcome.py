from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from journey_agent.container import Container
from journey_agent.main import create_app
from journey_agent.persistence.repository import PageRecord
from journey_agent.welcome import greeting_ok, parse_greeting, template_message
from tests.fakes import ScriptedLLM

GOOD = "Hi, welcome to Acme Loans! I'm your virtual assistant and I can help you apply for a personal loan."


def greet(text: str) -> str:
    return json.dumps({"greeting": text})


@pytest.mark.parametrize(
    "greeting",
    [
        "Hi! Ask me anything about this page.",  # the old fallback: no company, "page"
        "Hi, welcome to Acme Loans! Ask me anything about this page.",
        "Welcome to Acme Loans! Explore our website to learn more.",
        "Hi! I'm the assistant for Acme Loans. Rates from 9.99%.",
        "Hi! Acme Loans here: see https://evil.example",
        "Hi! Acme Loans guarantees instant approval.",
        "Hi! Please share your PAN so Acme Loans can help.",
        "Hi! I can help you with loans.",  # doesn't name the company
    ],
)
def test_unsafe_or_page_talk_greetings_are_rejected(greeting: str) -> None:
    assert not greeting_ok("Acme Loans", greeting)


def test_good_greeting_and_template_pass() -> None:
    assert greeting_ok("Acme Loans", GOOD)
    assert greeting_ok("Acme Loans", template_message("Acme Loans"))  # the fallback must pass too
    assert parse_greeting(f"Sure:\n```json\n{greet(GOOD)}\n```", "Acme Loans") == GOOD


async def test_greeting_names_the_ingested_company_and_is_cached(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.greeting = greet(GOOD)
    first = await container.welcome.welcome(page)
    assert first.generated_by == "llm" and first.company_name == "Acme Loans"
    system = llm.side_calls[-1][0].content
    assert "Acme Loans" in system and "COMPANY" not in system  # prompt carries the fixed name
    n = len(llm.side_calls)
    again = await container.welcome.welcome(await container.repo.get_page(page.page_id) or page)
    assert again.message == GOOD and len(llm.side_calls) == n


async def test_rejected_greeting_falls_back_to_template_without_caching(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.greeting = greet("Hi! Ask me anything about this page.")
    w = await container.welcome.welcome(page)
    assert w.generated_by == "template" and w.message == template_message("Acme Loans")
    stored = await container.repo.get_page(page.page_id)
    assert stored is not None and stored.welcome_message is None  # retried next time
    llm.greeting = greet(GOOD)
    assert (await container.welcome.welcome(stored)).generated_by == "llm"


async def test_stale_cached_greeting_is_regenerated(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    await container.repo.set_welcome(page.page_id, "Hi! Ask me anything about this page.")
    llm.greeting = greet(GOOD)
    w = await container.welcome.welcome(await container.repo.get_page(page.page_id) or page)
    assert w.message == GOOD


def test_same_company_in_endpoint_greeting_and_canned_replies(
    container: Container, page: PageRecord, llm: ScriptedLLM
) -> None:
    llm.greeting = greet(GOOD)
    with TestClient(create_app(container.settings, container)) as client:
        body = client.get(f"/api/pages/{page.page_id}/welcome").json()
        assert body == {"company_name": "Acme Loans", "message": GOOD, "generated_by": "llm"}
        resp = client.post(
            "/api/chat",
            json={"page_id": page.page_id, "session_id": "session-0009", "message": "hello"},
        )
        assert (
            "**Acme Loans**" in resp.text and "page" not in resp.text.split('"answer"')[-1].lower()
        )
