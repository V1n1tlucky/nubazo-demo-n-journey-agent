from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from journey_agent.container import Container
from journey_agent.main import create_app


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container.settings, container)) as c:
        yield c


def sse_events(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        name = next(line[6:].strip() for line in block.split("\n") if line.startswith("event:"))
        data = next(line[5:].strip() for line in block.split("\n") if line.startswith("data:"))
        events.append((name, json.loads(data)))
    return events


def test_upload_chat_and_audit_trail(client: TestClient, page_html: str) -> None:
    resp = client.post("/api/pages/upload", files={"file": ("page.html", page_html, "text/html")})
    assert resp.status_code == 200, resp.text
    page = resp.json()
    assert page["url"] == "https://acme.example/loans"  # from <link rel=canonical>

    snap = client.get(f"/api/pages/{page['page_id']}/snapshot")
    csp = snap.headers["content-security-policy"]
    assert "script-src 'none'" in csp
    assert "frame-ancestors 'self' http://127.0.0.1:5173" in csp  # console on its own origin

    preflight = client.options(
        "/api/chat",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    resp = client.post(
        "/api/chat",
        json={
            "page_id": page["page_id"],
            "session_id": "session-0001",
            "message": "Who is eligible for the personal loan, what age?",
        },
    )
    events = sse_events(resp.text)
    names = [e[0] for e in events]
    assert names[0] == "start" and names[-1] == "answer"
    assert [e[1]["node"] for e in events if e[0] == "node"][-1] == "write"
    answer = events[-1][1]
    assert answer["route"] == "answered"

    run = client.get(f"/api/runs/{answer['run_id']}").json()
    assert run["output_tokens"] == 45


def test_unsafe_url_rejected(client: TestClient) -> None:
    resp = client.post("/api/pages", json={"url": "http://127.0.0.1:22"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "unsafe_url"


def test_page_id_path_traversal_rejected(client: TestClient) -> None:
    assert client.get("/api/pages/..%2F..%2Fetc/snapshot").status_code in {404, 422}


def test_config_lists_pipeline_nodes(client: TestClient) -> None:
    nodes = [n["id"] for n in client.get("/api/config").json()["nodes"]]
    assert nodes == [
        "pre_guardrail",
        "retrieve",
        "prompt_builder",
        "generate",
        "post_guardrail",
        "write",
    ]


def test_root_redirects_to_console(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "http://127.0.0.1:5173"
