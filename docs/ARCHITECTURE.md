# Architecture

## Request flow

```
Browser (frontend/: console / widget, served from its own origin)
   │  CORS + snapshot frame-ancestors allow JA_FRONTEND_ORIGINS
   │  POST /api/chat  ── SSE stream: start → node × N → answer
   ▼
FastAPI ──► LangGraph (compiled once at startup)
              │
              ├─ pre_guardrail   deterministic, <1 ms; loads session history from SQLite
              ├─ retrieve        FAISS (per page) · content+meta and payload searched separately
              │                  relevance gate: best cross-encoder rerank < threshold → off_topic
              ├─ prompt_builder  persona + locked grounding rules + <page_content> + <page_data>
              │                  + last N turns, trimmed to a char budget
              ├─ generate        OpenAI-compatible local server; semaphore = 1 concurrent call
              ├─ post_guardrail  abstention → canned reply · solicitation · prompt leak · PII · link allow-list ·
              │                  numeric grounding · semantic grounding
              └─ write           RunRecord → SQLite (always runs; never fails the reply)
```

## Key properties

- **Page isolation.** One FAISS index per page. Retrieval physically cannot return another
  page's content, which is a stronger guarantee than metadata filtering.
- **No unverified text reaches the user.** The answer is released only after the post-guardrails
  run. The UI streams *node progress* rather than tokens. See DECISIONS §4.
- **Raw PII is never stored or sent to the model.** The pre-guardrail redacts first, and only the
  redacted text flows onwards (asserted in `tests/test_graph.py`).
- **Injection defence in depth:** regex pre-filter, then locked grounding rules appended to every
  system prompt, then page text fenced as data (`<page_content>`), then prompt-leak and
  solicitation post-checks.
- **Stateless graph, durable memory.** Conversation history is read from the `runs` table rather
  than an in-memory checkpointer, so it survives restarts and is the same data you audit.
- **Swappable edges.** `ChatModel` and `Embedder` are Protocols, the `Repository` hides SQLite, and
  all dependencies are composed in `container.py`. Tests inject fakes.

## Data model (SQLite)

`pages(page_id, url, title, fetched_with, chunk_count, payload_count, topics, …)`

`runs(run_id, session_id, page_id, route, user_message [redacted], ai_response, model,
system_prompt, prompt_messages, retrieved, guardrails, node_latency_ms, input_tokens,
output_tokens, tokens_estimated, total_latency_ms, grounding_score, error, created_at)`

`route` is one of `answered · smalltalk · blocked · off_topic · fallback · error`.

## Files on disk

```
backend/data/journey_agent.sqlite3        pages + runs (WAL)
backend/data/indexes/<page_id>/index.faiss
backend/data/indexes/<page_id>/chunks.json  (records which embedder built it, mismatch → re-ingest)
backend/data/snapshots/<page_id>.html       sanitised snapshot for the left pane
```
