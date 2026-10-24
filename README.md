# Nubazo Journey Agent

A **locally hosted, page-scoped customer assistant**. Point it at any web page and
it answers visitor questions using only what that page contains: its visible text,
its structured data, and the JSON payloads it loads. It refuses anything else.

- **Backend:** FastAPI · LangGraph · FAISS · fastembed · any OpenAI-compatible local LLM server
- **Model:** open-source and runs on a MacBook Air (16 GB) via Ollama or MLX
- **Frontend:** static HTML/JS with no build step, themed on nubazo.com and served separately
  from the API. The web page is on the left, with the assistant as a visitor sees it: a bot
  launcher in the lower-right corner that opens the chat. The live LangGraph is on the right,
  showing every node's checks, retrieval scores, prompt parts, raw output and stored record.

```
backend/    Python API (uv project): FastAPI, LangGraph, RAG, guardrails, tests, evals
frontend/   static console, embed view and widget.js; API base URL in frontend/config.js
```

```
pre_guardrail ─► retrieve ─► prompt_builder ─► generate ─► post_guardrail ─► write
     │              │       (system · retrieval ·  │                         ▲
     └─ blocked ────┴─ off-page   payload inputs)  └─ LLM error ─────────────┘
```

Every path ends in `write`, so every turn is persisted: prompts, redacted human
message, AI response, tokens in and out, per-node latency, guardrail verdicts and
grounding score. That includes refusals and failures.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and one local model server.

```bash
# 1. Model server: pick ONE
brew install ollama && OLLAMA_CONTEXT_LENGTH=8192 ollama serve   # terminal A
ollama pull qwen2.5:7b-instruct
#   …or MLX (Apple-native, no Homebrew, served via uv):
make model-mlx   # then: export JA_LLM_BASE_URL=http://localhost:8080/v1 JA_LLM_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit

# 2. App
make install                         # backend/.venv from backend/uv.lock
make browser                         # optional but recommended: JS pages + XHR payloads
cp backend/.env.example backend/.env
make dev                             # API on :8000, console on http://127.0.0.1:5173
```

`make backend` and `make frontend` run either half on its own. If you serve the frontend
from somewhere else, set `apiBase` in `frontend/config.js` and add the frontend's origin to
`JA_FRONTEND_ORIGINS` in `backend/.env`. The backend uses that list for CORS and for the
snapshot's `frame-ancestors`.

Paste `https://nubazo.com` and click **Load page**, then click the bot icon at the bottom-right
of the page and ask away.

> The first start downloads the ~130 MB embedding model and the ~1 GB reranker that gates
> off-page questions (set `JA_RERANKER_MODEL=` to skip it). `uvx` needs `--python 3.12`
> for MLX because MLX has no wheels for newer Pythons yet.

### Choosing a model for 16 GB unified memory

| Model (4-bit) | RAM | Notes |
|---|---|---|
| `qwen2.5:7b-instruct` (default) | ~4.7 GB | Best instruction-following/grounding at this size; multilingual (Hindi ok) |
| `llama3.1:8b` | ~4.9 GB | Strong English, slightly chattier |
| `qwen2.5:3b` / `llama3.2:3b` | ~2 GB | 2–3× faster; noticeably weaker at refusing |
| `Qwen2.5-0.5B` | ~0.4 GB | Smoke tests only. It hallucinates, which is useful for exercising guardrails |

Leave around 8 GB for macOS, the browser and the embedder. Don't run more than one 7B model.

## How it reads a page

| Source | How | Gets response payloads? |
|---|---|---|
| URL, browser mode | headless Chromium renders the page and records every XHR/fetch JSON response | ✅ |
| URL, static mode | single HTTP GET (used when Chromium isn't installed) | inline JSON only (`__NEXT_DATA__`, JSON-LD) |
| **HAR upload** | DevTools → Network → *Save all as HAR with content* | ✅ (best fallback for bot-protected sites) |
| HTML upload | *File → Save Page As* | inline JSON only |

If a site blocks the fetch or returns an empty JavaScript shell, the UI asks for a HAR or HTML upload.

The left pane shows a **sanitised snapshot**, not a live iframe. Most sites, including
nubazo.com, send `X-Frame-Options: DENY`, so they can't be framed. The snapshot keeps the
CSS, strips every script, and is served with `script-src 'none'` inside a sandboxed iframe.

## Embedding on a site

```html
<script src="http://localhost:5173/widget.js" defer></script>
```

The widget is served by the frontend. It looks up the ingested page by its URL and opens
`embed.html` in an isolated iframe on the frontend origin, which then calls the API.
For anything customer-facing, read [docs/DECISIONS.md](docs/DECISIONS.md) first (hosting,
CORS, prompt override).

## Development

```bash
make check      # ruff format --check, ruff lint, mypy --strict, pytest
make test       # tests use fake LLM/embedder, so no downloads or GPU are needed
make eval       # calibrate the off-page gate against nubazo.com (EVAL=…, PAGE_ID=… for other pages)
```

API docs are at `http://127.0.0.1:8000/docs`. The SQLite audit DB is at `backend/data/journey_agent.sqlite3`.

```
backend/src/journey_agent/
  ingestion/   fetch (static / Chromium), HAR, extraction, chunking, snapshot sanitiser, SSRF guard
  rag/         embedders (fastembed / OpenAI-compatible), cross-encoder reranker, per-page FAISS indexes
  guardrails/  pre (injection, abuse, PII, small talk) · post (solicitation, prompt leak, PII, links, numbers, grounding)
  graph/       LangGraph state, nodes, wiring, prompts
  llm/         OpenAI-compatible chat client (Ollama, MLX, LM Studio, llama.cpp)
  persistence/ SQLite repository (pages, runs)
  api/         FastAPI routes (SSE chat stream)
frontend/      console (index.html), embed view (embed.html), widget.js, config.js
  vendor/      pinned lottie-web light player (launcher animation); sources + checksums in vendor/README.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
[docs/DECISIONS.md](docs/DECISIONS.md) for the trade-offs and open questions.
