# Decisions, trade-offs and open challenges

Each entry covers what was asked, what was built, and why. Entries marked **❓ Open** need a
product decision from you.

---

## 1. ❓ "Hosted locally" conflicts with "hosted on any webpage"

A model on a MacBook Air can't serve real customers. `localhost` isn't reachable from a
visitor's browser, and a 7B model on 16 GB generates one answer at a time at roughly 20–30 tok/s
(we queue with `llm_concurrency=1`). That works for a demo console and fails at around three
concurrent visitors.

**Built:** a local-first app whose only coupling to the model is an OpenAI-compatible URL.
**Decision needed:** is this a **local demo/PoC**, or a **product**? For a product, run the
*same* open-source model on one GPU box with vLLM or TGI (also OpenAI-compatible). That's a
config change with no code change, and data still stays in your infrastructure.

## 2. The left pane can't be a live iframe

nubazo.com sends `X-Frame-Options: DENY` and `frame-ancestors 'none'`, as most production
sites do. **Built:** a sanitised snapshot served from our origin with scripts stripped,
`script-src 'none'` CSP, and a sandboxed iframe. Content that sites hide until JavaScript
reveals it (nubazo's `.reveal { opacity: 0 }`) is forced visible. **Consequence:** the left
pane is a frozen copy, so carousels, forms and logged-in states don't work there.

## 3. ❓ Page scope is thinner than you'd expect

The nubazo.com home page yields only about 8 k characters (20 chunks). Real visitors ask about
pricing, careers, contact details and security, and those live on *other* pages. Strict
page-scope will produce many honest "I couldn't find that on this page" replies.
**Decision needed:** page scope (as requested) or **site scope**, meaning a same-origin crawl of
the sitemap capped at N pages, with retrieval still confined to that site. The index-per-scope
design supports either.

## 4. No token streaming to the customer, by design

Post-guardrails need the whole answer. Streaming tokens and then retracting them means the
customer has *already seen* the unsafe text. **Built:** the console streams node progress
over SSE, and the answer is released only after it is verified. The cost is about 2–6 s of
wait on a 7B model for a typical answer. *Alternative if latency matters more:* sentence-level
release, where each sentence is checked before it is sent.

## 5. The relevance gate is a cross-encoder, measured rather than guessed

Cosine similarity from `bge-small` can't separate on-page from off-page questions. On nubazo.com
the margin was 0.005. On a Poonawalla loan page, "Who is the captain of India?" scored **0.593**
and passed the gate, while "What is the eligibility criteria?" scored only 0.568. The LLM then
abstained but added outside-world advice ("check sports news").
**Built:**

- **Cross-encoder gate.** `bge-reranker-base` rescores the retrieved chunks, and the gate uses
  the best rerank score. Measured with `make eval`, the margin is **+0.72** on nubazo.com and
  **+2.24** on the Poonawalla page (min on-page −4.29, max off-page −5.01). The threshold is
  −4.65. It costs about 300 ms on CPU and ~1 GB of RAM. Setting `JA_RERANKER_MODEL=""` falls
  back to the cosine gate.
- **Abstention backstop.** If the answer *opens* by saying the page doesn't cover the question,
  the canned off-page reply replaces the whole answer. Small models abstain and then answer
  anyway. *Trade-off:* "I couldn't find X, but the page says Y" is also replaced; answers that
  lead with content and abstain later are kept.

*Next step:* build an eval set from **real customer questions** for each site. Domain-adjacent
questions ("current repo rate in India?") remain the closest call. Run `make eval
EVAL=evals/<site>.json PAGE_ID=<id>` against the stored index, because live fetches can return a
different page variant (the Poonawalla page without its ad parameters has no eligibility
section).

### 5a. Reply language is chosen in code, not by the model

Told only to "reply in the visitor's language", qwen2.5-7b answered "Tell me about 'Other
Loan'" in **Portuguese 5 times in 8**. Naming the language at the end of the user turn ("Reply
in English.") fixed it in every trial. "Reply in the same language as the question" helped but
still drifted once in 12. **Built:**

- **Language detection over a closed set.** `lingua`, restricted to `JA_REPLY_LANGUAGES`
  (default English and Hindi), picks the language, and the prompt names it. Open-set detection
  is confidently wrong on short or romanised input: "gold loan ka interest rate kitna hai?"
  comes out as Romanian 0.95 and "ok thanks" as Afrikaans. Hinglish and unsupported languages
  therefore get the site default (English).
- **`reply_language` post-check.** An answer of six words or more that is clearly in a
  different language falls back. Long text is reliable to detect against the open set.
- **The reranker gate is English-only (`JA_RERANKER_LANGUAGES`).** `bge-reranker-base` refused
  every Hindi question on an English page. The multilingual rerankers (`bge-reranker-v2-m3`,
  `jina-reranker-v2-base-multilingual`) still overlapped on Hindi, cross-lingual, in
  `evals/poonawalla_home.json`. Hindi questions therefore use the cosine gate, plus the
  grounding rules and post-guardrails.

*Next step:* translate non-English questions to English (one short LLM call) before
retrieval and the gate. 7B Hindi output is also noticeably weaker than its English, so a
bigger model or human review of Hindi canned replies matters before a Hindi launch. The
canned replies (off-page, fallback) are English-only today.

## 6. Similarity-based grounding misses invented facts

A fabricated answer ("flat fee of $499/month, 30-day trial") scored 0.554, which counts as
"grounded" because the wording is on-topic. **Built:** a deterministic
**numeric-grounding** check, under which every number in the answer must appear in the
retrieved context or the question. In BFSI, invented rates and fees are the costliest
hallucination.

## 7. Guardrails catch accidents, not determined attackers

Regex injection checks are bypassed by paraphrase or another language. **The real safety
boundary is architectural:** the model has no tools, no secrets and no private data, only
public page text. The worst case is an off-brand answer, not a breach. **Keep it that way:**
don't add actions such as "book a demo" or "check my application" without designing authz
first. One concrete failure we hit: the 0.5B model *asked a user for their PAN* to book a
demo. That's now blocked by the `sensitive_solicitation` post-check.

## 8. ❓ Response payloads can leak personal data

A HAR captured **while logged in** contains *that user's* API responses, and the index is
shared by every visitor. **Built:** PII is redacted from payloads at ingest, and HAR
cookies/headers are never read or stored. **Policy needed:** capture HARs logged out, and
consider an allow-list of payload URLs per site.

## 9. ❓ The audit log is a personal-data store (DPDP Act 2023)

The write node stores prompts, messages and responses. Messages are PII-redacted *before*
storage, but free text can still identify people. **Decision needed:** a retention period
(for example 30 days, then purge), a consent notice in the widget, and who can read `data/`.

## 10. Editing the persona is a demo feature

`JA_ALLOW_PROMPT_OVERRIDE=true` lets the console edit the persona. The locked grounding rules
are appended regardless, but in production the server can't tell the console from the
widget. **Set it to `false` for anything exposed**, or put the console behind auth.

## 11. ❓ "Nodes in the UI": inspect or edit?

**Built:** every node is visible live, with its inputs, outputs and latency. The prompt
builder exposes its three parts (system prompt, retrieval, payload inputs), and the persona
is editable. **Not built:** a drag-and-drop graph editor (Langflow-style). That is a
different product, needing versioned configs, validation and rollback. Tell me if that's the
actual goal.

## 12. Smaller calls

| Choice | Why |
|---|---|
| Vanilla JS, no build step | No Node on the target machine and around 700 lines of UI; move to React+Vite if the console grows auth or editing |
| FAISS `IndexFlatIP`, one per page | Exact search is faster *and* more accurate than ANN below roughly 100 k vectors |
| fastembed (ONNX) rather than sentence-transformers | No PyTorch (about 2 GB saved), and memory stays free for the LLM |
| SQLite (WAL) | Zero-ops and inspectable; the `Repository` class keeps a Postgres move contained |
| History read from `runs`, no LangGraph checkpointer | Survives restarts, and memory is the same data you audit |
| Deterministic guardrails, no LLM judge | An LLM judge on a laptop roughly doubles latency; revisit on GPU hosting |
| SSRF guard with per-redirect checks | The server fetches user-supplied URLs; blocks `169.254.169.254`, LAN and localhost |

## Not done yet (before any production use)

- AuthN for the console, rate limiting per session/IP, and CORS locked to embedding origins
- Scheduled re-ingest (pages change; use ETag/Last-Modified)
- An answer-quality eval set (golden Q&A per page) wired into CI
- Hindi/Hinglish guardrail patterns if your traffic is bilingual
