// Console: the page with its assistant as a visitor sees it (left) + the live LangGraph (right).

import { createApi } from "./api.js";
import { createChat } from "./chat.js";
import { el, escapeHtml, ms } from "./render.js";
import { applyChatTheme, tintAnimation } from "./theme.js";

const api = createApi();
const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  page: null,
  trace: {}, // node id -> latest NodeTrace
  expanded: new Set(), // node ids whose details are open; a finished run opens all of them
  run: null, // { status: "running" | route, run_id } for the latest turn
  theme: null, // the loaded page's brand colour (#rrggbb) or null
  persona: null, // edited persona, null = default
  promptTab: "system_prompt",
};

// --- notices -------------------------------------------------------------------
function notify(text, kind = "warn") {
  const box = $("notice");
  box.className = `notice ${kind}`;
  box.innerHTML = text; // callers escape dynamic parts
  box.hidden = false;
}
const clearNotice = () => { $("notice").hidden = true; };

// --- health --------------------------------------------------------------------
async function refreshHealth() {
  const status = $("llm-status");
  try {
    const h = await api.health();
    status.className = `status ${h.llm_reachable ? "ok" : "bad"}`;
    status.querySelector(".status-text").textContent = h.llm_reachable ? h.model : `${h.model} · offline`;
    status.title = h.llm_reachable ? `Model server reachable · ${h.embedder}`
      : "Model server not reachable. Start Ollama (see README).";
  } catch {
    status.className = "status bad";
    status.querySelector(".status-text").textContent = "API offline";
  }
}

// --- LangGraph pane ---------------------------------------------------------------
const PROMPT_PARTS = [
  ["system_prompt", "system"],
  ["retrieval_context", "retrieval"],
  ["payload_inputs", "payload"],
];

const order = () => state.config.nodes.map((n) => n.id);
const nodeStatus = (t) => t?.status ?? t?.ui ?? "idle";

function nodeTime(t) {
  if (t?.latency_ms != null) return ms(t.latency_ms);
  return { running: "running…", skipped: "skipped" }[t?.ui] ?? "";
}

function nodeCard(node) {
  const t = state.trace[node.id];
  const status = nodeStatus(t);
  const open = state.expanded.has(node.id);
  const summary = t?.summary ?? (status === "skipped" ? "skipped: an earlier node short-circuited" : node.description);
  // A conditional edge fired if the node after this one never ran.
  const next = order()[order().indexOf(node.id) + 1];
  const jumped = t?.latency_ms != null && next && state.trace[next]?.ui === "skipped";
  return el("li", { class: `gnode ${status}`, "data-node": node.id },
    el("span", { class: "rail-dot", "aria-hidden": "true" }),
    el("div", { class: "gnode-card" },
      el("button", { class: "gnode-head", type: "button", "aria-expanded": String(open), title: node.description, onclick: () => toggleNode(node.id) },
        el("span", { class: "gnode-label" }, node.label),
        el("span", { class: "gnode-summary" }, summary),
        el("span", { class: "gnode-time" }, nodeTime(t)),
        el("span", { class: "chev", "aria-hidden": "true" })),
      open ? el("div", { class: "gnode-body" }, t?.summary ? el("p", { class: "desc" }, node.description) : null, nodeDetails(node, t)) : null),
    jumped ? el("div", { class: "edge-note" }, `↳ jumps to Write · ${state.run?.status?.replace("_", " ") ?? status}`) : null);
}

function renderGraph() {
  $("graph").replaceChildren(...state.config.nodes.map(nodeCard));
  renderRunBar();
}

// Re-render one node in place so an open editor elsewhere (the persona) keeps its text.
function renderNode(nodeId) {
  const node = state.config.nodes.find((n) => n.id === nodeId);
  const li = $("graph").querySelector(`[data-node="${nodeId}"]`);
  if (node && li) li.replaceWith(nodeCard(node));
}

function toggleNode(nodeId) {
  if (!state.expanded.delete(nodeId)) state.expanded.add(nodeId);
  renderNode(nodeId);
}

function setAllExpanded(open) {
  state.expanded = new Set(open ? order() : []);
  renderGraph();
}
$("expand-all").addEventListener("click", () => setAllExpanded(true));
$("collapse-all").addEventListener("click", () => setAllExpanded(false));

function renderRunBar() {
  const w = state.trace.write?.details;
  const run = state.run;
  const stats = [];
  if (!run) {
    stats.push(el("span", { class: "hint" }, state.page
      ? "No run yet. Ask the assistant (bottom-right of the page) to see each node live."
      : "Load a page to start."));
  } else {
    stats.push(el("span", { class: `badge ${run.status}` }, run.status.replace("_", " ")));
    if (w) {
      stats.push(
        el("span", { class: "stat" }, el("b", {}, ms(w.total_latency_ms)), " total"),
        el("span", { class: "stat" }, el("b", {}, `${w.input_tokens} / ${w.output_tokens}`), w.tokens_estimated ? " tokens (est.)" : " tokens"),
        el("span", { class: "stat" }, w.model ?? "no LLM call"),
        w.grounding_score != null ? el("span", { class: "stat" }, "grounding ", el("b", {}, w.grounding_score.toFixed(2))) : null);
    }
    if (run.run_id) stats.push(el("span", { class: "stat", title: run.run_id }, `run ${run.run_id.slice(0, 8)}`));
  }
  $("run-bar").replaceChildren(...stats.filter(Boolean)); // replaceChildren would print null as text
}

function checksView(checks = []) {
  return el("div", { class: "checks" }, checks.map((c) =>
    el("div", { class: "check" }, el("b", {}, c.name), el("span", { class: `badge ${c.verdict}` }, c.verdict), el("span", {}, c.detail || ""))));
}

function hitsView(title, hits = []) {
  if (!hits.length) return el("p", { class: "desc" }, `${title}: none`);
  return el("div", {},
    el("p", { class: "eyebrow" }, title),
    el("div", { class: "hits" }, hits.map((h) =>
      el("div", { class: "hit" },
        el("div", { class: "hit-head" }, el("span", {}, `${h.source} · ${h.heading}`), el("span", { class: "score" }, h.rerank == null ? h.score.toFixed(3) : `cos ${h.score.toFixed(3)} · rerank ${h.rerank.toFixed(2)}`)),
        el("div", {}, h.text.length > 360 ? `${h.text.slice(0, 360)}…` : h.text)))));
}

function kvView(obj) {
  return el("dl", { class: "kv" }, Object.entries(obj).flatMap(([k, v]) =>
    [el("dt", {}, k), el("dd", {}, typeof v === "object" && v !== null ? JSON.stringify(v) : String(v ?? "—"))]));
}

function promptBuilderView(details) {
  const tabs = el("div", { class: "subtabs" }, PROMPT_PARTS.map(([key, label]) =>
    el("button", {
      class: `subtab ${state.promptTab === key ? "active" : ""}`, type: "button",
      onclick: () => { state.promptTab = key; renderNode("prompt_builder"); },
    }, label)));

  let body;
  if (state.promptTab === "system_prompt") {
    const editable = state.config.allow_prompt_override;
    const area = el("textarea", { class: "code", spellcheck: "false", readonly: !editable });
    area.value = state.persona ?? state.config.default_persona;
    body = el("div", {},
      el("p", { class: "desc" }, editable
        ? "Persona (editable — applies to your next message). {title} and {url} are filled in from the page."
        : "Persona (editing disabled by server config)."),
      area,
      editable ? el("div", { class: "row-actions" },
        el("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => {
          state.persona = area.value.trim() === state.config.default_persona ? null : area.value;
          notify("Persona updated for the next message.", "ok");
          setTimeout(clearNotice, 2000);
        } }, "Apply"),
        el("button", { class: "btn btn-outline btn-sm", type: "button", onclick: () => { state.persona = null; renderNode("prompt_builder"); } }, "Reset")) : null,
      el("p", { class: "eyebrow mt" }, "Locked grounding rules (always appended)"),
      el("pre", { class: "code" }, state.config.grounding_rules),
      details?.system_prompt ? el("p", { class: "eyebrow mt" }, "Rendered system prompt (last run)") : null,
      details?.system_prompt ? el("pre", { class: "code" }, details.system_prompt) : null);
  } else {
    const text = details?.[state.promptTab];
    body = el("pre", { class: "code" }, text || (details ? "(empty for this run)" : "Run a message to see this part of the prompt."));
  }
  const stats = details ? el("p", { class: "summary" },
    `~${details.estimated_prompt_tokens} tokens · ${details.history_turns} history turns · reply in ${details.reply_language ?? "—"} · chunks ${details.chunks_used.join(", ") || "none"}`) : null;
  return el("div", {}, tabs, stats, body);
}

function writeNodeView(details) {
  const wrap = el("div", {}, kvView(details));
  wrap.append(el("div", { class: "row-actions" },
    el("button", { class: "btn btn-outline btn-sm", type: "button", onclick: async () => {
      const run = await api.run(details.run_id);
      wrap.append(el("pre", { class: "code mt" }, JSON.stringify(run, null, 2)));
    } }, "Show full stored record")));
  return wrap;
}

function nodeDetails(node, t) {
  const d = t?.details;
  let body = null;
  switch (node.id) {
    case "pre_guardrail":
      body = d ? el("div", {}, checksView(d.checks),
        el("p", { class: "eyebrow mt" }, `Text forwarded (PII-redacted) · reply in ${d.reply_language ?? "—"}`),
        el("pre", { class: "code" }, d.redacted_input)) : null;
      break;
    case "retrieve":
      body = d ? el("div", {}, el("p", { class: "summary" }, `query: “${d.query}” · ${d.gate ?? "cosine"} threshold ${d.threshold}`),
        hitsView("Page content", d.content_hits), hitsView("Response payloads", d.payload_hits)) : null;
      break;
    case "prompt_builder":
      return promptBuilderView(d);
    case "generate":
      body = d ? el("div", {},
        d.error && /connection|connect|refused/i.test(d.error)
          ? el("div", { class: "notice error node-hint" },
            el("b", {}, "Model server not reachable. "),
            "Start it in a terminal: ", el("code", {}, "OLLAMA_CONTEXT_LENGTH=8192 ollama serve"),
            " (or ", el("code", {}, "make model-mlx"), " with JA_LLM_BASE_URL pointed at it). The status pill in the top bar turns green when it's up.")
          : null,
        d.error ? null : el("p", { class: "summary" }, `LLM latency ${ms(d.llm_latency_ms)}`),
        el("pre", { class: "code" }, d.raw_answer ?? d.error)) : null;
      break;
    case "post_guardrail":
      body = d ? el("div", {}, checksView(d.checks), el("p", { class: "eyebrow mt" }, "Text released"), el("pre", { class: "code" }, d.final_text)) : null;
      break;
    case "write":
      body = d ? writeNodeView(d) : null;
      break;
  }
  return body ?? el("p", { class: "desc" }, t?.ui === "skipped"
    ? "Skipped on the last run: an earlier node short-circuited to Write."
    : "Send a message to see this node's live inputs and outputs.");
}

// --- assistant launcher -----------------------------------------------------------
let launcherAnim = null;
let mascotJson = null; // untinted animation source; lottie mutates the objects it is given
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// (Re)build the Lottie mascot in the page's colour. Lottie can't recolour a running
// animation, so a theme change rebuilds it from the tinted data.
function renderMascot() {
  if (!mascotJson) return;
  launcherAnim?.destroy();
  const still = reducedMotion();
  launcherAnim = window.lottie.loadAnimation({
    container: $("bot-anim"), renderer: "svg", loop: true, autoplay: !still && $("chat-shell").hidden,
    animationData: state.theme ? tintAnimation(JSON.parse(mascotJson), state.theme) : JSON.parse(mascotJson),
  });
  if (still) launcherAnim.goToAndStop(20, true); // a frame with the bubbles in view
}

// Pure vector, no expressions, and the light player never evals. Any failure leaves
// the static SVG icon in place.
async function mountLauncherAnimation() {
  if (!window.lottie) return;
  try {
    mascotJson = await fetch("assets/chatbot.json").then((r) => r.text());
    renderMascot();
    $("bot-launcher").classList.add("animated");
    $("bot-launcher").addEventListener("mouseenter", () => launcherAnim?.setSpeed(1.6));
    $("bot-launcher").addEventListener("mouseleave", () => launcherAnim?.setSpeed(1));
  } catch { /* keep the static icon */ }
}

async function loadTheme(page) {
  state.theme = null;
  applyChatTheme($("bot"), null);
  $("page-theme").hidden = true;
  let color = null;
  try {
    color = (await api.theme(page.page_id)).theme_color;
  } catch { /* keep the console defaults */ }
  if (state.page?.page_id !== page.page_id) return; // another page was opened meanwhile
  state.theme = color;
  applyChatTheme($("bot"), color);
  renderMascot();
  if (color) {
    $("page-theme").hidden = false;
    $("page-theme").title = `Assistant themed to the page's brand colour ${color}`;
    $("page-theme").style.setProperty("--swatch", color);
  }
}

function setChatOpen(open) {
  const launcher = $("bot-launcher");
  $("chat-shell").hidden = !open;
  launcher.classList.toggle("open", open);
  launcher.setAttribute("aria-expanded", String(open));
  launcher.setAttribute("aria-label", open ? "Close assistant" : "Open assistant");
  if (launcherAnim && !reducedMotion()) {
    open ? launcherAnim.pause() : launcherAnim.play(); // hidden while open; don't burn frames
  }
  if (open) {
    launcher.classList.remove("has-unread");
    $("message-input").focus();
  }
}
$("bot-launcher").addEventListener("click", () => setChatOpen($("chat-shell").hidden));
$("chat-close").addEventListener("click", () => { setChatOpen(false); $("bot-launcher").focus(); });
$("chat-shell").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { setChatOpen(false); $("bot-launcher").focus(); }
});

// --- chat wiring ---------------------------------------------------------------

const chat = createChat({
  api,
  messagesEl: $("messages"),
  suggestionsEl: $("suggestions"),
  formEl: $("composer"),
  inputEl: $("message-input"),
  sendBtn: $("send-btn"),
  hooks: {
    persona: () => state.persona,
    onStart(data) {
      state.trace = { [order()[0]]: { ui: "running" } };
      state.run = { status: "running", run_id: data.run_id };
      switchTab("graph");
      renderGraph();
    },
    onNode(entry) {
      state.trace[entry.node] = entry;
      const ids = order();
      const next = ids[ids.indexOf(entry.node) + 1];
      if (next && !state.trace[next]) state.trace[next] = { ui: "running" };
      renderNode(entry.node);
      if (next) renderNode(next);
    },
    onAnswer(data) {
      for (const id of order()) {
        if (!state.trace[id] || state.trace[id].ui === "running") state.trace[id] = { ui: "skipped" };
      }
      state.run = { status: data.route, run_id: data.run_id ?? state.run?.run_id };
      state.expanded = new Set(order());
      renderGraph();
      if ($("chat-shell").hidden) $("bot-launcher").classList.add("has-unread");
      refreshRuns();
    },
    onInspectRun: (runId) => showRun(runId),
    onWelcome: (w) => { $("chat-title").textContent = w.company_name; },
  },
});

// --- pages -----------------------------------------------------------------------
async function openPage(page) {
  state.page = page;
  state.trace = {};
  state.run = null;
  $("site-frame").src = api.snapshotUrl(page.page_id);
  $("site-empty").hidden = true;
  $("url-input").value = page.url;
  $("page-meta").textContent =
    `${page.url} · ${page.chunk_count} chunks · ${page.payload_count} payloads · via ${page.fetched_with}`;
  history.replaceState(null, "", `?page=${page.page_id}`);
  chat.setPage(page);
  $("chat-title").textContent = page.company_name || page.title;
  $("bot").hidden = false;
  loadTheme(page); // async: the mascot and chat take on the page's colour when it arrives
  renderGraph();
  await refreshPages();
  await refreshRuns();
}

async function refreshPages() {
  const pages = await api.pages();
  const picker = $("page-picker");
  picker.replaceChildren(
    el("option", { value: "" }, pages.length ? "Recent pages…" : "No pages yet"),
    ...pages.map((p) => el("option", { value: p.page_id, selected: state.page?.page_id === p.page_id }, p.title)),
  );
  picker.onchange = () => {
    const page = pages.find((p) => p.page_id === picker.value);
    if (page) openPage(page);
  };
}

function needsUpload(detail) {
  notify(`<b>Couldn't read this page automatically.</b> ${escapeHtml(detail)}<br>
    Upload a <b>.har</b> (DevTools → Network → right-click → “Save all as HAR with content”) to include API
    response payloads, or a <b>.html</b> (File → Save Page As). Use the <b>Upload HTML / HAR</b> button.`);
}

async function withBusy(btn, label, fn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = label;
  try { await fn(); } finally { btn.disabled = false; btn.textContent = original; }
}

$("ingest-form").addEventListener("submit", (e) => {
  e.preventDefault();
  clearNotice();
  withBusy($("ingest-btn"), "Reading page…", async () => {
    try {
      await openPage(await api.ingest($("url-input").value));
    } catch (err) {
      if (err.code === "needs_upload") needsUpload(err.message);
      else notify(escapeHtml(err.message), "error");
    }
  });
});

$("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  clearNotice();
  withBusy($("ingest-btn"), "Uploading…", async () => {
    try {
      await openPage(await api.upload(file, $("url-input").value.trim() || null));
    } catch (err) {
      notify(escapeHtml(err.message), "error");
    }
  });
});

// --- runs tab ----------------------------------------------------------------------
async function refreshRuns() {
  if (!state.page) return;
  const runs = await api.runs(state.page.page_id);
  $("runs-count").textContent = runs.length;
  const rows = runs.map((r) => el("tr", { onclick: () => showRun(r.run_id) },
    el("td", { class: "num" }, new Date(r.created_at).toLocaleTimeString()),
    el("td", {}, el("span", { class: `badge ${r.route}` }, r.route)),
    el("td", { class: "clip", title: r.user_message }, r.user_message),
    el("td", { class: "clip", title: r.ai_response }, r.ai_response),
    el("td", { class: "num" }, `${r.input_tokens} / ${r.output_tokens}${r.tokens_estimated ? "*" : ""}`),
    el("td", { class: "num" }, ms(r.total_latency_ms)),
    el("td", { class: "num" }, r.grounding_score == null ? "—" : r.grounding_score.toFixed(2))));
  $("runs").replaceChildren(
    el("table", { class: "grid" },
      el("thead", {}, el("tr", {}, ["Time", "Route", "Human", "AI", "Tokens", "Latency", "Ground."].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows)),
    el("div", { id: "run-detail" }),
  );
}

async function showRun(runId) {
  switchTab("runs");
  await refreshRuns();
  const run = await api.run(runId);
  $("run-detail").replaceChildren(
    el("p", { class: "eyebrow", style: "margin:16px 0 8px" }, `Run ${run.run_id}`),
    el("pre", { class: "code" }, JSON.stringify(run, null, 2)));
  $("run-detail").scrollIntoView({ behavior: "smooth" });
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  $("graph-actions").hidden = name !== "graph";
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

// --- resizable split ---------------------------------------------------------------
(() => {
  const divider = $("divider");
  const split = $("split");
  const setLeft = (pct) => document.documentElement.style.setProperty("--left", `${Math.min(75, Math.max(25, pct))}%`);
  divider.addEventListener("pointerdown", (e) => {
    divider.setPointerCapture(e.pointerId);
    $("site-frame").style.pointerEvents = "none";
    const move = (ev) => {
      const rect = split.getBoundingClientRect();
      setLeft(((ev.clientX - rect.left) / rect.width) * 100);
    };
    const up = () => {
      divider.removeEventListener("pointermove", move);
      $("site-frame").style.pointerEvents = "";
    };
    divider.addEventListener("pointermove", move);
    divider.addEventListener("pointerup", up, { once: true });
  });
  divider.addEventListener("keydown", (e) => {
    const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--left"));
    if (e.key === "ArrowLeft") setLeft(cur - 2);
    if (e.key === "ArrowRight") setLeft(cur + 2);
  });
})();

// --- boot ----------------------------------------------------------------------------
(async function boot() {
  state.config = await api.config();
  renderGraph();
  mountLauncherAnimation();
  refreshHealth();
  setInterval(refreshHealth, 15000);
  await refreshPages();
  const pageId = new URLSearchParams(location.search).get("page");
  if (pageId) {
    try { await openPage(await api.page(pageId)); } catch { /* stale link */ }
  }
  if (!state.config.browser_fetch) {
    $("url-input").title = "Static fetch only. Install the browser extra to capture JS-rendered content and XHR payloads.";
  }
})();
