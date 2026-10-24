// Chat controller shared by the console and the embeddable widget.

import { el, renderMarkdown } from "./render.js";

function sessionFor(pageId) {
  const key = `ja-session-${pageId}`;
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

export function createChat({ api, messagesEl, suggestionsEl, formEl, inputEl, sendBtn, hooks = {} }) {
  let page = null;
  let busy = false;

  const scroll = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

  function addMessage(role, text, meta) {
    const bubble = el("div", { class: `msg ${role}` });
    if (role === "ai") bubble.innerHTML = renderMarkdown(text);
    else bubble.textContent = text;
    if (meta) bubble.append(meta);
    messagesEl.append(bubble);
    scroll();
    return bubble;
  }

  function setEnabled(on) {
    inputEl.disabled = !on || busy;
    sendBtn.disabled = !on || busy;
  }

  function renderSuggestions(topics) {
    const title = page.title.trim().toLowerCase();
    suggestionsEl.replaceChildren(
      ...topics.filter((t) => t.trim().toLowerCase() !== title).slice(0, 4).map((topic) =>
        el("button", { class: "chip", type: "button", title: topic, onclick: () => send(`Tell me about “${topic}”`) }, topic)),
    );
  }

  async function send(text) {
    const message = text.trim();
    if (!page || !message || busy) return;
    busy = true;
    setEnabled(true);
    inputEl.value = "";
    suggestionsEl.replaceChildren();
    addMessage("user", message);
    const pending = addMessage("ai", "Thinking…");
    pending.classList.add("thinking");
    let runId = null;

    try {
      await api.chat(
        {
          page_id: page.page_id,
          session_id: sessionFor(page.page_id),
          message,
          persona_override: hooks.persona?.() || null,
        },
        (event, data) => {
          if (event === "start") { runId = data.run_id; hooks.onStart?.(data); }
          else if (event === "node") hooks.onNode?.(data);
          else if (event === "answer") {
            pending.classList.remove("thinking");
            pending.innerHTML = renderMarkdown(data.answer);
            const meta = el("div", { class: "msg-meta" },
              el("span", { class: `badge ${data.route}` }, data.route.replace("_", " ")));
            if (hooks.onInspectRun) {
              meta.append(el("button", { type: "button", onclick: () => hooks.onInspectRun(runId) }, "inspect run"));
            }
            pending.append(meta);
            hooks.onAnswer?.(data);
          } else if (event === "error") {
            pending.classList.remove("thinking");
            pending.textContent = data.detail;
          }
        },
      );
    } catch (err) {
      pending.classList.remove("thinking");
      pending.textContent = `Something went wrong: ${err.message}`;
    } finally {
      busy = false;
      setEnabled(true);
      inputEl.focus();
      scroll();
    }
  }

  // Keep the latest message visible when the inspector opens and shrinks the chat.
  new ResizeObserver(scroll).observe(messagesEl);

  formEl.addEventListener("submit", (e) => { e.preventDefault(); send(inputEl.value); });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(inputEl.value); }
  });
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${inputEl.scrollHeight}px`;
  });

  return {
    setPage(next) {
      page = next;
      messagesEl.replaceChildren();
      inputEl.placeholder = "Ask about this page…";
      setEnabled(true);
      renderSuggestions(next.topics ?? []);
      // The greeting is LLM-written once per page on the server; show typing until it lands.
      const bubble = addMessage("ai", "");
      bubble.classList.add("typing");
      bubble.setAttribute("aria-label", "Assistant is typing");
      bubble.innerHTML = "<span></span><span></span><span></span>";
      api.welcome(next.page_id)
        .catch(() => ({ company_name: next.title, message: "Hi! Ask me anything about this page." }))
        .then((w) => {
          if (page !== next) return; // another page was opened meanwhile
          bubble.classList.remove("typing");
          bubble.removeAttribute("aria-label");
          // renderMarkdown escapes first, so bolding the name this way is safe.
          bubble.innerHTML = renderMarkdown(w.message.replace(w.company_name, `**${w.company_name}**`));
          scroll();
          hooks.onWelcome?.(w);
        });
    },
    send,
  };
}
