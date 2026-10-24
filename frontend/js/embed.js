// Chat-only view loaded inside the widget iframe: /embed.html?url=<host page URL>

import { createApi } from "./api.js";
import { createChat } from "./chat.js";
import { applyChatTheme } from "./theme.js";

const api = createApi();
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const chat = createChat({
  api,
  messagesEl: $("messages"),
  suggestionsEl: $("suggestions"),
  formEl: $("composer"),
  inputEl: $("message-input"),
  sendBtn: $("send-btn"),
  hooks: { onWelcome: (w) => { $("title").textContent = w.company_name; } },
});

(async () => {
  try {
    const page = params.get("page")
      ? await api.page(params.get("page"))
      : await api.lookup(params.get("url") ?? "");
    $("title").textContent = page.company_name || page.title;
    chat.setPage(page);
    api.theme(page.page_id).then((t) => applyChatTheme(document.body, t.theme_color), () => {});
  } catch {
    $("message-input").placeholder = "This page hasn't been set up for the assistant yet.";
  }
})();
