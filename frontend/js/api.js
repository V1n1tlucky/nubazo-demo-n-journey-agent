// Thin HTTP client for the Journey Agent API (no dependencies).

export class ApiError extends Error {
  constructor(status, code, detail) {
    super(detail);
    this.status = status;
    this.code = code;
  }
}

async function parse(resp) {
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    throw new ApiError(resp.status, body.code ?? "error", detail || resp.statusText);
  }
  return body;
}

export function createApi(base = window.JA_CONFIG?.apiBase ?? "") {
  const get = (path) => fetch(base + path).then(parse);
  const post = (path, data) =>
    fetch(base + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(parse);

  return {
    health: () => get("/api/health"),
    config: () => get("/api/config"),
    pages: () => get("/api/pages"),
    page: (id) => get(`/api/pages/${id}`),
    lookup: (url) => get(`/api/pages/lookup?url=${encodeURIComponent(url)}`),
    ingest: (url) => post("/api/pages", { url }),
    upload(file, url) {
      const form = new FormData();
      form.append("file", file);
      if (url) form.append("url", url);
      return fetch(base + "/api/pages/upload", { method: "POST", body: form }).then(parse);
    },
    runs: (pageId) => get(`/api/runs?limit=100${pageId ? `&page_id=${pageId}` : ""}`),
    run: (id) => get(`/api/runs/${id}`),
    snapshotUrl: (id) => `${base}/api/pages/${id}/snapshot`,
    theme: (id) => get(`/api/pages/${id}/theme`),
    welcome: (id) => get(`/api/pages/${id}/welcome`),

    /** POST /api/chat and dispatch Server-Sent Events to `onEvent(name, data)`. */
    async chat(payload, onEvent) {
      const resp = await fetch(base + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) await parse(resp);
      const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value.replace(/\r\n/g, "\n");
        let sep;
        while ((sep = buffer.indexOf("\n\n")) >= 0) {
          const raw = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          let event = "message";
          const data = [];
          for (const line of raw.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
          }
          if (data.length) onEvent(event, JSON.parse(data.join("\n")));
        }
      }
    },
  };
}
