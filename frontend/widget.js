/*
 * Embeddable launcher. Add to any page that has been ingested:
 *
 *   <script src="http://localhost:5173/widget.js" defer></script>
 *
 * Optional attributes: data-page="<page_id>" to pin a page, data-position="left".
 * The chat runs inside an iframe on the frontend origin, so the host page's
 * CSS/JS cannot read the conversation and vice versa.
 */
(function () {
  if (window.__journeyAgentLoaded) return;
  window.__journeyAgentLoaded = true;

  var script = document.currentScript;
  var origin = new URL(script.src).origin;
  var pinned = script.getAttribute("data-page");
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  var query = pinned ? "page=" + encodeURIComponent(pinned)
    : "url=" + encodeURIComponent(location.origin + location.pathname + location.search);

  // Static markup (no page data), so innerHTML is safe here.
  var BOT_ICON = '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" ' +
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 3.5V6"/><circle cx="12" cy="2.6" r="1.1" fill="currentColor" stroke="none"/>' +
    '<rect x="4" y="6" width="16" height="12.5" rx="4"/>' +
    '<circle cx="9.2" cy="11.8" r="1.4" fill="currentColor" stroke="none"/>' +
    '<circle cx="14.8" cy="11.8" r="1.4" fill="currentColor" stroke="none"/>' +
    '<path d="M9.8 15.3h4.4M2 11v3.2M22 11v3.2"/></svg>';
  var CLOSE_ICON = '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" ' +
    'stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

  var button = document.createElement("button");
  button.type = "button";
  button.setAttribute("aria-label", "Open assistant");
  button.setAttribute("aria-expanded", "false");
  button.innerHTML = BOT_ICON;
  button.style.cssText = "position:fixed;bottom:24px;" + side + ":24px;z-index:2147483646;" +
    "display:grid;place-items:center;width:60px;height:60px;padding:0;border:0;border-radius:50%;" +
    "cursor:pointer;color:#fff;background:#1e5eff;box-shadow:0 12px 30px rgba(30,94,255,.38)";

  var frame = document.createElement("iframe");
  frame.title = "Assistant";
  frame.src = origin + "/embed.html?" + query;
  frame.style.cssText = "position:fixed;bottom:98px;" + side + ":24px;z-index:2147483647;" +
    "width:380px;max-width:calc(100vw - 48px);height:560px;max-height:calc(100vh - 120px);" +
    "border:1px solid #e3e9f5;border-radius:22px;background:#fff;display:none;" +
    "box-shadow:0 22px 50px rgba(11,26,58,.18)";

  button.addEventListener("click", function () {
    var open = frame.style.display === "none";
    frame.style.display = open ? "block" : "none";
    button.innerHTML = open ? CLOSE_ICON : BOT_ICON;
    button.setAttribute("aria-label", open ? "Close assistant" : "Open assistant");
    button.setAttribute("aria-expanded", String(open));
  });

  document.body.appendChild(frame);
  document.body.appendChild(button);
})();
