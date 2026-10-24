// Page theming for the assistant: the page's brand palette (GET /api/pages/{id}/theme,
// primary first) drives the chat's accent colours and recolours the Lottie mascot.

const VARS = ["--bot-accent", "--bot-accent-deep", "--bot-on-accent", "--bot-glow", "--bot-accent-2", "--bot-accent-3"];

/** Primary / secondary / accent from a palette of 0–3 colours; missing roles reuse the last. */
export function paletteRoles(palette) {
  const p = (palette ?? []).filter(Boolean);
  return p.length ? { primary: p[0], secondary: p[1] ?? p[0], accent: p[2] ?? p[1] ?? p[0] } : null;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// h, s, l and r, g, b all in 0..1
function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
}

function hslToRgb(h, s, l) {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const ch = (t) => {
    const u = (t + 1) % 1;
    if (u < 1 / 6) return p + (q - p) * 6 * u;
    if (u < 1 / 2) return q;
    if (u < 2 / 3) return p + (q - p) * (2 / 3 - u) * 6;
    return p;
  };
  return [ch(h + 1 / 3), ch(h), ch(h - 1 / 3)];
}

function luminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const v = c / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

/** Set the assistant's palette CSS variables on `el`; an empty palette restores the defaults. */
export function applyChatTheme(el, palette) {
  const roles = paletteRoles(palette);
  if (!roles) {
    VARS.forEach((v) => el.style.removeProperty(v));
    return;
  }
  const hex = roles.primary;
  el.style.setProperty("--bot-accent-2", roles.secondary);
  el.style.setProperty("--bot-accent-3", roles.accent);
  const rgb = hexToRgb(hex);
  const [h, s, l] = rgbToHsl(...rgb.map((c) => c / 255));
  const deep = hslToRgb(h, s, Math.max(0, l - 0.12)).map((c) => Math.round(c * 255));
  el.style.setProperty("--bot-accent", hex);
  el.style.setProperty("--bot-accent-deep", `rgb(${deep.join(", ")})`);
  // WCAG AA for the header title and send button text.
  el.style.setProperty("--bot-on-accent", contrast(rgb, [255, 255, 255]) >= 4.5 ? "#ffffff" : "#0b1a3a");
  el.style.setProperty("--bot-glow", `rgba(${rgb.join(", ")}, .38)`);
}

// Which palette role each part of assets/chatbot.json takes, by its layer name.
// The robot (head, body, hands, tail, ears, face screen) is primary; the chat bubbles
// and their dots are secondary; the glowing cyan eyes, mic and highlights are accent.
const SECONDARY_LAYERS = /^(chat|rectangle)\b/i;
const isGlow = (h, l) => h >= 170 / 360 && h <= 190 / 360 && l > 0.65;

/**
 * Recolour a Lottie animation in place (pass a fresh copy) with the page's palette.
 * Each part takes its role's hue and saturation but keeps its own lightness, so the
 * gradients and highlights still read; whites, greys and the dark outline stay as drawn.
 */
export function tintAnimation(data, palette) {
  const roles = paletteRoles(palette);
  if (!roles) return data;
  const hsl = Object.fromEntries(Object.entries(roles).map(([k, hex]) => [k, rgbToHsl(...hexToRgb(hex).map((c) => c / 255))]));
  const tint = (r, g, b, role) => {
    const [h, s, l] = rgbToHsl(r, g, b);
    if (s < 0.2 || l < 0.15 || l > 0.95) return [r, g, b];
    const [th, ts] = hsl[role === "primary" && isGlow(h, l) ? "accent" : role];
    return hslToRgb(th, Math.min(1, 0.8 * ts + 0.2 * s), l);
  };
  const tintRgba = (role) => (arr) => [...tint(arr[0], arr[1], arr[2], role), ...arr.slice(3)];
  const tintStops = (stops, role) => (arr) => {
    const out = [...arr];
    for (let i = 0; i < stops; i++) {
      const j = i * 4 + 1; // [offset, r, g, b] per colour stop; opacity stops follow
      [out[j], out[j + 1], out[j + 2]] = tint(out[j], out[j + 1], out[j + 2], role);
    }
    return out;
  };
  // A property is either static ({a: 0, k: value}) or keyframed ({a: 1, k: [{s, e}, …]}).
  const eachValue = (prop, fn) => {
    if (prop?.a === 1 && Array.isArray(prop.k)) {
      for (const kf of prop.k) {
        if (kf.s) kf.s = fn(kf.s);
        if (kf.e) kf.e = fn(kf.e);
      }
    } else if (Array.isArray(prop?.k)) {
      prop.k = fn(prop.k);
    }
  };
  const walk = (o, role) => {
    if (Array.isArray(o)) { o.forEach((v) => walk(v, role)); return; }
    if (!o || typeof o !== "object") return;
    if (typeof o.ty === "number") role = SECONDARY_LAYERS.test(o.nm ?? "") ? "secondary" : "primary"; // a layer
    if ((o.ty === "fl" || o.ty === "st") && o.c) eachValue(o.c, tintRgba(role));
    if ((o.ty === "gf" || o.ty === "gs") && o.g) eachValue(o.g.k, tintStops(o.g.p, role));
    Object.values(o).forEach((v) => walk(v, role));
  };
  walk(data, "primary");
  return data;
}
