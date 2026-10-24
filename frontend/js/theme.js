// Page theming for the assistant: accent colours derived from the page's brand colour
// (GET /api/pages/{id}/theme) and a Lottie mascot recoloured to match it.

const VARS = ["--bot-accent", "--bot-accent-deep", "--bot-on-accent", "--bot-glow"];

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

/** Set the assistant's accent CSS variables on `el`; `null` restores the console defaults. */
export function applyChatTheme(el, hex) {
  if (!hex) {
    VARS.forEach((v) => el.style.removeProperty(v));
    return;
  }
  const rgb = hexToRgb(hex);
  const [h, s, l] = rgbToHsl(...rgb.map((c) => c / 255));
  const deep = hslToRgb(h, s, Math.max(0, l - 0.12)).map((c) => Math.round(c * 255));
  el.style.setProperty("--bot-accent", hex);
  el.style.setProperty("--bot-accent-deep", `rgb(${deep.join(", ")})`);
  // WCAG AA for the header title and send button text.
  el.style.setProperty("--bot-on-accent", contrast(rgb, [255, 255, 255]) >= 4.5 ? "#ffffff" : "#0b1a3a");
  el.style.setProperty("--bot-glow", `rgba(${rgb.join(", ")}, .38)`);
}

/**
 * Recolour a Lottie animation in place (pass a fresh copy): every saturated colour moves to the theme's hue.
 * Lightness is kept so the shading and highlights still read; whites, greys and the
 * near-black outline keep their tone.
 */
export function tintAnimation(data, hex) {
  const [th, ts] = rgbToHsl(...hexToRgb(hex).map((c) => c / 255));
  const tint = (r, g, b) => {
    const [, s, l] = rgbToHsl(r, g, b);
    if (s < 0.2 || l < 0.15 || l > 0.95) return [r, g, b];
    return hslToRgb(th, Math.min(1, 0.8 * ts + 0.2 * s), l);
  };
  const tintRgba = (arr) => [...tint(arr[0], arr[1], arr[2]), ...arr.slice(3)];
  const tintStops = (arr, stops) => {
    const out = [...arr];
    for (let i = 0; i < stops; i++) {
      const j = i * 4 + 1; // [offset, r, g, b] per colour stop; opacity stops follow
      [out[j], out[j + 1], out[j + 2]] = tint(out[j], out[j + 1], out[j + 2]);
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
  const walk = (o) => {
    if (Array.isArray(o)) { o.forEach(walk); return; }
    if (!o || typeof o !== "object") return;
    if ((o.ty === "fl" || o.ty === "st") && o.c) eachValue(o.c, tintRgba);
    if ((o.ty === "gf" || o.ty === "gs") && o.g) eachValue(o.g.k, (arr) => tintStops(arr, o.g.p));
    Object.values(o).forEach(walk);
  };
  walk(data);
  return data;
}
