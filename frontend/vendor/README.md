# Third-party files

Vendored (not loaded from a CDN) so the console has no runtime third-party script dependency.

| File | Source | License | Integrity |
|---|---|---|---|
| `lottie_light-5.13.0.min.js` | npm `lottie-web@5.13.0`, `build/player/lottie_light.min.js` | MIT (`lottie-web.LICENSE.md`) | tarball `sha512-+gfBXl6sxXMPe8tKQm7qzLnUy5DUPJPKIyRHwtpCpyUEYjHYRJC/5gjUvdkuO2c3JllrPtHXH5UJJK8LRYl5yQ==` |
| `../assets/chatbot.json` | ["Live chatbot"](https://lottiefiles.com/free-animation/live-chatbot-umPtBW5amb) on LottieFiles, extracted from the `.lottie` archive | [Lottie Simple License](https://lottiefiles.com/page/license) | vector only: no images, no expressions |

The *light* player is used on purpose: it renders SVG only and does not support expressions,
so it never `eval`s anything from an animation file.
