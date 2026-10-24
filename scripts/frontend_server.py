"""Static dev server for frontend/ that disables browser caching.

`python -m http.server` lets the browser reuse a cached app.css/app.js after the
HTML changes, which renders a half-old, broken-looking console. `no-store`
makes every reload pick up the latest files.
"""

import functools
import http.server
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    handler = functools.partial(NoCacheHandler, directory=str(FRONTEND))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving frontend/ on http://127.0.0.1:{port} (no-cache)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
