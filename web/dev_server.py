"""Phase 5.3 dev server: the Phase 5.2 API and the responsive frontend on
ONE origin, so the browser's fetch() calls need no CORS.

    python -m web.dev_server            # http://127.0.0.1:8000/ on this machine,
                                        # http://<this machine's LAN IP>:8000/ from a phone on the same WiFi

Why this exists (and why it's outside src/api/): src/api/main.py has no
CORS middleware and no static mount, so opening
design/askmadden-ui-mockup.html as a file:// page and pointing it at
http://localhost:8000 is blocked by the browser. Phase 5.3 was scoped
not to touch src/api/, so this wrapper composes the existing app rather
than editing it: it mounts design/ as static files under /ui and the
untouched API app under /, on one port. Phase 5.6 ("FastAPI mounts the
one responsive frontend as a static route") is where this folds into
src/api/main.py itself; until then this is the way to run the UI.

Starlette does not propagate a mounted app's lifespan, so the API's
once-per-process .env load is triggered here explicitly.

Binds to 0.0.0.0 (Phase 5.4), not 127.0.0.1: "Add to Home Screen" can only
be tested on a real phone, and a phone on the same WiFi can only reach a
server that listens on the machine's network interface. 0.0.0.0 is still
local-network-only -- nothing here is exposed to the internet unless the
router forwards the port, which it doesn't by default. Phase 5.6 is the
public, HTTPS deployment; this is not it.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.main import app as api_app
from src.reasoning import recommend

DESIGN_DIR = Path(__file__).resolve().parents[1] / "design"
UI_PATH = "/ui/askmadden-ui-mockup.html"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    recommend.load_dotenv_once()
    yield


def build_app() -> FastAPI:
    outer = FastAPI(title="Ask Madden (UI + API, one origin)", lifespan=_lifespan)

    @outer.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse(UI_PATH)

    outer.mount("/ui", StaticFiles(directory=str(DESIGN_DIR)), name="ui")
    outer.mount("/", api_app)  # its routes already carry the /api prefix
    return outer


app = build_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.dev_server:app", host="0.0.0.0", port=8000, reload=False)
