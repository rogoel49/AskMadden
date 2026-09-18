"""Local launcher for the one Ask Madden app: API + frontend on one origin.

    python -m web.dev_server            # http://127.0.0.1:8000/ on this machine,
                                        # http://<this machine's LAN IP>:8000/ from a phone on the same WiFi

As of Phase 5.6 there is nothing to compose here: src/api/main.py itself
serves design/ as a static mount at /ui, redirects / there, and starts the
Phase 5.7 background refresh from its own lifespan. This module only
survives so the documented one-liner keeps working and so a phone on the
LAN can reach it -- it binds 0.0.0.0, which `uvicorn src.api.main:app`
does not by default. (0.0.0.0 is still local-network-only: nothing is
exposed to the internet unless the router forwards the port.) The public
HTTPS deployment is the Dockerfile + fly.toml at the repo root, which runs
the same `src.api.main:app`.

History, for anyone reading old TODO.md entries: Phase 5.3 was scoped not
to touch src/api/, so this file used to wrap the untouched API app in an
outer FastAPI that mounted design/ and re-triggered the API's lifespan by
hand (Starlette does not propagate a mounted app's lifespan). Phase 5.7
then hung the refresh thread off that outer lifespan. Both moved into
src/api/main.py in 5.6; the env vars are unchanged
(ASKMADDEN_REFRESH_ENABLED=0 turns the in-process refresh off).
"""
from __future__ import annotations

from src.api.main import app  # noqa: F401  -- `uvicorn web.dev_server:app` still works

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
