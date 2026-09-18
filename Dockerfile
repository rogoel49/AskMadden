# Ask Madden -- one container, one app: the FastAPI API, the responsive
# frontend it serves at /ui, and the Phase 5.7 background refresh in-process.
#
#   docker build -t askmadden .
#   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... -v askmadden_data:/app/data askmadden
#
# Host-agnostic: fly.toml at the repo root is the checked-in config for
# Fly.io, but Railway/Render run this same image given (1) a persistent
# volume at /app/data and (2) the env vars listed in deploy/entrypoint.sh.
# The process listens on $PORT (Railway/Render inject it) or 8080 (Fly's
# internal_port).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so a code-only change doesn't reinstall them.
# constraints.txt pins what the suite was last green against on Rohan's
# machine (see its header); requirements.txt stays the unpinned dev list.
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY . .

# The two reference signals tables the prior-season fallback (Phase 3.6)
# and the tests depend on are committed under data/processed/signals/. A
# persistent volume mounted at /app/data hides everything the image has
# there, so keep a copy outside the mount; deploy/entrypoint.sh seeds the
# volume from it on first boot (and never overwrites a newer table).
RUN mkdir -p /app/seed && cp -r /app/data/processed/signals /app/seed/signals \
    && chmod +x /app/deploy/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
