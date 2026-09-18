#!/bin/sh
# Container entrypoint for Ask Madden (see Dockerfile). Three jobs, then
# exec uvicorn so it is PID 1 and receives the host's stop signal directly.
#
# Env vars the server reads (set them on the host; there is no .env here):
#   ANTHROPIC_API_KEY                 required -- chat (recommend()) is Claude-backed
#   ASKMADDEN_MODEL                   optional -- override recommend.py's default model
#   ASKMADDEN_DAILY_QUERY_CAP         optional -- Claude-backed chats per user per UTC day (default 25)
#   ASKMADDEN_REFRESH_ENABLED         optional -- 0 turns the in-process refresh off (then run
#                                                 `python -m src.scheduler.refresh --once` on a schedule)
#   ASKMADDEN_REFRESH_INTERVAL_SECONDS optional -- default 21600 (6h; nflverse rebuilds at most twice a day)
#   PORT                              optional -- injected by Railway/Render; Fly uses 8080 (fly.toml)
# SLEEPER_LEAGUE_ID / MY_ROSTER_ID are CLI-only conveniences; the server
# never needs them (every request carries its own league and roster).
set -eu
cd /app

DATA=/app/data
SIGNALS="$DATA/processed/signals"

# 1. Seed the (possibly empty, freshly-mounted) volume with the committed
#    reference signals tables. Copy only what is missing: a table the
#    refresh has since recomputed on the volume is newer than the image's.
mkdir -p "$SIGNALS" "$DATA/raw/leagues"
for f in /app/seed/signals/*.parquet; do
  [ -e "$f" ] || continue
  dest="$SIGNALS/$(basename "$f")"
  [ -e "$dest" ] || cp "$f" "$dest"
done

# 2. Keep chromadb's default embedding model (an ~80MB ONNX download on the
#    first embed, into ~/.cache/chroma) on the volume, so a redeploy does
#    not fetch it again.
mkdir -p "$DATA/.cache/chroma" "$HOME/.cache"
[ -L "$HOME/.cache/chroma" ] || { rm -rf "$HOME/.cache/chroma"; ln -s "$DATA/.cache/chroma" "$HOME/.cache/chroma"; }

# 3. Serve. src.api.main:app is the API, the /ui frontend and (unless
#    ASKMADDEN_REFRESH_ENABLED=0) the background refresh, all in one.
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8080}" --proxy-headers --forwarded-allow-ips='*'
