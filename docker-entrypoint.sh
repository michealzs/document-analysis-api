#!/usr/bin/env bash
# Ensure the data dir exists and is writable, then exec the server.
set -euo pipefail

DATA_DIR="${DOCANALYSIS_DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR"/uploads "$DATA_DIR"/results "$DATA_DIR"/logs

echo "[entrypoint] data dir: $DATA_DIR"
echo "[entrypoint] starting: $*"
exec "$@"
