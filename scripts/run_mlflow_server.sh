#!/bin/bash
# Run a local MLflow tracking server backed by sqlite.
#
# Matches the `mlflow_tracking_uri = "http://localhost:5000"` convention
# documented in configs/README.md, so any config with that field set will
# log to this server.
#
# Usage: ./scripts/run_mlflow_server.sh [--port PORT] [--host HOST]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

HOST="127.0.0.1"
PORT="5000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

BACKEND_STORE_URI="sqlite:///$PROJECT_ROOT/mlflow.db"
ARTIFACT_ROOT="$PROJECT_ROOT/mlruns"

mkdir -p "$ARTIFACT_ROOT"

echo "Backend store:   $BACKEND_STORE_URI"
echo "Artifact root:   $ARTIFACT_ROOT"
echo "Serving at:      http://$HOST:$PORT"

exec mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host "$HOST" \
    --port "$PORT"
