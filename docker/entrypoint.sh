#!/bin/sh
# Build the retrieval index on first start (needs the embedding API from .env); the app still
# starts if this fails: retrieve_context then returns a RETRIEVAL_UNAVAILABLE warning.
set -e
if [ ! -f artifacts/index/manifest.json ]; then
  python scripts/build_index.py || echo "WARNING: retrieval index not built (check EMBEDDING_* settings)"
fi
exec "$@"
