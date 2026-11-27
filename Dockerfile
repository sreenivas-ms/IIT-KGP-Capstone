# VeriSheet application container (Streamlit UI + agent backend).
# No credentials or certificates are baked in: pass .env at run time and mount config/ read-only.
#
#   docker build -t verisheet .
#   docker run --rm -p 8501:8501 --env-file .env -v "$PWD/config:/app/config:ro" verisheet
#
# The app runs as a non-root user. The Executor runs only allowlisted, typed pandas operations
# (no model-written code); uploads, artifacts and traces are written under /app/artifacts and
# /app/results (mount volumes there to keep them).
# 3.12-slim: newer patched base (3.11-slim was flagged with high-severity CVEs). Rebuild with --pull.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
RUN pip install -e . \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p artifacts results/traces \
    && chown -R appuser:appuser /app/artifacts /app/results \
    && chmod +x docker/entrypoint.sh

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
