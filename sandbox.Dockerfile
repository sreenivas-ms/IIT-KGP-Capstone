# Separate image for RestrictedSandbox; do not use the application container as the sandbox.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "pandas>=2.2,<3" "numpy>=1.26,<3" "matplotlib>=3.8,<4" "openpyxl>=3.1,<4" \
    && useradd --uid 65534 --no-create-home --shell /usr/sbin/nologin sandbox
WORKDIR /work
USER 65534:65534
