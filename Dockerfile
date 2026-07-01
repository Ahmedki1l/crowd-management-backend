# Multi-stage image used for BOTH the API and the pipeline workers (HLD 12.4).
# Run mode is chosen by the command (see docker-compose.yaml / HLD 12.5).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps: ffmpeg/OpenCV runtime, and the Microsoft ODBC driver for pyodbc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg apt-transport-https ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
FROM base AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
# Install the service with the production extras (SQL Server + inference + cache).
# Add the "reid" extra on GPU/Re-ID hosts: pip install .[mssql,inference,redis,reid]
RUN pip install --upgrade pip && pip install ".[mssql,inference,redis]"

# ---------------------------------------------------------------------------
FROM base AS runtime
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY . .

# Non-root runtime user.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8008

# Default = API + engine (small deployment). Override for workers:
#   command: ["python", "-m", "app.main", "--workers"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8008/health').status==200 else 1)" || exit 1

CMD ["python", "-m", "app.main", "--api", "--host", "0.0.0.0", "--port", "8008"]
