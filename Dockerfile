# python:3.13-slim is multi-arch, so this image builds on both Intel and ARM
# Synology models without changes.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a code change does not re-run the dependency install.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY foolwatch/ ./foolwatch/
COPY config.toml ./
# Chart surfaces are validated against this theme's background, so ship it.
COPY .streamlit/ ./.streamlit/

# --uid 1000 is load-bearing: the bind-mounted host data dir must be chown'd
# to 1000 to match, or every SQLite write fails silently and the scraper
# quietly stops persisting anything.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

VOLUME ["/app/data"]

EXPOSE 8501

# Per-service healthchecks live in docker-compose.yml: the dashboard serves
# HTTP and the worker does not, so a single image-level check cannot describe
# both.

# Overridden per service in docker-compose.yml.
CMD ["python", "-m", "foolwatch.scheduler"]
