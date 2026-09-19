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

# The SQLite database and log live here; mount a volume over it so they
# survive a rebuild.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8501

# Overridden per service in docker-compose.yml.
CMD ["python", "-m", "foolwatch.scheduler"]
