# Build context: repo root (see infra/docker-compose.yml)
FROM apache/airflow:3.3.2-python3.12
COPY --chown=airflow:root packages/core /opt/alphalens/packages/core
RUN pip install --no-cache-dir /opt/alphalens/packages/core
