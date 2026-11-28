# Build context: repo root (see infra/docker-compose.yml)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /repo
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock* ./
COPY packages/core packages/core
COPY apps/api apps/api
RUN uv sync --package alphalens-api --no-dev
ENV PATH="/repo/.venv/bin:$PATH"
WORKDIR /repo/apps/api
EXPOSE 8000
# Hosts like Railway pass the port to listen on in $PORT; default to 8000 locally.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
