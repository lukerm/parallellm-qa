FROM python:3.14-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

COPY pyproject.toml .
RUN uv sync

COPY src src
COPY config/state.yaml config
#COPY bin bin

CMD ["python", "-c", "print('foo')"]
