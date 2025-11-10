FROM selenium/standalone-chrome:latest

WORKDIR /app

USER root

# Ensure seluser owns their venv completely
RUN chown -R seluser:seluser /home/seluser/venv && \
    chmod -R u+w /home/seluser/venv

COPY pyproject.toml .
RUN uv sync

COPY src src
COPY config/state.yaml config
#COPY bin bin

CMD ["python", "-c", "print('foo')"]
