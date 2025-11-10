FROM selenium/standalone-chrome:latest

WORKDIR /app

USER root

# Ensure seluser owns their venv completely
RUN chown -R seluser:seluser /home/seluser/venv && \
    chmod -R u+w /home/seluser/venv

# Install cron
RUN apt-get update && \
    apt-get install -y cron && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/
COPY pyproject.toml .
# specific version of python installed with selenium img
RUN /home/seluser/venv/bin/python -m pip install .

COPY src/ src/

COPY docker/container_cron /etc/cron.d/container_cron
RUN chmod 0644 /etc/cron.d/container_cron && \
    echo "" >> /etc/cron.d/container_cron && \
    touch /etc/crontab

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
