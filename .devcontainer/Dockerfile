FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Optional: install basics
RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally for the vscode user and CI-style workflows.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx
