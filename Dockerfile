# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Ensure git and curl are available
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv explicitly to ensure we have the latest version for pip install
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy the entire project to /app/env
COPY . /app/env
WORKDIR /app/env

# Install dependencies globally into the container (--system) to avoid venv migration issues.
# We remove uv.lock because it may contain platform-specific pins that conflict with Linux.
# "openenv-core" and other dependencies from pyproject.toml are installed here.
RUN rm -f uv.lock && \
    uv pip install --system --no-cache .

# Set PYTHONPATH so that 'import AutoMathReasoner' works correctly.
# The project layout has the AutoMathReasoner namespace mapped to the root directory.
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Enable Web Interface for OpenEnv
ENV ENABLE_WEB_INTERFACE=true

# Hugging Face Spaces dynamically assigns a port, but usually expects 7860
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the FastAPI server.
# We use the module string to ensure the setup-tools mapped namespace is used.
CMD ["uvicorn", "AutoMathReasoner.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
