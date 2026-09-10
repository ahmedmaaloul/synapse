# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
#
# BUILD CONTEXT IS THE REPO ROOT, like the backend and frontend images — the
# image must ship LICENSE and NOTICE (NOTICE requires the notices be retained
# in container images), and those live at the repo root. Build with:
#   docker build -f packages/synapse-graphrag/Dockerfile -t synapse-mcp .
#
# Runs the MCP server over streamable HTTP on :8765 (/mcp). Point it at the
# backend with SYNAPSE_URL (docker-compose sets http://backend:8000).
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY packages/synapse-graphrag/ ./
# In-tree build leaves build/ and src/*.egg-info behind; the installed package
# lives in site-packages, so nothing under /app is imported at runtime.
RUN pip install --no-cache-dir . && rm -rf build src/synapse_graphrag.egg-info

# Attribution — required by NOTICE for every distribution of this image.
COPY LICENSE NOTICE /app/

RUN useradd --system --uid 10001 --create-home synapse
USER synapse

EXPOSE 8765

CMD ["synapse-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8765"]
