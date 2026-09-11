# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
#
# BUILD CONTEXT IS THE REPO ROOT, like the backend and frontend images, so
# compose, CI and the release workflow build all three the same way. The image
# ships the client package's OWN LICENSE and NOTICE (Apache-2.0) — not the
# backend's PolyForm Noncommercial pair at the repo root. Build with:
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

# Attribution — the client's Apache-2.0 LICENSE and NOTICE travel with every
# distribution of this image (CI checks `ls /app/LICENSE /app/NOTICE`).
COPY packages/synapse-graphrag/LICENSE packages/synapse-graphrag/NOTICE /app/

RUN useradd --system --uid 10001 --create-home synapse
USER synapse

EXPOSE 8765

CMD ["synapse-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8765"]
