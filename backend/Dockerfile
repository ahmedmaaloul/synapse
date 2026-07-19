# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
#
# BUILD CONTEXT IS THE REPO ROOT, not ./backend — the image must ship LICENSE
# and NOTICE (NOTICE requires the notices be retained in container images), and
# those live at the repo root. Build with:
#   docker build -f backend/Dockerfile -t synapse-backend .
FROM python:3.12-slim

WORKDIR /app

# System deps: gcc (builds), libgomp1 (OpenMP runtime required by
# onnxruntime/fastembed — otherwise embeddings crash at runtime on slim images).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (separate layer so code edits don't re-run pip)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./

# Attribution — required by NOTICE for every distribution of this image.
COPY LICENSE NOTICE /app/

EXPOSE 8000

# Production command (no --reload). docker-compose overrides this with --reload
# for local hot-reloading against the mounted source.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
