FROM python:3.12-slim

WORKDIR /app

# System deps: gcc (builds), libgomp1 (OpenMP runtime required by
# onnxruntime/fastembed — otherwise embeddings crash at runtime on slim images).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Production command (no --reload). docker-compose overrides this with --reload
# for local hot-reloading against the mounted source.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
