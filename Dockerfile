FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-hackathon"
LABEL org.opencontainers.image.description="Customer Support Triage — OpenEnv Environment"

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY customer_support_env/ ./customer_support_env/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# HF Spaces expects port 7860
EXPOSE 7860

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
