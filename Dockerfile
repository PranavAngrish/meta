FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    gradio>=4.0.0 \
    requests>=2.31.0 \
    openai>=1.0.0

# Copy application code
COPY models.py ./
COPY client.py ./
COPY scenarios/ ./scenarios/
COPY server/ ./server/

# Expose port
EXPOSE 7860

# Environment variables
ENV PORT=7860
ENV OPENENV_ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Run the server
CMD ["python", "server/app.py"]
