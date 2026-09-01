# Multi-stage production container for PramanAI FastAPI Backend on Google Cloud Run
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    UV_SYSTEM_PYTHON=1

# Install required Linux system dependencies (OpenCV, Poppler PDF rendering, Tesseract OCR, PostgreSQL dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-hin \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml uv.lock* ./

# Install project dependencies
RUN uv sync --no-dev

# Copy application source code
COPY . .

# Expose container port (Cloud Run sets $PORT dynamically)
EXPOSE 8000

# Run FastAPI server using Uvicorn with dynamic Cloud Run PORT binding
CMD ["sh", "-c", "uv run uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
