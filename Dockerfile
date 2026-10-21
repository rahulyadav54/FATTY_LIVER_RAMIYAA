FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies (OpenGL and GLib for image/computer vision processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for fast build and small footprint (~700MB vs 4GB GPU image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Setup non-root user (compatible with Hugging Face Spaces & standard security policies)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000 10000 7860

# Run with Gunicorn production WSGI server (1 worker + 2 threads with gthread worker class optimized for Render free 512MB RAM)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 180 --worker-class gthread app:app"]

