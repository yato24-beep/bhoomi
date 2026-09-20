FROM python:3.11-slim

# Prevent bytecode compilation and ensure unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    FLAGS_use_mkldnn=0 \
    DATABASE_URL="sqlite:////data/land_records.db"

# Install essential system dependencies for OCR, PDF rendering, and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kan \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (UID 1000 required for Hugging Face Docker Spaces)
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/app /data/models/trocr/checkpoint-12000 /data/storage && \
    chown -R user:user /home/user /data

USER user
WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source tree and fixtures
COPY --chown=user:user . .

# Expose default Hugging Face Space port
EXPOSE 7860

# Launch FastAPI application using port from environment variable, defaulting to 7860
CMD ["sh", "-c", "exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
