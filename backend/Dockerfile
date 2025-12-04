# Octavia Video Translator - Docker Image


FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts test_samples temp

# Download test sample (optional)
RUN if [ ! -f "test_samples/sample_30s.mp4" ]; then \
    echo "Downloading test sample..."; \
    wget -q -O test_samples/sample_30s.mp4 https://example.com/sample.mp4 || true; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Expose API port
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "==========================================="\n\
echo "Octavia Video Translator - Technical Assessment"\n\
echo "==========================================="\n\
echo "Hardware detected:"\n\
echo "  CPUs: $(nproc)"\n\
echo "  RAM: $(free -h | awk '\''/^Mem:/{print $2}'\'')"\n\
echo "  Python: $(python --version)"\n\
echo "==========================================="\n\
\n\
# Run integration test first\n\
if [ "$1" = "test" ]; then\n\
    echo "Running integration test..."\n\
    python cli.py test-integration\n\
    exit $?\n\
fi\n\
\n\
# Otherwise run the API server\n\
echo "Starting API server on http://0.0.0.0:8000"\n\
echo "API Docs: http://0.0.0.0:8000/docs"\n\
echo "\n\
Available commands:\n\
  docker run octavia test     - Run integration test\n\
  docker run octavia cli ...  - Run CLI command\n\
"\n\
exec uvicorn app:app --host 0.0.0.0 --port 8000\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["api"]