FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common tzdata \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
COPY . ./
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]