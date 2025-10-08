FROM python:3.11-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/installed_models /app/static /app/templates

# Expose ports
EXPOSE 5000 5001

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_NVML_BASED_CUDA_CHECK=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV BNB_CUDA_VERSION=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/local-whisper/health || exit 1

# Run the application
CMD ["python", "app.py"]