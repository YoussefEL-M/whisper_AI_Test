# Whisper App - Podman Setup

This document explains how to run the Whisper app using Podman instead of Nix shell.

## Prerequisites

- Podman installed and configured
- NVIDIA GPU with CUDA support (for GPU acceleration)
- At least 8GB of available RAM
- Sufficient disk space for models (~2-4GB)

## Quick Start

### Option 1: Simple Mode (Recommended - Uses Existing venv)

This approach uses your existing virtual environment from nix-shell, avoiding the long build process:

```bash
# First, ensure you have a venv (run nix-shell once to create it)
nix-shell --run "python3 -c 'import whisper; print(\"venv ready\")'"

# Then start with Podman using the existing venv
./start-whisper-podman-simple.sh

# Start with logs following
./start-whisper-podman-simple.sh --logs
```

### Option 2: Full Container Build

This approach builds a complete container (takes longer but is self-contained):

```bash
# Start the Whisper app with Podman
./start-whisper-podman.sh

# Start with logs following
./start-whisper-podman.sh --logs
```

### Option 2: Using Individual Commands

```bash
# Build the container
./podman-commands.sh build

# Start the container
./podman-commands.sh start

# Check status
./podman-commands.sh status

# View logs
./podman-commands.sh logs-f
```

### Option 3: Manual Podman Commands

```bash
# Build the image
podman build -t local-whisper:latest .

# Start the container
podman run -d \
    --name local-whisper \
    --restart unless-stopped \
    -p 5000:5000 \
    -p 5001:5001 \
    -v ./models:/app/models \
    -v ./installed_models:/app/installed_models \
    -v ./TinyLlama-1.1B-Chat-v1.0:/app/TinyLlama-1.1B-Chat-v1.0 \
    -v ./static:/app/static \
    -v ./templates:/app/templates \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH=/app \
    local-whisper:latest
```

## Available Endpoints

Once the container is running, the following endpoints will be available:

### Local Endpoints
- **Flask App**: http://localhost:5000
- **WebSocket**: ws://localhost:5001
- **Health Check**: http://localhost:5000/local-whisper/health
- **Live CC Info**: http://localhost:5000/live-cc

### Nginx Endpoints (if configured)
- **HTTPS**: https://whisper.semaphor.dk
- **Health**: https://whisper.semaphor.dk/local-whisper/health
- **Live CC**: https://whisper.semaphor.dk/live-cc
- **WebSocket**: wss://whisper.semaphor.dk/ws

## Management Commands

Use the `podman-commands.sh` script for easy container management:

```bash
# Show all available commands
./podman-commands.sh

# Common operations
./podman-commands.sh start      # Start container
./podman-commands.sh stop       # Stop container
./podman-commands.sh restart    # Restart container
./podman-commands.sh logs-f     # Follow logs
./podman-commands.sh status     # Show status
./podman-commands.sh shell      # Enter container shell
./podman-commands.sh health     # Check health
./podman-commands.sh clean      # Remove container and image
```

## Container Details

### Volumes
- `./models:/app/models` - Whisper model storage
- `./installed_models:/app/installed_models` - Installed model cache
- `./TinyLlama-1.1B-Chat-v1.0:/app/TinyLlama-1.1B-Chat-v1.0` - Llama model
- `./static:/app/static` - Static files
- `./templates:/app/templates` - HTML templates

### Environment Variables
- `CUDA_VISIBLE_DEVICES=0` - Use first GPU
- `PYTHONPATH=/app` - Python path
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` - CUDA device ordering
- `PYTORCH_NVML_BASED_CUDA_CHECK=1` - CUDA compatibility
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` - Memory optimization
- `CUDA_LAUNCH_BLOCKING=0` - Async CUDA operations
- `BNB_CUDA_VERSION=""` - Disable bitsandbytes CUDA version check

### Ports
- `5000` - Flask application
- `5001` - WebSocket server

## Troubleshooting

### Container Won't Start
```bash
# Check if port is already in use
netstat -tlnp | grep -E ":5000|:5001"

# Check container logs
podman logs local-whisper

# Check if image was built correctly
podman images | grep local-whisper
```

### GPU Issues
```bash
# Check if NVIDIA runtime is available
podman info | grep -i nvidia

# Check CUDA in container
podman exec local-whisper python3 -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Check container resource usage
podman stats local-whisper

# Check available GPU memory
podman exec local-whisper nvidia-smi
```

### Model Loading Issues
```bash
# Check if models directory exists
ls -la ./models/

# Check model files in container
podman exec local-whisper ls -la /app/models/
```

## Migration from Nix Shell

If you were previously using the Nix shell setup:

1. Stop any running Nix shell processes
2. Use the new podman scripts instead of `start-whisper.sh`
3. The same functionality is preserved, just running in a container

## Performance Notes

- The container includes memory optimizations for CUDA
- Models are loaded on-demand to manage GPU memory
- The container uses float16 instead of 8-bit quantization for better compatibility
- GPU memory is managed with expandable segments and split size limits

## Support

For issues or questions:
1. Check the container logs: `podman logs local-whisper`
2. Verify GPU availability: `podman exec local-whisper nvidia-smi`
3. Check health endpoint: `curl http://localhost:5000/local-whisper/health`