#!/bin/bash

# Simple Whisper App Startup Script for Podman
# This script starts the Whisper app using Podman with the existing venv

echo "🎙️ Starting Whisper App with Podman (Simple Mode)..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the whisper directory."
    exit 1
fi

# Check if Podman is available
if ! command -v podman &> /dev/null; then
    echo "❌ Error: Podman is not installed or not in PATH"
    exit 1
fi

echo "✅ Podman found"
echo "📁 Working directory: $(pwd)"
echo ""

# Function to stop existing container
stop_existing_container() {
    if podman ps -q --filter "name=local-whisper" | grep -q .; then
        echo "🛑 Stopping existing local-whisper container..."
        podman stop local-whisper
        podman rm local-whisper
    fi
}

# Function to start container with existing venv
start_container_with_venv() {
    echo "🚀 Starting Whisper container with existing virtual environment..."
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        echo "❌ Error: Virtual environment not found. Please run with nix-shell first to create venv."
        exit 1
    fi
    
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
        -v ./venv:/app/venv \
        -v ./app.py:/app/app.py \
        -v ./sockpy.py:/app/sockpy.py \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTHONPATH=/app \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e PYTORCH_NVML_BASED_CUDA_CHECK=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
        -e CUDA_LAUNCH_BLOCKING=0 \
        -e BNB_CUDA_VERSION="" \
        python:3.11-slim \
        bash -c "apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && cd /app && source venv/bin/activate && python app.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to start container"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "✅ Whisper service started successfully!"
    echo ""
    echo "🌐 Local endpoints:"
    echo "   - Flask app: http://localhost:5000"
    echo "   - WebSocket: ws://localhost:5001"
    echo "   - Health check: http://localhost:5000/local-whisper/health"
    echo "   - Live CC info: http://localhost:5000/live-cc"
    echo ""
    echo "🌐 Nginx endpoints:"
    echo "   - HTTPS: https://whisper.semaphor.dk"
    echo "   - Health: https://whisper.semaphor.dk/local-whisper/health"
    echo "   - Live CC: https://whisper.semaphor.dk/live-cc"
    echo "   - WebSocket: wss://whisper.semaphor.dk/ws"
    echo ""
    echo "📊 Container status:"
    podman ps --filter "name=local-whisper" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "📋 Useful commands:"
    echo "   - View logs: podman logs -f local-whisper"
    echo "   - Stop service: podman stop local-whisper"
    echo "   - Restart service: podman restart local-whisper"
    echo "   - Enter container: podman exec -it local-whisper bash"
    echo ""
}

# Main execution
stop_existing_container
start_container_with_venv
show_status

# Follow logs if requested
if [ "$1" = "--logs" ]; then
    echo "📋 Following container logs (Ctrl+C to exit)..."
    podman logs -f local-whisper
fi