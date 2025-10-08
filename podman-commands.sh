#!/bin/bash

# Podman Commands for Whisper App
# This file contains useful podman commands for managing the Whisper container

echo "🎙️ Whisper App - Podman Commands"
echo "================================="
echo ""

# Function to show available commands
show_commands() {
    echo "Available commands:"
    echo "  build     - Build the container image"
    echo "  start     - Start the container"
    echo "  stop      - Stop the container"
    echo "  restart   - Restart the container"
    echo "  logs      - Show container logs"
    echo "  logs-f    - Follow container logs"
    echo "  status    - Show container status"
    echo "  shell     - Enter container shell"
    echo "  clean     - Remove container and image"
    echo "  health    - Check container health"
    echo ""
}

# Function to build container
build_container() {
    echo "🔨 Building Whisper container..."
    podman build -t local-whisper:latest .
}

# Function to start container
start_container() {
    echo "🚀 Starting Whisper container..."
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
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e PYTORCH_NVML_BASED_CUDA_CHECK=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
        -e CUDA_LAUNCH_BLOCKING=0 \
        -e BNB_CUDA_VERSION="" \
        local-whisper:latest
}

# Function to stop container
stop_container() {
    echo "🛑 Stopping Whisper container..."
    podman stop local-whisper
}

# Function to restart container
restart_container() {
    echo "🔄 Restarting Whisper container..."
    podman restart local-whisper
}

# Function to show logs
show_logs() {
    echo "📋 Container logs:"
    podman logs local-whisper
}

# Function to follow logs
follow_logs() {
    echo "📋 Following container logs (Ctrl+C to exit)..."
    podman logs -f local-whisper
}

# Function to show status
show_status() {
    echo "📊 Container status:"
    podman ps --filter "name=local-whisper" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to enter shell
enter_shell() {
    echo "🐚 Entering container shell..."
    podman exec -it local-whisper bash
}

# Function to clean up
clean_up() {
    echo "🧹 Cleaning up container and image..."
    podman stop local-whisper 2>/dev/null
    podman rm local-whisper 2>/dev/null
    podman rmi local-whisper:latest 2>/dev/null
    echo "✅ Cleanup complete"
}

# Function to check health
check_health() {
    echo "🏥 Checking container health..."
    podman exec local-whisper curl -f http://localhost:5000/local-whisper/health
}

# Main command handling
case "$1" in
    "build")
        build_container
        ;;
    "start")
        start_container
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        restart_container
        ;;
    "logs")
        show_logs
        ;;
    "logs-f")
        follow_logs
        ;;
    "status")
        show_status
        ;;
    "shell")
        enter_shell
        ;;
    "clean")
        clean_up
        ;;
    "health")
        check_health
        ;;
    *)
        show_commands
        echo "Usage: $0 <command>"
        echo "Example: $0 start"
        ;;
esac