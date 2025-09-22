#!/bin/bash

# Whisper App Startup Script
# This script starts the Whisper app using Nix shell

echo "🎙️ Starting Whisper App with Nix shell..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "sockpy.py" ]; then
    echo "❌ Error: sockpy.py not found. Please run this script from the whisper directory."
    exit 1
fi

# Check if Nix is available
if ! command -v nix &> /dev/null; then
    echo "❌ Error: Nix is not installed or not in PATH"
    exit 1
fi

echo "✅ Nix found, entering shell..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo ""

# Start the Whisper app
echo "🚀 Starting Whisper transcription service..."
echo "   - Flask app: http://localhost:5000"
echo "   - WebSocket: ws://localhost:5001"
echo "   - Health check: http://localhost:5000/health"
echo "   - Live CC info: http://localhost:5000/live-cc"
echo ""
echo "🌐 Nginx endpoints:"
echo "   - HTTPS: https://whisper.semaphor.dk"
echo "   - Health: https://whisper.semaphor.dk/health"
echo "   - Live CC: https://whisper.semaphor.dk/live-cc"
echo "   - WebSocket: wss://whisper.semaphor.dk/ws"
echo ""

# Run with nix-shell
nix-shell --run "python3 sockpy.py"