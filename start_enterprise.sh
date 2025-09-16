#!/bin/bash

# Enterprise Jitsi CC Server Startup
cd /opt/praktik/whisper

echo "🏢 Starting Enterprise Jitsi CC Server"
echo "✨ Features: Multi-user, Translation, Smooth UI"
echo ""

# Kill existing servers
pkill -f "whisper_server.py" 2>/dev/null || true

echo "🔄 Entering nix-shell with FastAPI..."
nix-shell --run "python3 enterprise_whisper_server.py"