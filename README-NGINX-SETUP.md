# Whisper App Nginx Setup

This document explains how to run your Whisper app with Nginx configuration.

## 🎙️ Application Overview

Your Whisper app (`sockpy.py`) provides:

- **Flask HTTP Server** (Port 5000):
  - `/health` - Health check endpoint
  - `/live-cc` - Live captions information endpoint

- **WebSocket Server** (Port 5001):
  - Real-time audio transcription
  - Accepts PCM audio data
  - Returns JSON transcription results

## 🚀 Running the Application

### Method 1: Using the Startup Script
```bash
cd /opt/praktik/whisper
./start-whisper.sh
```

### Method 2: Manual Nix Shell
```bash
cd /opt/praktik/whisper
nix-shell
python3 sockpy.py
```

## 🌐 Nginx Configuration

The Whisper app endpoints have been integrated into your existing `/etc/nginx/sites-available/rosetta.semaphor.dk` configuration with the following endpoints:

### HTTPS Endpoints (via Nginx - Integrated with rosetta.semaphor.dk)
- **Health Check**: `https://rosetta.semaphor.dk/whisper-health`
- **Live CC Info**: `https://rosetta.semaphor.dk/whisper-live-cc`
- **WebSocket**: `wss://rosetta.semaphor.dk/whisper-ws`
- **Alternative WebSocket**: `wss://rosetta.semaphor.dk/whisper-live-cc-ws`

### Local Development Endpoints
- **HTTP**: `http://localhost:5000`
- **WebSocket**: `ws://localhost:5001`

## 🧪 Testing

Run the test script to verify all endpoints:
```bash
cd /opt/praktik/whisper
python3 test-endpoints.py
```

## 📋 WebSocket Protocol

### Connection
Connect to: `wss://whisper.semaphor.dk/ws`

### Initial Message
The server sends a welcome message:
```json
{
  "type": "status",
  "message": "Connected to transcription service"
}
```

### Audio Data
Send PCM audio data as binary:
- Format: 16-bit PCM
- Sample Rate: 16kHz
- Channels: Mono
- Chunk Size: 1 second (32,000 bytes)

### Transcription Response
```json
{
  "type": "transcription",
  "text": "Transcribed text here",
  "participant": "Speaker"
}
```

### Error Response
```json
{
  "type": "error",
  "message": "Error description"
}
```

## 🔧 Configuration Details

### SSL Certificates
- Uses wildcard certificate: `wildcard.semaphor.dk`
- Certificate path: `/etc/nginx/ssl/wildcard.semaphor.dk/`

### CORS Headers
- All endpoints allow cross-origin requests
- WebSocket connections support CORS

### File Upload Limits
- Maximum request size: 50MB (for audio files)

### Timeouts
- HTTP: 60 seconds
- WebSocket: 7 days (for long-running connections)

## 🐛 Troubleshooting

### Check if Nginx is running:
```bash
systemctl status nginx
```

### Check Nginx configuration:
```bash
nginx -t
```

### Reload Nginx after changes:
```bash
systemctl reload nginx
```

### Check Nginx logs:
```bash
tail -f /var/log/nginx/whisper-app-access.log
tail -f /var/log/nginx/whisper-app-error.log
```

### Check if Whisper app is running:
```bash
curl http://localhost:5000/health
```

### Test WebSocket connection:
```bash
wscat -c ws://localhost:5001
```

## 📁 File Structure

```
/opt/praktik/whisper/
├── sockpy.py              # Main application
├── shell.nix             # Nix environment
├── start-whisper.sh      # Startup script
├── test-endpoints.py     # Test script
└── README-NGINX-SETUP.md # This file

/etc/nginx/sites-available/
└── whisper-app           # Nginx configuration
```

## 🔄 Restarting Services

### Restart Whisper App:
```bash
# Kill existing process
pkill -f sockpy.py

# Start again
cd /opt/praktik/whisper
./start-whisper.sh
```

### Restart Nginx:
```bash
systemctl restart nginx
```

## 📊 Monitoring

### Health Check
```bash
curl https://rosetta.semaphor.dk/whisper-health
```

### Live CC Info
```bash
curl https://rosetta.semaphor.dk/whisper-live-cc
```

## 🎯 Usage Examples

### JavaScript WebSocket Client
```javascript
const ws = new WebSocket('wss://rosetta.semaphor.dk/whisper-ws');

ws.onopen = function() {
    console.log('Connected to Whisper service');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'transcription') {
        console.log('Transcription:', data.text);
    }
};

// Send audio data
ws.send(audioBuffer);
```

### Python WebSocket Client
```python
import asyncio
import websockets
import json

async def test_whisper():
    uri = "wss://rosetta.semaphor.dk/whisper-ws"
    async with websockets.connect(uri) as websocket:
        # Send audio data
        audio_data = b'\x00' * 32000  # 1 second of silence
        await websocket.send(audio_data)
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Transcription: {data}")

asyncio.run(test_whisper())
```

---

**Note**: Make sure your Whisper app is running before testing the endpoints. The Nginx configuration will proxy requests to the local application.