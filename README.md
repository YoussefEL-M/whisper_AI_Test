# 🎙️ Whisper AI Transcription Service

A powerful, GPU-accelerated speech-to-text transcription service with real-time capabilities, multilingual support, and AI-powered summarization.

## ✨ Features

### 🚀 Core Functionality
- **Real-time Transcription**: Live audio streaming with WebSocket support
- **File Upload**: Support for audio/video files (MP3, WAV, FLAC, M4A, OGG, WebM, MP4)
- **GPU Acceleration**: NVIDIA CUDA support for fast processing
- **Turbo Model**: Uses OpenAI's fastest Whisper model for speed
- **Multilingual**: Auto-detection and support for multiple languages

### 🌍 Language Support
- **Auto-detection**: Automatically detects language from audio
- **Supported Languages**: Danish, English, Swedish, German, French, Spanish, Korean, and more
- **Translation**: Danish ↔ English translation capabilities
- **Summarization**: AI-powered text summarization using TinyLlama

### 🔧 Technical Features
- **WebSocket API**: Real-time bidirectional communication
- **REST API**: HTTP endpoints for file upload and processing
- **Health Monitoring**: Built-in health checks and status monitoring
- **Memory Management**: Optimized GPU memory usage with model unloading
- **Docker Support**: Containerized deployment with Docker Compose

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   Flask Server   │◄──►│  Whisper Model  │
│   (Frontend)    │    │   (Port 5000)    │    │   (GPU/Turbo)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐             │
         └─────────────►│  WebSocket API   │◄────────────┘
                        │   (Port 5001)    │
                        └──────────────────┘
                                 │
                        ┌──────────────────┐
                        │  TinyLlama Model │
                        │  (Summarization) │
                        └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB RAM
- 4GB+ free disk space for models

### 1. Clone and Setup
```bash
git clone <your-repo>
cd whisper
```

### 2. Start the Service
```bash
# Start with Docker Compose (Recommended)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Access the Application
- **Web Interface**: http://localhost:5000
- **Health Check**: http://localhost:5000/local-whisper/health
- **Production**: https://rosetta.semaphor.dk (if configured)

## 📡 API Endpoints

### HTTP Endpoints

#### Health Check
```bash
GET /local-whisper/health
```
Returns service status and model availability.

#### File Upload & Transcription
```bash
POST /transcribe
Content-Type: multipart/form-data
```
Upload audio/video file for transcription.

**Response:**
```json
{
  "status": "success",
  "transcription": "Transcribed text here...",
  "language": "en",
  "duration": 120.5
}
```

#### SRT Subtitle Generation
```bash
POST /transcribe/srt
Content-Type: multipart/form-data
```
Generate SRT subtitle file from audio.

#### Text Summarization
```bash
POST /summarize_from_json
Content-Type: application/json

{
  "text": "Text to summarize...",
  "language": "en"
}
```

### WebSocket API

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:5001');
```

#### Real-time Audio Streaming
Send 16-bit PCM audio data (16kHz, mono) for real-time transcription.

**Audio Format:**
- Sample Rate: 16,000 Hz
- Bit Depth: 16-bit
- Channels: Mono
- Chunk Size: 1 second (32,000 bytes)

#### WebSocket Messages

**Server Status:**
```json
{
  "type": "status",
  "message": "Connected to transcription service"
}
```

**Transcription Result:**
```json
{
  "type": "transcription",
  "text": "Transcribed text here",
  "language": "en",
  "confidence": 0.95
}
```

**Error Response:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

## 🐳 Docker Configuration

### Docker Compose
The service uses `docker-compose.yml` with GPU support:

```yaml
version: '3.8'
services:
  whisper:
    build: .
    container_name: local-whisper
    ports:
      - "5000:5000"
      - "5001:5001"
    volumes:
      - ./models:/app/models
      - ./installed_models:/app/installed_models
      - ./TinyLlama-1.1B-Chat-v1.0:/app/TinyLlama-1.1B-Chat-v1.0
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES=0`: Use first GPU
- `PYTHONPATH=/app`: Python module path
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`: CUDA device ordering
- `PYTORCH_NVML_BASED_CUDA_CHECK=1`: CUDA compatibility check
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`: Memory optimization

## 🔧 Management Commands

### Docker Compose Commands
```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# Rebuild and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Container Management
```bash
# Enter container shell
docker exec -it local-whisper bash

# Check GPU status
docker exec local-whisper python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check health
curl http://localhost:5000/local-whisper/health
```

## 🌐 Production Deployment

### Nginx Configuration
The service integrates with Nginx for production deployment:

**Endpoints:**
- `https://rosetta.semaphor.dk/` - Main application
- `https://rosetta.semaphor.dk/local-whisper/health` - Health check
- `wss://rosetta.semaphor.dk/ws` - WebSocket connection

### SSL/TLS
- Uses wildcard certificate: `wildcard.semaphor.dk`
- Supports HTTPS and WSS protocols
- CORS enabled for cross-origin requests

## 📊 Performance

### GPU Acceleration
- **NVIDIA RTX A2000 12GB**: Primary GPU
- **CUDA Support**: Full PyTorch CUDA integration
- **Memory Management**: Optimized GPU memory allocation
- **Model Loading**: On-demand model loading to save memory

### Processing Speed
- **Turbo Model**: Fastest Whisper model available
- **Real-time**: Sub-second latency for live transcription
- **Batch Processing**: Efficient handling of large files (400MB+)

### Supported File Formats
- **Audio**: MP3, WAV, FLAC, M4A, OGG
- **Video**: MP4, WebM (audio extraction)
- **Size Limit**: Up to 500MB per file

## 🛠️ Development

### Project Structure
```
whisper/
├── app.py                 # Main Flask application
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Container build instructions
├── requirements.txt      # Python dependencies
├── templates/
│   └── indexY.html       # Web interface
├── static/               # Static assets
├── models/               # Whisper model storage
├── installed_models/     # Translation model cache
└── README.md            # This file
```

### Dependencies
- **Flask**: Web framework
- **Flask-SocketIO**: WebSocket support
- **OpenAI Whisper**: Speech recognition
- **Transformers**: AI model support
- **TinyLlama**: Text summarization
- **ArgosTranslate**: Translation services
- **PyTorch**: Deep learning framework (CUDA-enabled)

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check if ports are available
netstat -tlnp | grep -E ":5000|:5001"

# Check Docker logs
docker-compose logs

# Verify GPU access
docker exec local-whisper nvidia-smi
```

#### GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker info | grep -i nvidia

# Verify CUDA in container
docker exec local-whisper python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Check container resource usage
docker stats local-whisper

# Monitor GPU memory
docker exec local-whisper nvidia-smi
```

#### Model Loading Errors
```bash
# Check model directory
ls -la ./models/

# Verify model files in container
docker exec local-whisper ls -la /app/models/
```

### Health Checks
```bash
# Service health
curl http://localhost:5000/local-whisper/health

# GPU status
docker exec local-whisper python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
```

## 📈 Monitoring

### Health Endpoint
```json
{
  "status": "healthy",
  "service": "local_streaming_whisper",
  "whisper_loaded": true,
  "llama_loaded": true,
  "connections": 2
}
```

### Log Monitoring
```bash
# Follow logs in real-time
docker-compose logs -f

# Check specific service logs
docker logs local-whisper
```

## 🔄 Updates & Maintenance

### Updating Models
```bash
# Rebuild container to update models
docker-compose down
docker-compose up -d --build
```

### Backup Models
```bash
# Backup model directory
tar -czf models-backup.tar.gz ./models/
```

## 📝 Usage Examples

### JavaScript Client
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:5001');

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
function sendAudio(audioBuffer) {
    ws.send(audioBuffer);
}
```

### Python Client
```python
import requests
import websockets
import json

# File upload
def transcribe_file(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/transcribe', files=files)
    return response.json()

# WebSocket streaming
async def stream_audio():
    uri = "ws://localhost:5001"
    async with websockets.connect(uri) as websocket:
        # Send audio data
        audio_data = b'\x00' * 32000  # 1 second of silence
        await websocket.send(audio_data)
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Transcription: {data}")
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review container logs: `docker-compose logs`
3. Verify GPU status: `docker exec local-whisper nvidia-smi`
4. Check health endpoint: `curl http://localhost:5000/local-whisper/health`

---

**🎉 Enjoy your AI-powered transcription service!**