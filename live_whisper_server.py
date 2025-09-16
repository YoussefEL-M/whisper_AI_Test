#!/usr/bin/env python3
"""
Live Whisper Server - Based on working app.py configuration
Uses turbo/base models like your successful Jitsi finalize script
Optimized for live transcription without hallucinations
"""

import asyncio
import json
import numpy as np
import websockets
import torch
import os
import logging
import time
import threading
from collections import deque
from flask import Flask
import signal
import sys
import gc
import whisper

# Environment setup (based on your app.py)
os.environ.update({
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
    'CUDA_VISIBLE_DEVICES': '0'
})

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🎯 Live Whisper Server - Based on Working app.py")
print(f"📱 Device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        print("🔥 GPU: CUDA Available")

class LiveWhisperProcessor:
    """Based on your working app.py whisper implementation"""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        
    def load_whisper_model(self):
        """Load Whisper model exactly like your app.py"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    print("🎯 Loading Whisper model (app.py style)...")
                    start_time = time.time()
                    
                    try:
                        # Try turbo first (like your app.py)
                        self.model = whisper.load_model("turbo", device=DEVICE)
                        print(f"✅ Turbo model loaded in {time.time() - start_time:.1f}s")
                    except (torch.cuda.OutOfMemoryError, Exception) as e:
                        print(f"Turbo failed: {e}")
                        print("🔄 Falling back to base model (like app.py)...")
                        try:
                            self.model = whisper.load_model("base", device=DEVICE)
                            print(f"✅ Base model loaded in {time.time() - start_time:.1f}s")
                        except Exception as e2:
                            print(f"Base failed: {e2}")
                            print("🔄 Final fallback to small model...")
                            self.model = whisper.load_model("small", device=DEVICE)
                            print(f"✅ Small model loaded in {time.time() - start_time:.1f}s")
                    
        return self.model
    
    def unload_whisper_model(self):
        """Unload model like your app.py"""
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("🗑️ Whisper model unloaded")
    
    def transcribe_live(self, audio_bytes, language="en"):
        """Live transcription with app.py quality"""
        # Convert PCM16 to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Basic silence detection (not too strict)
        if len(audio) < 4000 or np.max(np.abs(audio)) < 0.01:
            return None
            
        try:
            model = self.load_whisper_model()
            
            # Use app.py style transcription settings
            with torch.no_grad():
                result = model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    task='transcribe',
                    temperature=0.0,          # Deterministic
                    best_of=1,                # Speed vs quality balance
                    beam_size=1,              # Fast decoding
                    word_timestamps=False,
                    verbose=False,
                    condition_on_previous_text=False,  # Prevent context drift
                    no_speech_threshold=0.6,  # Better silence detection
                    logprob_threshold=-1.0,   # Filter low confidence
                    compression_ratio_threshold=2.4  # Filter repetitive text
                )
            
            text = result.get('text', '').strip()
            
            # Quality filtering (not too aggressive)
            if text and len(text) > 2:
                # Filter obvious repetitive patterns
                words = text.lower().split()
                if len(words) >= 3:
                    # Check for excessive repetition
                    unique_words = len(set(words))
                    total_words = len(words)
                    if unique_words / total_words < 0.4:  # More than 60% repetition
                        return None
                
                return {
                    "type": "final",
                    "text": text,
                    "language": result.get('language', language),
                    "confidence": getattr(result, 'language_probability', 0.8),
                    "timestamp": int(time.time() * 1000)
                }
                
        except Exception as e:
            logger.error(f"Live transcription error: {e}")
            
        return None

# Global processor
live_whisper = LiveWhisperProcessor()
connections = {}

class AudioBuffer:
    """Smart audio buffering for live transcription"""
    def __init__(self):
        self.chunks = deque(maxlen=25)  # ~1.25 seconds
        self.last_process = 0
        self.min_audio_length = 6400   # ~400ms at 16kHz
        self.process_interval = 0.8    # Process every 800ms for quality
        
    def add_chunk(self, data):
        """Add audio chunk"""
        self.chunks.append(data)
        
    def should_process(self):
        """Process when we have enough audio and time"""
        now = time.time()
        has_enough_audio = len(self.chunks) >= 10  # ~500ms
        enough_time_passed = (now - self.last_process) >= self.process_interval
        
        if has_enough_audio and enough_time_passed:
            self.last_process = now
            return True
        return False
        
    def get_audio_data(self):
        """Get substantial audio chunk for quality transcription"""
        if len(self.chunks) < 8:
            return None
            
        # Take a good chunk for quality (like app.py would process)
        audio_chunks = list(self.chunks)[-20:]  # Last ~1 second
        audio_data = b''.join(audio_chunks)
        
        return audio_data if len(audio_data) >= self.min_audio_length else None

async def handle_live_websocket(websocket):
    """Handle WebSocket for live transcription"""
    client_addr = websocket.remote_address
    
    try:
        path = getattr(websocket, 'path', '/ws/default')
    except:
        path = '/ws/default'
    
    # Extract room name
    path_parts = path.strip('/').split('/')
    room_name = path_parts[-1] if path_parts else 'default'
    
    print(f"🎯 Live connection: {client_addr} → {room_name}")
    
    # Create buffer for this connection
    buffer = AudioBuffer()
    conn_id = f"{client_addr[0]}:{client_addr[1]}"
    connections[conn_id] = {
        'websocket': websocket,
        'room': room_name,
        'buffer': buffer,
        'transcription_count': 0,
        'start_time': time.time()
    }
    
    try:
        # Send welcome message
        welcome = {
            "type": "status",
            "message": "Connected to live whisper service",
            "room": room_name,
            "model": "turbo/base (app.py style)",
            "features": ["multilingual", "quality_filtering", "live_processing"],
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(welcome))
        
        # Process incoming audio
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 60:
                # Parse Skynet format from your integration
                header_bytes = message[:60]
                pcm_data = message[60:]
                
                if len(pcm_data) > 0:
                    buffer.add_chunk(pcm_data)
                    
                    # Process when ready
                    if buffer.should_process():
                        audio_data = buffer.get_audio_data()
                        if audio_data:
                            # Extract language from header
                            try:
                                header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                                if '|' in header_str:
                                    _, language = header_str.split('|', 1)
                                else:
                                    language = 'en'
                            except:
                                language = 'en'
                            
                            # Transcribe with app.py quality
                            result = live_whisper.transcribe_live(audio_data, language)
                            if result:
                                result['participant'] = 'live-whisper'
                                result['room'] = room_name
                                
                                await websocket.send(json.dumps(result))
                                connections[conn_id]['transcription_count'] += 1
                                print(f"🎯 {room_name}: '{result['text']}'")
                                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Live disconnection: {conn_id}")
    except Exception as e:
        logger.error(f"Live connection error: {e}")
    finally:
        if conn_id in connections:
            count = connections[conn_id]['transcription_count']
            duration = time.time() - connections[conn_id]['start_time']
            print(f"📊 {conn_id} - Transcriptions: {count}, Duration: {duration:.1f}s")
            del connections[conn_id]

# HTTP server
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def health():
    return {
        "service": "Live Whisper Server",
        "status": "running",
        "connections": len(connections),
        "model": "turbo/base (app.py style)",
        "features": ["multilingual", "quality_filtering", "live_processing"]
    }

@app.route('/unload_model', methods=['POST'])
def unload_model():
    live_whisper.unload_whisper_model()
    return {"status": "model_unloaded"}

async def start_live_server():
    """Start the live server"""
    print(f"🎯 Starting Live WebSocket server on 0.0.0.0:5001")
    
    server = await websockets.serve(
        handle_live_websocket,
        "0.0.0.0",
        5001,
        ping_interval=30,
        ping_timeout=10,
        max_size=5*1024*1024,
        compression=None
    )
    
    print("🎯 Live Whisper server ready!")
    print("✅ Based on your working app.py configuration")
    print("✅ Turbo/Base models for quality")
    print("✅ Multilingual support (Danish, English, etc.)")
    print("✅ Quality filtering without over-filtering")
    print("✅ Compatible with GEMTIMPROVEDSKYNET0309ENHANCED.js")
    
    await server.wait_closed()

def start_http():
    def run():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    threading.Thread(target=run, daemon=True).start()
    print("🌐 HTTP server on port 5000")

def cleanup_handler(sig, frame):
    print("\n🎯 Live server shutting down...")
    live_whisper.unload_whisper_model()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("=" * 60)
    print("🎯 LIVE WHISPER SERVER")
    print("📋 Based on your working app.py")
    print("🌍 Multilingual (Danish, English, etc.)")
    print("✨ Quality transcription without hallucinations")
    print("=" * 60)
    
    start_http()
    
    try:
        asyncio.run(start_live_server())
    except KeyboardInterrupt:
        print("\n🎯 Live server stopped")
    except Exception as e:
        logger.error(f"Live server error: {e}")
    finally:
        live_whisper.unload_whisper_model()
    
    print("✨ Live cleanup complete")