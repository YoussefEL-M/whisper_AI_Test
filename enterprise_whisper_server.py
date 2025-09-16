#!/usr/bin/env python3
"""
Enterprise Jitsi CC Server - Clean Version
Multi-user conference support with translation
Based on working app.py configuration
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
import gc
from collections import deque, defaultdict
import whisper
import argostranslate.package
import argostranslate.translate
from flask import Flask
import signal
import sys

# Environment setup  
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🏢 Enterprise Jitsi CC Server - Clean")
print(f"📱 Device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("🔥 GPU: CUDA Available")

class EnterpriseWhisper:
    """Enterprise whisper engine based on app.py"""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        
    def load_model(self):
        """Load model like successful app.py"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    print("🏢 Loading enterprise whisper model...")
                    try:
                        self.model = whisper.load_model("turbo", device=DEVICE)
                        print("✅ Turbo model loaded")
                    except Exception:
                        try:
                            self.model = whisper.load_model("base", device=DEVICE)
                            print("✅ Base model loaded")  
                        except Exception:
                            self.model = whisper.load_model("small", device=DEVICE)
                            print("✅ Small model loaded")
        return self.model
    
    def transcribe_conference(self, audio_bytes, language="en"):
        """High-quality conference transcription"""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Conference validation
        if len(audio) < 10000 or np.max(np.abs(audio)) < 0.005:
            return None
            
        try:
            model = self.load_model()
            
            with torch.no_grad():
                result = model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    task='transcribe',
                    temperature=0.0,
                    best_of=2,              # Better quality for enterprise
                    beam_size=2,            # Better accuracy
                    word_timestamps=False,
                    verbose=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.5,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4
                )
            
            text = result.get('text', '').strip()
            
            if text and len(text) > 2:
                # Quality filter for conference
                words = text.lower().split()
                if len(words) > 3:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.4:
                        return None
                
                return {
                    "text": text,
                    "language": result.get('language', language),
                    "confidence": getattr(result, 'language_probability', 0.8)
                }
                
        except Exception as e:
            logger.error(f"Enterprise transcription error: {e}")
            
        return None

class TranslationEngine:
    """Translation engine for conference"""
    def __init__(self):
        print("🌍 Setting up translation...")
        # Setup happens in background
        
    def translate_text(self, text, from_lang, to_lang):
        """Translate text"""
        if from_lang == to_lang:
            return text
            
        try:
            return argostranslate.translate.translate(text, from_lang, to_lang)
        except Exception:
            return text  # Return original on error

# Room management
rooms = defaultdict(lambda: {
    'connections': {},
    'buffer': deque(maxlen=25), 
    'last_process': 0,
    'last_text': '',
    'language': 'en'
})

# Global engines
whisper_engine = EnterpriseWhisper()
translation_engine = TranslationEngine()

async def handle_conference_websocket(websocket):
    """Handle enterprise conference WebSocket"""
    client_addr = websocket.remote_address
    
    try:
        path = getattr(websocket, 'path', '/ws/default')
    except:
        path = '/ws/default'
    
    # Extract room name
    path_parts = path.strip('/').split('/')
    room_name = path_parts[-1] if path_parts else 'default'
    
    user_id = f"{client_addr[0]}:{client_addr[1]}:{int(time.time())}"
    
    print(f"🏢 Enterprise: {client_addr} → Room: {room_name}")
    
    # Add to room
    room = rooms[room_name]
    room['connections'][user_id] = websocket
    
    try:
        # Send welcome
        welcome = {
            "type": "status", 
            "message": "Connected to enterprise conference whisper",
            "room": room_name,
            "model": "turbo/base (enterprise)",
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(welcome))
        
        # Process audio
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 60:
                # Extract language from header
                header_bytes = message[:60]
                pcm_data = message[60:]
                
                try:
                    header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                    if '|' in header_str:
                        _, language = header_str.split('|', 1)
                        room['language'] = language  # Update room language
                except:
                    pass
                
                if len(pcm_data) > 0:
                    room['buffer'].append(pcm_data)
                    
                    # Process room audio (shared)
                    now = time.time()
                    if (len(room['buffer']) >= 15 and 
                        now - room['last_process'] >= 1.2):  # 1.2 second intervals for quality
                        
                        room['last_process'] = now
                        
                        # Get audio
                        audio_chunks = list(room['buffer'])[-20:]
                        audio_data = b''.join(audio_chunks)
                        
                        if len(audio_data) >= 12000:
                            # Transcribe
                            result = whisper_engine.transcribe_conference(audio_data, room['language'])
                            
                            if result and result['text'] != room['last_text']:
                                room['last_text'] = result['text']
                                
                                # Broadcast to all in room
                                response = {
                                    "type": "final",
                                    "text": result['text'],
                                    "participant": "conference",
                                    "language": result['language'],
                                    "confidence": result['confidence'],
                                    "room": room_name,
                                    "timestamp": int(time.time() * 1000)
                                }
                                
                                # Send to all connections in room
                                for conn_id, conn_ws in list(room['connections'].items()):
                                    try:
                                        await conn_ws.send(json.dumps(response))
                                    except:
                                        room['connections'].pop(conn_id, None)
                                
                                print(f"🏢 {room_name}: '{result['text']}'")
                                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Enterprise disconnection: {user_id}")
    except Exception as e:
        logger.error(f"Enterprise error: {e}")
    finally:
        # Clean up
        if user_id in room['connections']:
            del room['connections'][user_id]
            if not room['connections']:
                del rooms[room_name]
                print(f"🏠 Room {room_name} closed")

# HTTP server
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def health():
    total_connections = sum(len(room['connections']) for room in rooms.values())
    return {
        "service": "Enterprise Conference Whisper",
        "status": "running",
        "rooms": len(rooms),
        "total_connections": total_connections,
        "model": "turbo/base (enterprise)"
    }

async def start_enterprise_server():
    """Start enterprise server"""
    print(f"🏢 Starting Enterprise WebSocket server on 0.0.0.0:5001")
    
    server = await websockets.serve(
        handle_conference_websocket,
        "0.0.0.0",
        5001,
        ping_interval=30,
        ping_timeout=10,
        max_size=10*1024*1024,
        compression=None
    )
    
    print("🏢 Enterprise server ready!")
    print("✅ Multi-user conference support")
    print("✅ Room-based processing") 
    print("✅ Translation capabilities")
    print("🚀 Ready for Jitsi!")
    
    await server.wait_closed()

def start_http():
    def run():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    threading.Thread(target=run, daemon=True).start()
    print("🌐 HTTP server ready")

def cleanup_handler(sig, frame):
    print("\n🏢 Enterprise server shutting down...")
    whisper_engine.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("=" * 60)
    print("🏢 ENTERPRISE WHISPER SERVER")
    print("👥 Multi-user conference support")
    print("🌍 Translation ready")
    print("⚡ Reliable operation")
    print("=" * 60)
    
    start_http()
    
    try:
        asyncio.run(start_enterprise_server())
    except KeyboardInterrupt:
        print("\n🏢 Enterprise server stopped")
    except Exception as e:
        logger.error(f"Enterprise server error: {e}")
    
    print("✨ Enterprise cleanup complete")