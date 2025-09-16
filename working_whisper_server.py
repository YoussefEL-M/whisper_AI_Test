#!/usr/bin/env python3
"""
Working Whisper Server - Immediate startup for Jitsi CC
Based on your app.py but optimized for live conference use
Simple, fast, reliable
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
from collections import deque, defaultdict
from flask import Flask
import signal
import sys
import gc
import whisper

# Simple setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🎯 Working Whisper Server - Conference Ready")
print(f"📱 Device: {DEVICE}")

class WorkingWhisper:
    """Simple working whisper based on your app.py"""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        
    def load_model(self):
        """Load model like your successful app.py"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    print("🎯 Loading whisper model (app.py style)...")
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
        """Conference transcription"""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Conference audio validation
        if len(audio) < 8000 or np.max(np.abs(audio)) < 0.005:
            return None
            
        try:
            model = self.load_model()
            
            with torch.no_grad():
                result = model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    task='transcribe',
                    temperature=0.0,
                    best_of=1,
                    beam_size=1,
                    word_timestamps=False,
                    verbose=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.5,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4
                )
            
            text = result.get('text', '').strip()
            
            if text and len(text) > 2:
                # Basic quality filter
                words = text.lower().split()
                if len(words) > 3:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.4:  # Too repetitive
                        return None
                
                return text
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            
        return None

# Room management
rooms = defaultdict(lambda: {
    'connections': {},
    'buffer': deque(maxlen=25),
    'last_process': 0,
    'last_text': ''
})

# Global whisper
whisper_engine = WorkingWhisper()

async def handle_conference_websocket(websocket):
    """Handle conference WebSocket connections"""
    client_addr = websocket.remote_address
    
    try:
        path = getattr(websocket, 'path', '/ws/default')
    except:
        path = '/ws/default'
    
    # Extract room name
    path_parts = path.strip('/').split('/')
    room_name = path_parts[-1] if path_parts else 'default'
    
    user_id = f"{client_addr[0]}:{client_addr[1]}:{int(time.time())}"
    
    print(f"🎯 Conference: {client_addr} → {room_name}")
    
    # Add to room
    rooms[room_name]['connections'][user_id] = websocket
    
    try:
        # Send welcome
        welcome = {
            "type": "status",
            "message": "Connected to working conference whisper",
            "room": room_name,
            "model": "turbo/base/small",
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(welcome))
        
        # Process messages
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 60:
                pcm_data = message[60:]
                
                if len(pcm_data) > 0:
                    room = rooms[room_name]
                    room['buffer'].append(pcm_data)
                    
                    # Process room audio (shared)
                    now = time.time()
                    if (len(room['buffer']) >= 15 and 
                        now - room['last_process'] >= 1.0):  # 1 second intervals
                        
                        room['last_process'] = now
                        
                        # Get audio
                        audio_chunks = list(room['buffer'])[-20:]
                        audio_data = b''.join(audio_chunks)
                        
                        if len(audio_data) >= 10000:  # Good chunk size
                            # Transcribe
                            text = whisper_engine.transcribe_conference(audio_data, "en")
                            
                            if text and text != room['last_text']:
                                room['last_text'] = text
                                
                                # Send to all connections in room
                                response = {
                                    "type": "final",
                                    "text": text,
                                    "participant": "conference",
                                    "room": room_name,
                                    "timestamp": int(time.time() * 1000)
                                }
                                
                                # Broadcast to all users in room
                                for conn_id, conn_ws in list(room['connections'].items()):
                                    try:
                                        await conn_ws.send(json.dumps(response))
                                    except:
                                        room['connections'].pop(conn_id, None)
                                
                                print(f"🎯 {room_name}: '{text}'")
                                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Conference disconnection: {user_id}")
    except Exception as e:
        logger.error(f"Conference error: {e}")
    finally:
        # Clean up
        if room_name in rooms and user_id in rooms[room_name]['connections']:
            del rooms[room_name]['connections'][user_id]
            if not rooms[room_name]['connections']:
                del rooms[room_name]
                print(f"🏠 Room {room_name} closed")

# Simple HTTP server
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def health():
    total_connections = sum(len(room['connections']) for room in rooms.values())
    return {
        "service": "Working Conference Whisper",
        "status": "running",
        "rooms": len(rooms),
        "total_connections": total_connections
    }

async def start_working_server():
    """Start the working server immediately"""
    print(f"🎯 Starting Conference WebSocket server on 0.0.0.0:5001")
    
    server = await websockets.serve(
        handle_conference_websocket,
        "0.0.0.0",
        5001,
        ping_interval=30,
        ping_timeout=10,
        max_size=10*1024*1024,
        compression=None
    )
    
    print("🎯 Working server ready!")
    print("✅ Conference room support")
    print("✅ Multi-user without conflicts")
    print("✅ Quality transcription")
    print("🚀 Ready for connections!")
    
    await server.wait_closed()

def start_http():
    def run():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    threading.Thread(target=run, daemon=True).start()
    print("🌐 HTTP server ready")

def cleanup(sig, frame):
    print("\n🎯 Working server stopping...")
    whisper_engine.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("=" * 50)
    print("🎯 WORKING CONFERENCE WHISPER SERVER")
    print("🚀 Immediate startup, reliable operation")
    print("👥 Multi-user conference support")
    print("=" * 50)
    
    start_http()
    
    try:
        asyncio.run(start_working_server())
    except KeyboardInterrupt:
        print("\n🎯 Working server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")
    
    print("✨ Cleanup complete")