#!/usr/bin/env python3
"""
Jitsi Room-Aware Whisper Server
Handles multiple connections per room properly
One transcription process per room, not per connection
Designed for Jitsi mixed audio streams
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

# Environment setup
os.environ.update({
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
    'CUDA_VISIBLE_DEVICES': '0'
})

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🏠 Jitsi Room-Aware Whisper Server")
print(f"📱 Device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("🔥 GPU: CUDA Available")

class JitsiWhisperProcessor:
    """Room-aware whisper processor"""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        
    def load_whisper_model(self):
        """Load model like your app.py"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    print("🎯 Loading Whisper model for Jitsi...")
                    start_time = time.time()
                    
                    try:
                        self.model = whisper.load_model("turbo", device=DEVICE)
                        print(f"✅ Turbo model loaded in {time.time() - start_time:.1f}s")
                    except Exception as e:
                        print(f"Turbo failed: {e}, trying base...")
                        try:
                            self.model = whisper.load_model("base", device=DEVICE)
                            print(f"✅ Base model loaded in {time.time() - start_time:.1f}s")
                        except Exception as e2:
                            print(f"Base failed: {e2}, using small...")
                            self.model = whisper.load_model("small", device=DEVICE)
                            print(f"✅ Small model loaded in {time.time() - start_time:.1f}s")
                    
        return self.model
    
    def transcribe_jitsi_audio(self, audio_bytes, language="en"):
        """Transcribe Jitsi mixed audio"""
        # Convert PCM16 to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Quality checks for Jitsi audio
        if len(audio) < 8000:  # Less than 500ms
            return None
            
        # Check for actual speech (not too strict for Jitsi)
        energy = np.sqrt(np.mean(audio**2))
        if energy < 0.005:  # Very permissive for mixed audio
            return None
            
        try:
            model = self.load_whisper_model()
            
            # High-quality transcription settings
            with torch.no_grad():
                result = model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    task='transcribe',
                    temperature=0.0,
                    best_of=2,                # Better quality for Jitsi
                    beam_size=2,              # Better quality 
                    word_timestamps=False,
                    verbose=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.5,  # More permissive for mixed audio
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.4
                )
            
            text = result.get('text', '').strip()
            
            # Filter for Jitsi context
            if text and len(text) > 1:
                # Check for obvious repetition
                words = text.lower().split()
                if len(words) >= 4:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.5:  # Too repetitive
                        return None
                
                # Filter common Jitsi meeting artifacts
                jitsi_artifacts = ['you are muted', 'unmute', 'muted', 'connection issue']
                if any(artifact in text.lower() for artifact in jitsi_artifacts):
                    return None
                
                return {
                    "type": "final",
                    "text": text,
                    "language": result.get('language', language),
                    "confidence": getattr(result, 'language_probability', 0.8),
                    "timestamp": int(time.time() * 1000)
                }
                
        except Exception as e:
            logger.error(f"Jitsi transcription error: {e}")
            
        return None

class RoomManager:
    """Manages transcription per room, not per connection"""
    def __init__(self):
        self.rooms = {}  # room_name -> RoomState
        self.lock = threading.Lock()
        
    def get_or_create_room(self, room_name):
        """Get or create room state"""
        with self.lock:
            if room_name not in self.rooms:
                self.rooms[room_name] = {
                    'connections': {},  # connection_id -> websocket
                    'buffer': deque(maxlen=30),  # Audio buffer for room
                    'last_process': 0,
                    'last_transcription': '',
                    'transcription_count': 0,
                    'language': 'en'
                }
            return self.rooms[room_name]
    
    def add_connection(self, room_name, conn_id, websocket):
        """Add connection to room"""
        room = self.get_or_create_room(room_name)
        room['connections'][conn_id] = websocket
        print(f"🏠 Room '{room_name}': {len(room['connections'])} connections")
        
    def remove_connection(self, room_name, conn_id):
        """Remove connection from room"""
        with self.lock:
            if room_name in self.rooms:
                room = self.rooms[room_name]
                room['connections'].pop(conn_id, None)
                
                if not room['connections']:  # No more connections
                    del self.rooms[room_name]
                    print(f"🏠 Room '{room_name}' closed")
                else:
                    print(f"🏠 Room '{room_name}': {len(room['connections'])} connections remaining")
    
    def should_process_room(self, room_name):
        """Check if room should be processed (avoid duplicate processing)"""
        room = self.rooms.get(room_name)
        if not room:
            return False
            
        now = time.time()
        has_enough_audio = len(room['buffer']) >= 12  # ~600ms
        enough_time_passed = (now - room['last_process']) >= 1.0  # 1 second intervals for quality
        
        if has_enough_audio and enough_time_passed:
            room['last_process'] = now
            return True
        return False
        
    def add_audio_to_room(self, room_name, audio_data):
        """Add audio to room buffer"""
        room = self.rooms.get(room_name)
        if room:
            room['buffer'].append(audio_data)
            
    def get_room_audio(self, room_name):
        """Get accumulated audio for room"""
        room = self.rooms.get(room_name)
        if not room or len(room['buffer']) < 10:
            return None
            
        # Get substantial audio chunk
        audio_chunks = list(room['buffer'])[-20:]  # Last ~1 second
        audio_data = b''.join(audio_chunks)
        
        return audio_data if len(audio_data) >= 8000 else None
        
    async def broadcast_to_room(self, room_name, message):
        """Send transcription to all connections in room"""
        room = self.rooms.get(room_name)
        if not room:
            return
            
        # Send to all connections in room
        for conn_id, websocket in list(room['connections'].items()):
            try:
                await websocket.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send to {conn_id}: {e}")
                # Remove dead connection
                room['connections'].pop(conn_id, None)

# Global instances
jitsi_processor = JitsiWhisperProcessor()
room_manager = RoomManager()

async def handle_jitsi_websocket(websocket):
    """Handle WebSocket with room-aware processing"""
    client_addr = websocket.remote_address
    
    try:
        path = getattr(websocket, 'path', '/ws/default')
    except:
        path = '/ws/default'
    
    # Extract room name from path
    path_parts = path.strip('/').split('/')
    room_name = path_parts[-1] if path_parts else 'default'
    
    conn_id = f"{client_addr[0]}:{client_addr[1]}:{int(time.time())}"
    
    print(f"🏠 Jitsi connection: {client_addr} → Room: {room_name}")
    
    # Add to room
    room_manager.add_connection(room_name, conn_id, websocket)
    
    try:
        # Send welcome
        welcome = {
            "type": "status",
            "message": "Connected to Jitsi room whisper service",
            "room": room_name,
            "model": "turbo/base (Jitsi optimized)",
            "mode": "room_shared_transcription",
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(welcome))
        
        # Process incoming Jitsi audio
        async for message in websocket:
            if isinstance(message, bytes) and len(message) > 60:
                # Parse Skynet format
                header_bytes = message[:60]
                pcm_data = message[60:]
                
                if len(pcm_data) > 0:
                    # Extract language from header
                    try:
                        header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                        if '|' in header_str:
                            participant_id, language = header_str.split('|', 1)
                        else:
                            language = 'en'
                    except:
                        language = 'en'
                    
                    # Add audio to room buffer (shared processing)
                    room_manager.add_audio_to_room(room_name, pcm_data)
                    
                    # Process at room level (not per connection)
                    if room_manager.should_process_room(room_name):
                        audio_data = room_manager.get_room_audio(room_name)
                        if audio_data:
                            # Process once per room
                            result = jitsi_processor.transcribe_jitsi_audio(audio_data, language)
                            if result:
                                result['participant'] = 'jitsi-room'
                                result['room'] = room_name
                                
                                # Broadcast to ALL connections in room
                                await room_manager.broadcast_to_room(room_name, result)
                                print(f"🏠 {room_name} → ALL: '{result['text']}'")
                                
                                # Update room stats
                                room = room_manager.rooms.get(room_name)
                                if room:
                                    room['transcription_count'] += 1
                                    room['last_transcription'] = result['text']
                                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Jitsi disconnection: {conn_id}")
    except Exception as e:
        logger.error(f"Jitsi connection error: {e}")
    finally:
        room_manager.remove_connection(room_name, conn_id)

# HTTP server
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def health():
    total_connections = sum(len(room['connections']) for room in room_manager.rooms.values())
    return {
        "service": "Jitsi Room-Aware Whisper",
        "status": "running",
        "rooms": len(room_manager.rooms),
        "total_connections": total_connections,
        "model": "turbo/base (room-shared)",
        "mode": "jitsi_mixed_audio"
    }

@app.route('/rooms')
def rooms():
    room_info = {}
    for room_name, room in room_manager.rooms.items():
        room_info[room_name] = {
            "connections": len(room['connections']),
            "transcriptions": room['transcription_count'],
            "last_text": room['last_transcription'][:50] + "..." if len(room['last_transcription']) > 50 else room['last_transcription']
        }
    return {"rooms": room_info}

async def start_jitsi_server():
    """Start room-aware server"""
    print(f"🏠 Starting Jitsi Room-Aware WebSocket server on 0.0.0.0:5001")
    
    server = await websockets.serve(
        handle_jitsi_websocket,
        "0.0.0.0",
        5001,
        ping_interval=30,
        ping_timeout=15,
        max_size=10*1024*1024,
        compression=None
    )
    
    print("🏠 Jitsi Room-Aware server ready!")
    print("✅ Room-based transcription (one process per room)")
    print("✅ Multiple connections per room supported")
    print("✅ Jitsi mixed audio stream processing")
    print("✅ Prevents hallucinations from connection conflicts")
    print("✅ Compatible with your skynet-integration.js")
    
    await server.wait_closed()

def start_http():
    def run():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    threading.Thread(target=run, daemon=True).start()
    print("🌐 HTTP server on port 5000")

def cleanup_handler(sig, frame):
    print("\n🏠 Jitsi server shutting down...")
    jitsi_processor.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("=" * 65)
    print("🏠 JITSI ROOM-AWARE WHISPER SERVER")
    print("🎯 One transcription process per room")
    print("👥 Multiple connections per room supported")
    print("🎙️  Jitsi mixed audio stream processing")
    print("🚫 Prevents connection-based hallucinations")
    print("=" * 65)
    
    start_http()
    
    try:
        asyncio.run(start_jitsi_server())
    except KeyboardInterrupt:
        print("\n🏠 Jitsi server stopped")
    except Exception as e:
        logger.error(f"Jitsi server error: {e}")
    
    print("✨ Jitsi cleanup complete")