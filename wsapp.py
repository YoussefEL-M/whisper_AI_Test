import asyncio
import websockets
import json
import io
import numpy as np
import whisper
from flask import Flask
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global whisper model
whisper_model = None

def load_whisper_model():
    """Load the Whisper model once"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")  # or "small", "medium", "large"
        logger.info("Whisper model loaded successfully")
    return whisper_model

async def handle_websocket(websocket, path):
    """Handle WebSocket connections for live transcription"""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    
    try:
        # Load model if not already loaded
        model = load_whisper_model()
        
        # Send confirmation
        await websocket.send(json.dumps({
            "type": "status", 
            "message": "Connected to transcription service"
        }))
        
        audio_buffer = b''
        chunk_size = 16000 * 2  # 1 second of 16kHz 16-bit audio
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Received PCM audio data
                    audio_buffer += message
                    
                    # Process when we have enough data
                    if len(audio_buffer) >= chunk_size:
                        # Convert PCM bytes to numpy array
                        audio_data = np.frombuffer(audio_buffer[:chunk_size], dtype=np.int16)
                        
                        # Normalize to [-1, 1] range for Whisper
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        # Transcribe with Whisper
                        result = model.transcribe(
                            audio_float, 
                            language='en',  # or None for auto-detection
                            task='transcribe',
                            fp16=False
                        )
                        
                        text = result["text"].strip()
                        
                        if text:
                            logger.info(f"Transcribed: {text}")
                            # Send transcription back
                            await websocket.send(json.dumps({
                                "type": "transcription",
                                "text": text,
                                "participant": "Speaker"
                            }))
                        
                        # Keep remaining buffer
                        audio_buffer = audio_buffer[chunk_size:]
                        
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Transcription error: {str(e)}"
                }))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

def start_websocket_server():
    """Start the WebSocket server"""
    logger.info("Starting WebSocket server on port 5001...")
    start_server = websockets.serve(handle_websocket, "0.0.0.0", 5001)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server)
    loop.run_forever()

# Flask app for health checks
app = Flask(__name__)

@app.route('/health')
def health():
    return {"status": "ok", "service": "whisper-transcription"}

@app.route('/live-cc')
def live_cc_info():
    return {
        "message": "WebSocket endpoint for live captions", 
        "websocket_url": "ws://rosetta.semaphor.dk:5001/live-cc",
        "status": "ready"
    }

if __name__ == "__main__":
    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    print("Starting Flask HTTP server on 0.0.0.0:5000")
    print("WebSocket server running on 0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5000, debug=True)
