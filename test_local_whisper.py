#!/usr/bin/env python3
"""
Test script for the local whisper server integration
"""
import asyncio
import websockets
import json
import numpy as np
import time

async def test_websocket_connection():
    """Test WebSocket connection to local whisper server"""
    uri = "ws://localhost:5001/local-whisper/ws/test-room"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to local whisper WebSocket server")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            print(f"Received: {response}")
            
            # Create a test audio chunk (1 second of silence at 16kHz)
            sample_rate = 16000
            duration = 1.0  # 1 second
            samples = int(sample_rate * duration)
            
            # Generate PCM16 audio data (silence)
            audio_data = np.zeros(samples, dtype=np.int16)
            pcm_bytes = audio_data.tobytes()
            
            # Create header (participant_id|language)
            participant_id = "test-user"
            language = "en"
            header = f"{participant_id}|{language}".ljust(60, '\0')
            header_bytes = header.encode('utf-8')
            
            # Combine header and audio data
            message = header_bytes + pcm_bytes
            
            print(f"Sending test audio chunk ({len(message)} bytes)")
            await websocket.send(message)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"Received transcription: {response}")
            except asyncio.TimeoutError:
                print("No transcription received within 10 seconds")
            
            print("Test completed successfully")
            
    except Exception as e:
        print(f"Test failed: {e}")

async def test_health_endpoint():
    """Test health endpoint"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:5000/local-whisper/health') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Health check passed: {data}")
                else:
                    print(f"Health check failed: {response.status}")
    except Exception as e:
        print(f"Health check error: {e}")

async def test_rooms_endpoint():
    """Test rooms endpoint"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:5000/local-whisper/rooms') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Rooms status: {data}")
                else:
                    print(f"Rooms check failed: {response.status}")
    except Exception as e:
        print(f"Rooms check error: {e}")

async def main():
    """Run all tests"""
    print("Testing local whisper server integration...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    await test_health_endpoint()
    
    # Test rooms endpoint
    print("\n2. Testing rooms endpoint...")
    await test_rooms_endpoint()
    
    # Test WebSocket connection
    print("\n3. Testing WebSocket connection...")
    await test_websocket_connection()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(main())