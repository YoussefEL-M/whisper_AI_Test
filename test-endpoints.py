#!/usr/bin/env python3
"""
Test script to verify Whisper app endpoints
"""

import requests
import json
import websockets
import asyncio
import time

def test_http_endpoints():
    """Test HTTP endpoints"""
    print("🧪 Testing HTTP Endpoints")
    print("=" * 40)
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test live-cc endpoint
    try:
        response = requests.get(f"{base_url}/live-cc", timeout=5)
        print(f"✅ Live CC endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Live CC endpoint failed: {e}")

async def test_websocket():
    """Test WebSocket connection"""
    print("\n🔌 Testing WebSocket Connection")
    print("=" * 40)
    
    try:
        uri = "ws://localhost:5001"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Wait for initial message
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(message)
                print(f"✅ Received initial message: {data}")
            except asyncio.TimeoutError:
                print("⚠️ No initial message received")
            
            # Send a test message (empty audio data)
            test_audio = b'\x00' * 32000  # 1 second of silence
            await websocket.send(test_audio)
            print("✅ Sent test audio data")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)
                print(f"✅ Received response: {data}")
            except asyncio.TimeoutError:
                print("⚠️ No response received (this is normal for silence)")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

def test_nginx_endpoints():
    """Test Nginx endpoints"""
    print("\n🌐 Testing Nginx Endpoints")
    print("=" * 40)
    
    nginx_urls = [
        "https://rosetta.semaphor.dk/whisper-health",
        "https://rosetta.semaphor.dk/whisper-live-cc"
    ]
    
    for url in nginx_urls:
        try:
            response = requests.get(url, timeout=5, verify=False)
            print(f"✅ {url}: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"❌ {url} failed: {e}")

def main():
    print("🎙️ Whisper App Endpoint Tester")
    print("=" * 50)
    
    # Test local endpoints
    test_http_endpoints()
    
    # Test WebSocket
    asyncio.run(test_websocket())
    
    # Test Nginx endpoints
    test_nginx_endpoints()
    
    print("\n🎉 Testing completed!")
    print("\n📋 Available Endpoints:")
    print("   Local HTTP:")
    print("     - http://localhost:5000/health")
    print("     - http://localhost:5000/live-cc")
    print("   Local WebSocket:")
    print("     - ws://localhost:5001")
    print("   Nginx HTTPS (via rosetta.semaphor.dk):")
    print("     - https://rosetta.semaphor.dk/whisper-health")
    print("     - https://rosetta.semaphor.dk/whisper-live-cc")
    print("     - wss://rosetta.semaphor.dk/whisper-ws")
    print("     - wss://rosetta.semaphor.dk/whisper-live-cc-ws")

if __name__ == "__main__":
    main()