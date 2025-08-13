#!/bin/bash

# Ensure the script fails fast
set -e

# Path to the video file passed as argument
VIDEO_PATH="$1"

# Check if video path was provided
if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: $0 /path/to/video.mp4"
    exit 1
fi

# Extract the directory and filename parts
VIDEO_DIR=$(dirname "$VIDEO_PATH")
VIDEO_FILENAME=$(basename "$VIDEO_PATH")
VIDEO_BASENAME="${VIDEO_FILENAME%.*}"

# Define the output .srt path
SRT_PATH="${VIDEO_DIR}/${VIDEO_BASENAME}.srt"

# Endpoint for your transcription API
API_ENDPOINT="http://your-api/transcribe"

echo "Sending $VIDEO_FILENAME to API..."

# Use curl to send the video and save the resulting SRT
curl -s -X POST "$API_ENDPOINT" \
    -F "file=@$VIDEO_PATH" \
    -o "$SRT_PATH"

echo "Transcription saved to $SRT_PATH"