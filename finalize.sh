#!/bin/bash

RECORDINGS_DIR="$1"

if [ -z "$RECORDINGS_DIR" ]; then
    RECORDINGS_DIR="/recordings"
fi

echo "[Finalize] Running at $(date)" >> /tmp/finalize.log
echo "[Finalize] RECORDINGS_DIR=$RECORDINGS_DIR" >> /tmp/finalize.log

timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Parent directory where the session folder is located
parent_dir=$(dirname "$RECORDINGS_DIR")
# New name for the folder
new_dir="${parent_dir}/${timestamp}"

echo "[Finalize] Renaming \"$RECORDINGS_DIR\" to \"$new_dir\"" >> /tmp/finalize.log
mv "$RECORDINGS_DIR" "$new_dir"

echo "[Finalize] Done." >> /tmp/finalize.log
