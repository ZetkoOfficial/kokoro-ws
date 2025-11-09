#!/bin/bash
set -e 

# check if .onnx file does not exist
if [ ! -f /model/kokoro.onnx ]; then
    DOWNLOAD_MODEL=true
fi

# check if voice file does not exist
if [ ! -f /model/voice.npy ]; then
    DOWNLOAD_VOICE=true
fi

# check if config file does not exist
if [ ! -f /model/config.json ]; then
    DOWNLOAD_CONFIG=true
fi


# Download files if neccesary
if [ "$DOWNLOAD_VOICE" = true ]; then
    wget -O /application/voice.pt "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/${VOICE_NAME:-af_bella}.pt"
fi
if [ "$DOWNLOAD_CONFIG" = true ]; then
    wget -O /model/config.json "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"
fi

# create the remaining files
DOWNLOAD_MODEL=$DOWNLOAD_MODEL DOWNLOAD_VOICE=$DOWNLOAD_VOICE python3 downloader.py