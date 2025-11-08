#!/bin/bash

set -e

# check if /data/downloaded file is present and if it is not
# download the model files from huggingface.

DOWNLOADED_FILE=/model/downloaded

# if the FORCE_DOWNLOAD env variable is set, delete the /data/downloaded file.
if [ -n $FORCE_DOWNLOAD ]; then
    rm -f $DOWNLOADED_FILE
fi

if [ ! -f $DOWNLOADED_FILE ]; then
    echo "[@] Some model files not downloaded, downloading..."
    #wget --verbose -O /data/kokoro.pth https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth
    #wget --verbose -O /data/voice.pt https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/${VOICE_NAME}.pt
    wget --verbose -O /model/config.json https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
    touch /model/downloaded
fi

# run the websocket server.
# TODO