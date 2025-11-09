# kokoro-ws
Docker image for minimal websocket implementation of Kokoro TTS.

## Building the docker image 
To build the docker image first create a model/ folder, which contains the files:
- `kokoro.onnx`, the onnx file of the Kokoro TTS model
- `voice.npy`, the voice tensor saved using `numpy.save`
- `config.json`, the config file of the Kokoro TTS model

## Running the image
To run the image use the command:
```bash
# To evaluate the model on the CPU
docker compose up --build cpu

# To evaluate the model on the GPU
docker compose up --build gpu
```