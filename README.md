# kokoro-ws
Docker image for minimal websocket implementation of Kokoro TTS.

## Preparing the model files
To build the docker image first create a model/ folder, which contains the files:
- `kokoro.onnx`, the onnx file of the Kokoro TTS model
- `voice.npy`, the voice tensor saved using `numpy.save`
- `config.json`, the config file of the Kokoro TTS model.

This can easily be done by running the following command:
```bash
docker compose run --rm download

# after creating the model files, you may also remove the (rather large) download image
docker image rm kokoro-ws-download 
```
To change the downloaded voice set the `VOICE_NAME` environment variable in the docker compose file. 
The files are redownloaded only if they are missing.

## Running the image
To run the docker image use the command:
```bash
# To evaluate the model on the CPU
docker compose up cpu

# To evaluate the model on the GPU
docker compose up gpu
```