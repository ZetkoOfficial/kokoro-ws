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

## Usage
The started docker container runs a websocket server on port 8888.
Speech to text is started by sending a message of the form:
```json
{
    "text": "The text, which you want to turn into speech.",
    "speed": 1,
    "threshold_rms": 0.01
}
```
The parameters `speed, threshold_rms` are optional, and are by default set to `1` and `0.01` respectively.
The `threshold_rms` controls the threshold root-mean-square of the audio chunks, which are trimmed from the end of the audio, so a higher value will trim audio more generously.

After sending a valid message to the server it responds by sending back raw mono-channel PCM audio chunks using the `f32le` encoding and a sample rate of `24kHz`. After the transmission of all chunks has completed (or an error has occured), the server sends an empty message:
```
-> JSON request

<- PCM audio chunk 1
<- PCM audio chunk 2
...
<- PCM audio chunk n

<- empty message
```
## Examples 
Examples on how to communicate with the server and convert text into speech, can be found in the `src/examples` folder.