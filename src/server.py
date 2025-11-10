from kokoro import Kokoro
from websockets.asyncio.server import serve
import json, asyncio, signal,time 

model = Kokoro(
    model_path="/model/kokoro.onnx",
    voice_path="/model/voice.npy",
    config_path="/model/config.json"
)
async def handle_connection(ws):
    print("[ws] Client connected")
    async for message in ws:
        try:
            message = json.loads(message) 
            
            if "text" not in message:
                await ws.send(b"")
                continue

            text = message["text"]
            speed = message.get("speed", 1.0)
            threshold_rms = message.get("threshold_rms", 0.01)
            
            async for chunk in model.tts_generator_async(text, speed, threshold_rms):
                print(f"[ws] Chunk sent ({len(chunk)} bytes)")
                await ws.send(chunk)
            
            # send empty message to denote end of transmission
            await ws.send(b"")
        except Exception as e:
            print(f"[ws] Error handling message: {message}\n{e}")
            await ws.send(b"")

    print("[ws] Client closed connection.")

async def server():   
    async with serve(handle_connection, "0.0.0.0", "8888", max_size=5*1024*1024) as server:
        print("[ws] Server started on 8888")

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, server.close)
        await server.wait_closed()
        print("[ws] Server closed.")

if __name__ == "__main__":
    asyncio.run(server())