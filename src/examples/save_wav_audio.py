from onnx.kokoro import Kokoro

print("Starting example 1")
model = Kokoro(
    model_path="/model/kokoro.onnx",
    voice_path="/model/voice.npy",
    config_path="/model/config.json"
)

# 24kHz f32le
output_audio = model.tts(
"""
The old woman sat on the park bench, not watching the children play, but the single, wilting dandelion pushing up through a crack in the concrete. It was the same dandelion she had spotted last year, and the year before. Each spring, it fought its way to the sun, a tiny, tenacious miracle. Today, however, the wind had changed. It was a sharp, late-autumn wind, and the dandelion's single, stubborn stalk was bent, almost broken, against the harsh gray. She reached out a trembling, gnarled finger and gently cupped the delicate head. She felt a familiar sadness, but also a strange sense of peace. The dandelion might not survive this cold snap, but it had lived its life, and for a few fleeting moments, under her care, it was safe. She closed her eyes, and the scent of damp earth and fading flowers filled her senses, a memory of seasons past, of life and loss, a story written not in words, but in the gentle, fleeting moments of a single, stubborn flower.")
""")
                         
with open("/output/out.pcm", "wb") as f:
    f.write(output_audio)