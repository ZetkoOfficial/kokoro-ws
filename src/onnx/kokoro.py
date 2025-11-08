import json, onnxruntime
import numpy as np
from phonemizer import phonemize

class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def to_tokens(self, phonems):
        return [ self.vocabulary.get(p) for p in phonems if p in self.vocabulary ]

    def to_phonems(self, text):
        text = text.strip()

        # TODO: for now we use the festival backend 
        result = filter(lambda p: p in self.vocabulary, phonemize(
            text,
            language="en-us",
            backend="espeak",
            preserve_punctuation=True,
            with_stress=True
        ))

        return "".join(result).strip()

class Kokoro:
    def __init__(self, model_path, voice_path, config_path):
        self.voice_path = voice_path

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.voice = np.load(voice_path)

        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.tokenizer = Tokenizer(self.config["vocab"])

    def _model_output(self, phonems, speed):
        tokens = np.array(
            self.tokenizer.to_tokens(phonems),
            dtype=np.int64
        )

        voice = self.voice[len(tokens)]
        tokens = [[0, *tokens, 0]]
        inputs = {
            "input_ids": tokens,
            "style": np.array(voice, dtype=np.float32),
            "speed": np.array([speed], dtype=np.int32)
        }

        output = self.onnx_session.run(None, inputs)[0]
        return output

    def tts(self, text, speed=1.0):
        #TODO batch phonems in sizes of max 500
        phonems = self.tokenizer.to_phonems(text)

        #TODO trim audio
        audio = self._model_output(phonems, speed)
        
        return audio