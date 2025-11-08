import json, onnxruntime
import numpy as np
from phonemizer import phonemize
import re

class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def to_tokens(self, phonems):
        return [ self.vocabulary.get(p) for p in phonems if p in self.vocabulary ]

    def to_phonems(self, text):
        text = text.strip()

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

        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("ONNXRunner used provider:", self.onnx_session.get_providers())

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

    def _to_batch(self, phonems):
        BATCH_SIZE = 300

        batches = []
        cur_batch = ""

        # preferably split at ending punctutation
        # TODO: support for sentences without punctuuation
        for part in re.split(r"([!.?])", phonems):

            if len(part) > BATCH_SIZE:
                print("Sentence too long. Consider adding ending punctuation.")
                return []

            if len(cur_batch) + len(part) <= BATCH_SIZE:
                cur_batch += part
                continue
            
            batches.append(cur_batch)
            cur_batch = part

        if len(cur_batch) > 0:
            batches.append(cur_batch)

        return batches

    def tts(self, text, speed=1.0):
        phonems = self.tokenizer.to_phonems(text)        
        data = bytearray()

        for phonems in self._to_batch(phonems):
            
            #TODO trim audio end
            audio = self._model_output(phonems, speed)
            data.extend(audio)

        return data