import json, onnxruntime
import numpy as np
from phonemizer import phonemize
import re, asyncio


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

    def _trim(self, audio, chunk_length, threshold_rms):
        chunk_count = len(audio) // chunk_length
        
        chunks = audio[:chunk_count * chunk_length].reshape(chunk_count, chunk_length)
        rms = np.sqrt(np.mean(np.square(chunks), axis=1))
        print(rms)

        is_noisy = (rms >= threshold_rms)[::-1]
        if not np.any(is_noisy):
            return np.array([], dtype=np.float32)
        
        end_chunk = chunk_count - np.argmax(is_noisy)

        return audio[:end_chunk*chunk_length]
        

    def _to_batch(self, phonems):
        BATCH_SIZE = 300

        batches = []
        cur_batch = ""

        # preferably split at punctutation
        # TODO: support for sentences without punctuuation
        for part in re.split(r"([!.?,])", phonems):
            if len(part) > BATCH_SIZE:
                print("Splitting by spaces as sentence is too long. Consider adding punctuation.")
                for subpart in re.split(r" ", part):
                    if len(cur_batch) + len(subpart) <= BATCH_SIZE:
                        cur_batch += subpart
                        continue

                    batches.append(cur_batch)
                    cur_batch = subpart
                
                continue

            if len(cur_batch) + len(part) <= BATCH_SIZE:
                cur_batch += part
                continue
            
            batches.append(cur_batch)
            cur_batch = part

        if len(cur_batch) > 0:
            batches.append(cur_batch)

        return batches
    
    def tts_generator(self, text, speed=1.0, threshold_rms=0.01):
        phonems = self.tokenizer.to_phonems(text)

        for phonems in self._to_batch(phonems):
            audio = self._model_output(phonems, speed)
            audio = self._trim(
                audio,
                chunk_length=2048,
                threshold_rms=threshold_rms
            )

            yield audio.tobytes()
        
        yield None

    async def tts_generator_async(self, text, speed=1.0, threshold_rms=0.01):
        generator = self.tts_generator(text, speed, threshold_rms)
        while True:
            chunk = await asyncio.to_thread(next, generator)
            if chunk is None: break
            yield chunk

    def tts(self, text, speed=1.0, threshold_rms=0.01):   
        data = bytearray()
        for chunk in self.tts_generator(text, speed, threshold_rms):
            if chunk is None: break

            data.extend(chunk)

        return data