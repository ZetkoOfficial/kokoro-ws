from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX
import torch, os
import numpy as np


# from: https://github.com/hexgrad/kokoro/blob/main/examples/export.py
def export_model():
  kmodel = KModel(disable_complex=True)
  model = KModelForONNX(kmodel).eval()

  input_ids = torch.randint(1, 100, (48,)).numpy()
  input_ids = torch.LongTensor([[0, *input_ids, 0]])
  style = torch.randn(1, 256)
  speed = torch.randint(1, 10, (1,)).int()

  torch.onnx.export(
      model,
      args = (input_ids, style, speed),
      f = "/model/kokoro.onnx",
      export_params = True,
      verbose = False,
      input_names = [ 'input_ids', 'style', 'speed' ],
      output_names = [ 'waveform', 'duration' ],
      opset_version = 17,
      dynamic_axes = {
          'input_ids': {0: "batch_size", 1: 'input_ids_len' },
          'style': {0: "batch_size"},
          "speed": {0: "batch_size"}
      },
      do_constant_folding = True,
  )

def export_voice():
  voice_numpy = torch.load("voice.pt").cpu().numpy()
  np.save("/model/voice.npy", voice_numpy)

if os.environ.get("DOWNLOAD_MODEL", "true") == "true":
  print("Creating .onnx file...")
  export_model()

if os.environ.get("DOWNLOAD_VOICE", "true") == "true":
  print("Creating voice .npy file...")
  export_voice()

print("Done!")