import torch
from safetensors.torch import save_file

class TensorStreamer:
    def __init__(self):
        self.count = 0
    def __iter__(self):
        for i in range(2):
            self.count += 1
            yield str(i), torch.zeros(10)
    def items(self):
        return self

streamer = TensorStreamer()
try:
    save_file(streamer, "test.safetensors")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
