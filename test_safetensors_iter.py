import numpy as np
from safetensors.numpy import save_file

class TensorStreamer:
    def __init__(self):
        self.count = 0
    def __iter__(self):
        for i in range(2):
            self.count += 1
            yield str(i), np.zeros(10, dtype=np.float32)

streamer = TensorStreamer()
try:
    save_file(streamer, "test.safetensors")
    print("SUCCESS:", streamer.count)
except Exception as e:
    print(f"FAILED: {e}")
