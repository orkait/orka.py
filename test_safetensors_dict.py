import torch
from safetensors.torch import save_file

class LazyDict:
    def __init__(self):
        self.keys_list = ["a", "b"]
        self.count = 0
    def keys(self):
        return self.keys_list
    def __getitem__(self, key):
        self.count += 1
        return torch.zeros(10)

try:
    save_file(LazyDict(), "test.safetensors")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
