import torch
import time

def assign(chunk, centroids, c_norm_sq):
    r_norm_sq = torch.sum(chunk * chunk, dim=1, keepdim=True)
    dists = torch.addmm(r_norm_sq + c_norm_sq, chunk, centroids.T, alpha=-2.0, beta=1.0)
    return torch.argmin(dists, dim=1)

device = "cuda"
# Using a safe chunk size (4096 rows vs 65536 centroids) to prevent OOM
chunk = torch.randn(4096, 8, device=device, dtype=torch.float32)
centroids = torch.randn(65536, 8, device=device, dtype=torch.float32)
c_norm_sq = torch.sum(centroids * centroids, dim=1, keepdim=True).T

# Warmup (no TF32)
torch.backends.cuda.matmul.allow_tf32 = False
assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()
t_fp32 = time.perf_counter() - t0

# Warmup (TF32)
torch.backends.cuda.matmul.allow_tf32 = True
assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()
t_tf32 = time.perf_counter() - t0

print(f"FP32 Eager: {t_fp32:.4f}s")
print(f"TF32 Eager: {t_tf32:.4f}s")
