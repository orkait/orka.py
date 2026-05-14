import torch
import time

def assign(chunk, centroids, c_norm_sq):
    r_norm_sq = torch.sum(chunk * chunk, dim=1, keepdim=True)
    dists = torch.addmm(r_norm_sq + c_norm_sq, chunk, centroids.T, alpha=-2.0, beta=1.0)
    return torch.argmin(dists, dim=1)

device = "cuda"
# Using a safe chunk size
chunk_fp32 = torch.randn(4096, 8, device=device, dtype=torch.float32)
centroids_fp32 = torch.randn(65536, 8, device=device, dtype=torch.float32)
c_norm_sq_fp32 = torch.sum(centroids_fp32 * centroids_fp32, dim=1, keepdim=True).T

chunk_fp16 = chunk_fp32.to(torch.float16)
centroids_fp16 = centroids_fp32.to(torch.float16)
c_norm_sq_fp16 = c_norm_sq_fp32.to(torch.float16)

# Warmup
assign(chunk_fp32, centroids_fp32, c_norm_sq_fp32)
assign(chunk_fp16, centroids_fp16, c_norm_sq_fp16)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    assign(chunk_fp32, centroids_fp32, c_norm_sq_fp32)
torch.cuda.synchronize()
t_fp32 = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(100):
    assign(chunk_fp16, centroids_fp16, c_norm_sq_fp16)
torch.cuda.synchronize()
t_fp16 = time.perf_counter() - t0

print(f"FP32: {t_fp32:.4f}s")
print(f"FP16: {t_fp16:.4f}s")
