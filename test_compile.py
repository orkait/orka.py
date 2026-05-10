import torch
import time

def assign(chunk, centroids, c_norm_sq):
    r_norm_sq = torch.sum(chunk * chunk, dim=1, keepdim=True)
    dists = torch.addmm(r_norm_sq + c_norm_sq, chunk, centroids.T, alpha=-2.0, beta=1.0)
    return torch.argmin(dists, dim=1)

device = "cuda"
chunk = torch.randn(65536, 8, device=device, dtype=torch.float16)
centroids = torch.randn(65536, 8, device=device, dtype=torch.float16)
c_norm_sq = torch.sum(centroids * centroids, dim=1, keepdim=True).T

# Warmup
assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()
t_eager = time.perf_counter() - t0

compiled_assign = torch.compile(assign, mode="reduce-overhead")
# Compile warmup
compiled_assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    compiled_assign(chunk, centroids, c_norm_sq)
torch.cuda.synchronize()
t_compiled = time.perf_counter() - t0

print(f"Eager: {t_eager:.4f}s")
print(f"Compiled: {t_compiled:.4f}s")
