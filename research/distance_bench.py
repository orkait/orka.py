import torch
import time
import json

def bench_distance():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = "cuda"
    # Typical large-scale VQ scenario: 100k vectors, 64k centroids, group size 8
    n_vectors = 100_000
    k_centroids = 65536
    dim = 8
    
    vectors = torch.randn(n_vectors, dim, device=device)
    centroids = torch.randn(k_centroids, dim, device=device)
    
    print(f"Benchmarking L2 Distance: {n_vectors} vectors x {k_centroids} centroids (dim={dim})")
    
    # 1. Warmup
    torch.cdist(vectors[:1000], centroids[:1000], p=2.0)
    torch.cuda.synchronize()
    
    # 2. Benchmark torch.cdist (Original)
    # We chunk it to avoid OOM, just like orka.py did
    start_cdist = time.perf_counter()
    chunk_size = 1024
    with torch.no_grad():
        for i in range(0, n_vectors, chunk_size):
            chunk = vectors[i:i+chunk_size]
            dists = torch.cdist(chunk, centroids, p=2.0).square()
            _ = torch.argmin(dists, dim=1)
    torch.cuda.synchronize()
    end_cdist = time.perf_counter()
    t_cdist = end_cdist - start_cdist
    
    # 3. Benchmark GEMM-based (New)
    start_gemm = time.perf_counter()
    with torch.no_grad():
        c_norm_sq = torch.sum(centroids * centroids, dim=1, keepdim=True).T
        for i in range(0, n_vectors, chunk_size):
            chunk = vectors[i:i+chunk_size]
            r_norm_sq = torch.sum(chunk * chunk, dim=1, keepdim=True)
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                chunk,
                centroids.T,
                alpha=-2.0,
                beta=1.0
            )
            _ = torch.argmin(dists, dim=1)
    torch.cuda.synchronize()
    end_gemm = time.perf_counter()
    t_gemm = end_gemm - start_gemm
    
    results = {
        "cdist_time_s": t_cdist,
        "gemm_time_s": t_gemm,
        "speedup": t_cdist / t_gemm,
        "n_vectors": n_vectors,
        "k_centroids": k_centroids,
        "dim": dim
    }
    
    print(f"\nOriginal (cdist): {t_cdist:.4f}s")
    print(f"Optimized (GEMM): {t_gemm:.4f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    
    with open("speed/bench_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    bench_distance()
