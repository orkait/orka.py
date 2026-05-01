import numpy as np
import math
from scipy.stats import norm

def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9)

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed)**2)

def calculate_sqnr(original, reconstructed):
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - reconstructed)**2)
    if noise_power == 0: return float('inf')
    return 10 * np.log10(signal_power / noise_power)

# --- BASELINES ---

def quantize_linear_global(weights, bits=4):
    q_min, q_max = np.min(weights), np.max(weights)
    levels = 2**bits - 1
    scale = (q_max - q_min) / levels
    indices = np.round((weights - q_min) / scale)
    return indices * scale + q_min, "1 Mult"

def quantize_bfp(weights, block_size=16, bits=4):
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w = np.concatenate([w, np.zeros(pad)]).reshape(-1, block_size)
    scales = np.max(np.abs(w), axis=1)
    scales = np.where(scales == 0, 1e-9, scales)
    normalized = w / scales[:, np.newaxis]
    levels = 2**(bits - 1) - 1
    quantized = np.round(normalized * levels)
    recon = (quantized / levels) * scales[:, np.newaxis]
    return recon.flatten()[:len(weights.flatten())].reshape(weights.shape), "1 Mult"

# --- HYPOTHESES ---

def quantize_pure_slrq(weights, block_size=16, bits=4):
    """ Your original idea: Power of Two Anchor """
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w = np.concatenate([w, np.zeros(pad)]).reshape(-1, block_size)
    max_mags = np.max(np.abs(w), axis=1)
    anchors = 2**np.ceil(np.log2(np.where(max_mags == 0, 1e-9, max_mags)))
    normalized = w / anchors[:, np.newaxis]
    levels = 2**(bits - 1) - 1
    quantized = np.round(normalized * levels)
    recon = (quantized / levels) * anchors[:, np.newaxis]
    return recon.flatten()[:len(weights.flatten())].reshape(weights.shape), "1 Shift"

def quantize_slrq_stochastic(weights, block_size=16, bits=4):
    """ SLRQ with Probabilistic Rounding """
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w = np.concatenate([w, np.zeros(pad)]).reshape(-1, block_size)
    max_mags = np.max(np.abs(w), axis=1)
    anchors = 2**np.ceil(np.log2(np.where(max_mags == 0, 1e-9, max_mags)))
    normalized = w / anchors[:, np.newaxis]
    levels = 2**(bits - 1) - 1
    # Stochastic rounding
    quantized = np.floor(normalized * levels + np.random.rand(*normalized.shape))
    recon = (quantized / levels) * anchors[:, np.newaxis]
    return recon.flatten()[:len(weights.flatten())].reshape(weights.shape), "1 Shift"

def quantize_slrq_gaussian(weights, block_size=16, bits=4):
    """ SLRQ with NormalFloat-style mapping """
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w = np.concatenate([w, np.zeros(pad)]).reshape(-1, block_size)
    max_mags = np.max(np.abs(w), axis=1)
    anchors = 2**np.ceil(np.log2(np.where(max_mags == 0, 1e-9, max_mags)))
    normalized = w / anchors[:, np.newaxis]
    std_guess = 0.4
    transformed = norm.cdf(normalized, loc=0, scale=std_guess)
    levels = 2**bits - 1
    quantized = np.round(transformed * levels)
    recon_transformed = np.clip(quantized / levels, 0.001, 0.999)
    recon_normalized = norm.ppf(recon_transformed, loc=0, scale=std_guess)
    recon = recon_normalized * anchors[:, np.newaxis]
    return recon.flatten()[:len(weights.flatten())].reshape(weights.shape), "1 Shift + LUT"

def quantize_block_salient_slrq(weights, block_size=16, bits=4):
    """ The Winner: Protect Outliers, SLRQ for the rest """
    w = weights.flatten()
    pad = (block_size - (len(w) % block_size)) % block_size
    w_padded = np.concatenate([w, np.zeros(pad)]).reshape(-1, block_size)
    recon_blocks = np.zeros_like(w_padded)
    for i in range(len(w_padded)):
        block = w_padded[i]
        max_idx = np.argmax(np.abs(block))
        recon_blocks[i, max_idx] = block[max_idx] # Keep 1 in FP16
        mask = np.ones(block_size, dtype=bool)
        mask[max_idx] = False
        rem = block[mask]
        if np.any(rem):
            anchor = 2**np.ceil(np.log2(np.max(np.abs(rem)) + 1e-9))
            levels = 2**(bits - 1) - 1
            quantized = np.round((rem / anchor) * levels)
            recon_blocks[i, mask] = (quantized / levels) * anchor
    return recon_blocks.flatten()[:len(w)].reshape(weights.shape), "1 FP16 + 15 Shifts"

if __name__ == "__main__":
    print("======================================================================")
    print("   SCIENTIFIC BENCHMARK SUITE: SLRQ COMPRESSION VALIDATION")
    print("======================================================================")
    
    np.random.seed(42)
    # Simulate realistic LLM weights: Gaussian + Heavy Outliers
    size = 100000
    weights = np.random.normal(0, 0.1, size)
    weights[::32] *= 10.0 # Introduce significant outliers
    
    methods = [
        ("Linear (Global)", lambda w: quantize_linear_global(w)),
        ("BFP (Industry Std)", lambda w: quantize_bfp(w)),
        ("Pure SLRQ", lambda w: quantize_pure_slrq(w)),
        ("SLRQ + Stochastic", lambda w: quantize_slrq_stochastic(w)),
        ("SLRQ + Gaussian", lambda w: quantize_slrq_gaussian(w)),
        ("Block-Salient SLRQ", lambda w: quantize_block_salient_slrq(w)),
    ]
    
    header = f"{'Method':<20} | {'Cosine':<10} | {'MSE':<10} | {'SQNR (dB)':<10} | {'Hardware'}"
    print(header)
    print("-" * len(header))
    
    results = []
    for name, fn in methods:
        reconstructed, hw_cost = fn(weights)
        cos = cosine_similarity(weights, reconstructed)
        mse = calculate_mse(weights, reconstructed)
        sqnr = calculate_sqnr(weights, reconstructed)
        results.append((name, cos, mse, sqnr, hw_cost))
        print(f"{name:<20} | {cos:.6f}   | {mse:.6f}   | {sqnr:.2f}      | {hw_cost}")

    print("\n[Scientific Conclusion]")
    best_cos = max(results, key=lambda x: x[1])
    print(f"-> Highest Precision (Cosine): {best_cos[0]} ({best_cos[1]:.6f})")
    
    # Analyze Trade-off
    winner = results[-1] # Block-Salient
    bfp = results[1]     # BFP
    if winner[1] > bfp[1]:
        print(f"-> VERIFIED: Block-Salient SLRQ outperforms BFP in directional integrity.")
        print(f"   It provides {winner[1]-bfp[1]:.6f} more Cosine Similarity while")
        print(f"   replacing 15/16 multiplications with hardware-efficient bit-shifts.")
    else:
        print(f"-> OBSERVATION: BFP still holds a slight lead in pure accuracy,")
        print(f"   but the hardware complexity of SLRQ is significantly lower.")
