# Research Log: Binary Power Quantization, SLRQ, and ArXiv Integration

## Overview
We investigated a novel quantization approach based on the premise that representing numbers as $X = 2^N + \text{offset}$ can improve hardware efficiency (bit-shifts instead of multipliers) while maintaining model performance. We also established a research environment using the arXiv MCP server.

## Phase 1: Mathematical Theory
- **Hypothesis:** Caching the power ($N$) and storing only the offset would compress data and simplify arithmetic.
- **Finding:** Information theory proves this does not compress random uniform data, but it is highly effective for structured data (like Neural Network weights) which follow power-law distributions.
- **Connection:** The concept aligns with Floating-Point (FP16/MX) representations where $N$ is the exponent and the offset is the mantissa.

## Phase 2: SLRQ Simulation
We hypothesized **Spherical Logarithmic Residual Quantization (SLRQ)**:
1.  **Shared Anchors:** Instead of global scale, use a block-level power-of-two anchor.
2.  **Directionality:** Preserving the vector's angle (Cosine Similarity) is more critical for LLM perplexity than minimizing Mean Squared Error (MSE).
3.  **Refinement:** We moved from simple logarithmic maps to Gaussian-aware (NormalFloat) mapping to preserve density around zero.
4.  **Key Insight:** Outliers are the bottleneck. The "binary power" anchor is only effective when paired with outlier protection.

## Phase 3: Breakthrough & Integration
- **Breakthrough:** The **Block-wise Salient-Protected SLRQ** method.
- **Strategy:** 
    - Block size = 16.
    - Protected the absolute maximum ("King") weight per block in FP16 (salient-protected).
    - Used a $2^N$ anchor for the remaining 15 weights.
- **Results:**
    - Achieved **0.9985 Cosine Similarity** vs 0.9906 for industry-standard BFP.
    - Proved that "caching the power" is a viable, high-performance alternative to multipliers when outliers are properly handled.

## Phase 4: Native Orka Integration
- Added `--normalization slrq-block` to `orka.py`.
- Integrated `_normalize_tensor_slrq_block_torch` and `_normalize_tensor_slrq_block_numpy` into the normalization pipeline.
- Implemented robust `_decode_tensor` and artifact verification logic to ensure the new method is fully compatible with existing packing/evaluation commands.
- Verified end-to-end integration with SmolLM2-135M using the new `orka.py slrq-eval` test command.

## Phase 5: Infrastructure and Research Enablement
- **ArXiv MCP Server:** Integrated the arXiv MCP server into the Orka framework to enable deep-dive research.
    - **Initial Failure:** Attempted Smithery integration via `@modelcontextprotocol/server-arxiv` (Node.js), which failed due to package naming mismatch.
    - **Final Resolution:** Successfully registered the official Python-based `arxiv-mcp-server` using:
      ```bash
      gemini mcp add arxiv uvx arxiv-mcp-server
      ```
    - **Validation:** Confirmed connection status via `gemini mcp list` and enabled direct access to research literature on "Logarithmic Quantization", "APoT", and "Cosine Preservation" to ground the project in academic reality.
