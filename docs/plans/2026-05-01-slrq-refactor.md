# Implementation Plan: Salient-Protected SLRQ Refactor

## Scope
Refactor `_normalize_tensor_slrq_block_torch` to implement "Block-wise Salient-Protected SLRQ" and update `_decode_tensor` to support re-injection of salient weights.

## Tasks

### Task 1: Refactor Encoding Logic
**Files:**
- Modify: `orka.py` (update `_normalize_tensor_slrq_block_torch`)

- [ ] **Step 1: Implement Salient Protection**
  - Update `_normalize_tensor_slrq_block_torch`:
    - Identify max absolute value per block.
    - Store salient weight and index (relative position in block).
    - Anchor remaining values to $2^N$.
    - Return structure including `salient_weights` and `salient_indices`.

### Task 2: Refactor Decoding Logic
**Files:**
- Modify: `orka.py` (update `_decode_tensor`)

- [ ] **Step 1: Implement Salient Re-injection**
  - Update `_decode_tensor`:
    - After initial dequantization, extract salient data from manifest/artifact.
    - Add logic to re-inject original precision values at their stored indices.

### Task 3: Verify Integration
**Files:**
- Test: `orka_test.py` (add test case)

- [ ] **Step 1: Add unit test for SLRQ round-trip**
  - Add test case verifying that `normalize` -> `denormalize` (with salient weights) preserves original values within expected tolerances.

### Task 4: Commit
**Files:**
- Commit all changes.
