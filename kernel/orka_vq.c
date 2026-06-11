/* ORKA_VQ dequantization kernel - CPU reference implementation.
 *
 * Mirrors orka/pipeline/decode.py:_decode_tensor exactly, including the
 * order of operations: stage sum -> outlier inject -> block scale ->
 * salient inject -> low-rank correction. Any divergence here would show up
 * as a non-zero diff against the Python decoder in the ctypes test.
 */
#include "orka_vq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int orka_vq_dequantize(const orka_vq_tensor *t, float *out) {
    if (!t || !out || t->n_stages <= 0 || !t->stages) {
        return -1;
    }

    const int64_t group_size = t->stages[0].group_size;
    if (group_size <= 0) {
        return -2;
    }
    const int64_t index_count = ceil_div(t->padded_values, group_size);
    const int64_t buf_len = index_count * group_size;

    float *acc = (float *)calloc((size_t)buf_len, sizeof(float));
    if (!acc) {
        return -3;
    }

    /* 1. Sum of stage codebook lookups. Each stage may have its own group
     *    size (scalar stages use group_size 1); every stage decodes to the
     *    same padded element count, summed elementwise. */
    for (int32_t s = 0; s < t->n_stages; ++s) {
        const orka_vq_stage *st = &t->stages[s];
        const int64_t g = st->group_size;
        if (g <= 0 || !st->codebook || !st->indices) {
            free(acc);
            return -4;
        }
        const int64_t s_index_count = ceil_div(t->padded_values, g);
        for (int64_t v = 0; v < s_index_count; ++v) {
            const int64_t idx = st->indices[v];
            const float *row = st->codebook + idx * g;
            float *dst = acc + v * g;
            for (int64_t j = 0; j < g; ++j) {
                dst[j] += row[j];
            }
        }
    }

    /* 2. Truncate to the real element count. */
    const int64_t n = t->packed_values;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = acc[i];
    }
    free(acc);

    /* 3. Outlier escape (normalized space, before scaling). */
    if (t->outlier_count > 0 && t->outlier_pos && t->outlier_val) {
        for (int32_t o = 0; o < t->outlier_count; ++o) {
            const int64_t p = t->outlier_pos[o];
            if (p >= 0 && p < n) {
                out[p] = t->outlier_val[o];
            }
        }
    }

    /* 4. slrq-block / block-max scale: each contiguous block of
     *    block_scale_size elements is multiplied by its stored scale. The
     *    final partial block (from group padding) shares the last scale,
     *    matching numpy's pad-then-reshape. */
    if (t->block_scale_size > 0 && t->scales && t->scale_count > 0) {
        const int64_t bs = t->block_scale_size;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = i / bs;
            const float scale = (b < t->scale_count) ? t->scales[b] : 1.0f;
            out[i] *= scale;
        }
    }

    /* 5. Salient escape (slrq): one weight per block re-injected AFTER
     *    scaling, addressed by local index within its block. */
    if (t->salient_count > 0 && t->salient_idx && t->salient_val) {
        const int64_t bs = (t->block_scale_size > 0) ? t->block_scale_size : 32;
        for (int32_t b = 0; b < t->salient_count; ++b) {
            const int64_t flat = (int64_t)b * bs + t->salient_idx[b];
            if (flat >= 0 && flat < n) {
                out[flat] = t->salient_val[b];
            }
        }
    }

    /* 6. Low-rank correction: out += A @ B^T (row i = A[i,:] . B[col,:]). */
    if (t->lowrank_rank > 0 && t->lowrank_a && t->lowrank_b) {
        const int64_t r = t->lowrank_rank;
        const int64_t rows = t->rows;
        const int64_t cols = t->cols;
        if (rows * cols <= n) {
            for (int64_t i = 0; i < rows; ++i) {
                const float *a_row = t->lowrank_a + i * r;
                float *out_row = out + i * cols;
                for (int64_t jc = 0; jc < cols; ++jc) {
                    const float *b_row = t->lowrank_b + jc * r;
                    float dot = 0.0f;
                    for (int64_t kk = 0; kk < r; ++kk) {
                        dot += a_row[kk] * b_row[kk];
                    }
                    out_row[jc] += dot;
                }
            }
        }
    }

    return 0;
}
