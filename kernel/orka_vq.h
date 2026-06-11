/* ORKA_VQ dequantization kernel - CPU reference.
 *
 * Decodes one Orka-packed weight tensor (N-stage residual vector
 * quantization + slrq block scale + outlier/salient escape + low-rank
 * correction) back to float32. This is the numerical core a future
 * GGML_TYPE_ORKA_VQ would run; it is intentionally free of any llama.cpp
 * dependency so it can be unit-tested in isolation against the reference
 * Python decoder.
 *
 * Indices arrive already unpacked (bit-unpacking and any stream
 * entropy-decode are the loader's job, not the kernel's), as int32.
 * Codebooks/scales/sidecar values arrive as float32. The caller owns all
 * buffers; the kernel writes exactly `packed_values` floats into `out`.
 */
#ifndef ORKA_VQ_H
#define ORKA_VQ_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t        group_size;    /* weights per codebook vector for this stage */
    int32_t        codebook_size; /* k (rows of codebook) */
    const float   *codebook;      /* [codebook_size * group_size], row-major */
    const int32_t *indices;       /* [ceil(padded_values / group_size)] */
} orka_vq_stage;

typedef struct {
    int64_t              packed_values;   /* original element count (before group padding) */
    int64_t              padded_values;   /* element count after group padding */
    int32_t              rows;            /* shape[0] */
    int32_t              cols;            /* prod(shape[1:]) */

    int32_t              n_stages;
    const orka_vq_stage *stages;

    /* slrq-block / block-max scale: out_block[b] *= scale[b]. 0 to skip. */
    int32_t              block_scale_size;
    int32_t              scale_count;
    const float         *scales;          /* [scale_count] or NULL */

    /* Outlier escape: written in normalized space, BEFORE scaling. */
    int32_t              outlier_count;
    const int64_t       *outlier_pos;     /* [outlier_count] or NULL */
    const float         *outlier_val;     /* [outlier_count] or NULL */

    /* Salient escape (slrq): written AFTER scaling, local index within block. */
    int32_t              salient_count;
    const int32_t       *salient_idx;     /* [salient_count] local index, or NULL */
    const float         *salient_val;     /* [salient_count] or NULL */

    /* Low-rank correction: out += A @ B^T, applied last. */
    int32_t              lowrank_rank;
    const float         *lowrank_a;       /* [rows * rank] or NULL */
    const float         *lowrank_b;       /* [cols * rank] or NULL */
} orka_vq_tensor;

/* Decode `t` into `out` (length >= t->packed_values). Returns 0 on success,
 * negative on a structural error. */
int orka_vq_dequantize(const orka_vq_tensor *t, float *out);

#ifdef __cplusplus
}
#endif

#endif /* ORKA_VQ_H */
