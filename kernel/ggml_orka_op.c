/* Proof that an Orka per-tensor codebook decode runs inside ggml's compute
 * backend and produces a correct matmul - the load-bearing unknown of a
 * GGML_TYPE_ORKA_VQ integration.
 *
 * Layout: the entire compressed weight (all stage codebooks + indices +
 * scales + sidecars) is packed into ONE f32 ggml tensor `blob`, exactly as a
 * GGUF ORKA_VQ weight would carry its data in a backend buffer. A custom op
 * (ggml_map_custom2) reads `blob` + the activation tensor, decodes the weight
 * with the reference kernel, and does the GEMM - all dispatched by ggml's
 * CPU backend threads.
 *
 * Build: see kernel/Makefile target `ggml-op`. Run with a dumped tensor:
 *   ./ggml_orka_op /tmp/orka_ggml/t
 */
#include "ggml.h"
#include "ggml-cpu.h"

#include "orka_vq.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal JSON scalar extraction - the meta file is machine-generated and
 * flat, so a key search to the next number is sufficient (no nesting except
 * the stages array, handled separately). */
static long meta_long(const char *json, const char *key, long dflt) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *p = strstr(json, pat);
    if (!p) return dflt;
    p += strlen(pat);
    while (*p == ' ' || *p == '\n' || *p == '\t') p++;
    return strtol(p, NULL, 10);
}

typedef struct {
    char  *meta;
    float *blob;
    long   rows, cols;
} orka_userdata;

/* Build an orka_vq_tensor whose pointers index into the f32 blob, then decode
 * into `w` (rows*cols floats). */
static void decode_from_blob(const char *meta, const float *blob, float *w,
                             long rows, long cols) {
    orka_vq_tensor t = {0};
    t.packed_values = meta_long(meta, "packed_values", rows * cols);
    t.padded_values = meta_long(meta, "padded_values", rows * cols);
    t.rows = (int)rows;
    t.cols = (int)cols;

    int n_stages = 0;
    /* Count + parse stages from the "stages":[ ... ] array. */
    const char *sp = strstr(meta, "\"stages\":");
    orka_vq_stage stages[8];
    if (sp) {
        const char *cur = sp;
        while ((cur = strstr(cur, "\"group_size\":")) != NULL) {
            const char *obj = cur;
            long g   = meta_long(obj, "group_size", 0);
            long k   = meta_long(obj, "codebook_size", 0);
            long cbo = meta_long(obj, "cb_off", 0);
            long io  = meta_long(obj, "idx_off", 0);
            stages[n_stages].group_size    = (int)g;
            stages[n_stages].codebook_size = (int)k;
            stages[n_stages].codebook      = blob + cbo;
            /* indices stored as float in the blob; copy to int32 scratch. */
            long icount = meta_long(obj, "idx_count", 0);
            int32_t *idx = (int32_t *)malloc((size_t)icount * sizeof(int32_t));
            for (long i = 0; i < icount; ++i) idx[i] = (int32_t)lrintf(blob[io + i]);
            stages[n_stages].indices = idx;
            n_stages++;
            cur = obj + 13;
            if (n_stages >= 8) break;
        }
    }
    t.n_stages = n_stages;
    t.stages = stages;

    t.block_scale_size = (int)meta_long(meta, "block_scale_size", 0);
    if (t.block_scale_size > 0) {
        t.scale_count = (int)meta_long(meta, "scale_count", 0);
        t.scales = blob + meta_long(meta, "scale_off", 0);
    }
    int oc = (int)meta_long(meta, "outlier_count", 0);
    int64_t *opos = NULL;
    if (oc > 0) {
        long po = meta_long(meta, "outlier_pos_off", 0);
        opos = (int64_t *)malloc((size_t)oc * sizeof(int64_t));
        for (int i = 0; i < oc; ++i) {
            /* positions stored as int32 bit pattern (exact past 2^24) */
            int32_t v;
            memcpy(&v, &blob[po + i], sizeof(int32_t));
            opos[i] = (int64_t)v;
        }
        t.outlier_count = oc;
        t.outlier_pos = opos;
        t.outlier_val = blob + meta_long(meta, "outlier_val_off", 0);
    }
    int sc = (int)meta_long(meta, "salient_count", 0);
    int32_t *sidx = NULL;
    if (sc > 0) {
        long io = meta_long(meta, "salient_idx_off", 0);
        sidx = (int32_t *)malloc((size_t)sc * sizeof(int32_t));
        for (int i = 0; i < sc; ++i) sidx[i] = (int32_t)lrintf(blob[io + i]);
        t.salient_count = sc;
        t.salient_idx = sidx;
        t.salient_val = blob + meta_long(meta, "salient_val_off", 0);
    }
    int rank = (int)meta_long(meta, "lowrank_rank", 0);
    if (rank > 0) {
        t.lowrank_rank = rank;
        t.lowrank_a = blob + meta_long(meta, "lowrank_a_off", 0);
        t.lowrank_b = blob + meta_long(meta, "lowrank_b_off", 0);
    }

    orka_vq_dequantize(&t, w);

    for (int s = 0; s < n_stages; ++s) free((void *)stages[s].indices);
    free(opos);
    free(sidx);
}

/* Custom ggml op: dst[rows, n] = decode(blob) @ x[cols, n].
 * a = blob (1-D f32), b = activations [cols, n]. Single-threaded for the POC
 * (correctness over throughput); the real GGML type would split the GEMM. */
static void orka_matmul_op(struct ggml_tensor *dst, const struct ggml_tensor *a,
                           const struct ggml_tensor *b, int ith, int nth, void *userdata) {
    (void)nth;
    if (ith != 0) return;  /* one thread does the whole op */
    orka_userdata *ud = (orka_userdata *)userdata;
    const long rows = ud->rows, cols = ud->cols;
    const long n = b->ne[1];

    float *w = (float *)malloc((size_t)rows * cols * sizeof(float));
    decode_from_blob(ud->meta, (const float *)a->data, w, rows, cols);

    const float *x = (const float *)b->data;  /* [cols, n], col-major: x[j*cols + c] */
    float *y = (float *)dst->data;             /* flat row-major [rows, n] to match numpy */
    for (long r = 0; r < rows; ++r) {
        const float *wr = w + r * cols;
        for (long j = 0; j < n; ++j) {
            const float *xj = x + j * cols;
            float acc = 0.0f;
            for (long c = 0; c < cols; ++c) acc += wr[c] * xj[c];
            y[r * n + j] = acc;
        }
    }
    free(w);
}

static float *read_file_f32(const char *path, long *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = (float *)malloc(bytes);
    if (fread(buf, 1, bytes, f) != (size_t)bytes) { exit(1); }
    fclose(f);
    *count = bytes / 4;
    return buf;
}

static char *read_file_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(bytes + 1);
    if (fread(buf, 1, bytes, f) != (size_t)bytes) { exit(1); }
    buf[bytes] = 0;
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *prefix = argc > 1 ? argv[1] : "/tmp/orka_ggml/t";
    char path[1024];

    snprintf(path, sizeof(path), "%s.meta.json", prefix);
    char *meta = read_file_text(path);
    long rows = meta_long(meta, "rows", 0);
    long cols = meta_long(meta, "cols", 0);
    long x_cols = meta_long(meta, "x_cols", 0);

    long blob_len, x_len, yref_len;
    snprintf(path, sizeof(path), "%s.blob", prefix);
    float *blob = read_file_f32(path, &blob_len);
    snprintf(path, sizeof(path), "%s.x", prefix);
    float *x = read_file_f32(path, &x_len);
    snprintf(path, sizeof(path), "%s.yref", prefix);
    float *yref = read_file_f32(path, &yref_len);

    /* dst inherits t_blob's length, so the arena holds ~2 blobs + x + slack. */
    struct ggml_init_params p = { (size_t)128 * 1024 * 1024 + (size_t)blob_len * 4 * 3, NULL, false };
    struct ggml_context *ctx = ggml_init(p);

    struct ggml_tensor *t_blob = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blob_len);
    memcpy(t_blob->data, blob, (size_t)blob_len * 4);
    struct ggml_tensor *t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, x_cols);
    memcpy(t_x->data, x, (size_t)x_len * 4);

    orka_userdata ud = { meta, blob, rows, cols };
    /* Custom op reads the compressed weight from t_blob (a real ggml tensor in
     * the context buffer) + activations from t_x. dst inherits t_blob's 1-D
     * length, which exceeds rows*x_cols, so the op fills it row-major. */
    struct ggml_tensor *t_y = ggml_map_custom2(ctx, t_blob, t_x, orka_matmul_op, 1, &ud);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float *y = (const float *)t_y->data;
    double max_abs = 0.0, ref_max = 0.0;
    for (long i = 0; i < rows * x_cols; ++i) {
        double d = fabs((double)y[i] - (double)yref[i]);
        if (d > max_abs) max_abs = d;
        if (fabs(yref[i]) > ref_max) ref_max = fabs(yref[i]);
    }
    double rel = max_abs / (ref_max > 0 ? ref_max : 1.0);
    printf("rows=%ld cols=%ld x_cols=%ld blob_floats=%ld\n", rows, cols, x_cols, blob_len);
    printf("ggml-backend matmul vs numpy reference: max_abs=%.3e rel=%.3e -> %s\n",
           max_abs, rel, rel < 1e-4 ? "MATCH" : "MISMATCH");

    ggml_free(ctx);
    free(meta); free(blob); free(x); free(yref);
    return rel < 1e-4 ? 0 : 1;
}
