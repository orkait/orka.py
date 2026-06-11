/* Self-contained sanity test for the ORKA_VQ kernel. Synthetic inputs with
 * hand-computed expected outputs - no artifact, no Python. Exercises the full
 * chain ordering: 2-stage sum -> outlier -> block scale -> salient -> lowrank.
 *
 * Build + run:
 *   gcc -O2 orka_vq.c test_orka_vq.c -lm -o test_orka_vq && ./test_orka_vq
 */
#include "orka_vq.h"

#include <math.h>
#include <stdio.h>

static int approx(float a, float b) { return fabsf(a - b) < 1e-5f; }

int main(void) {
    /* 1 row x 4 cols, group_size 2, 2 stages. */
    float cb0[] = {1.0f, 2.0f, 3.0f, 4.0f}; /* k=2, g=2 */
    int32_t idx0[] = {0, 1};                /* vec0 -> [1,2], vec1 -> [3,4] */
    float cb1[] = {0.1f, 0.1f, 0.2f, 0.2f};
    int32_t idx1[] = {1, 0};                /* vec0 += [0.2,0.2], vec1 += [0.1,0.1] */

    orka_vq_stage stages[2] = {
        {2, 2, cb0, idx0},
        {2, 2, cb1, idx1},
    };
    /* stage sum: [1.2, 2.2, 3.1, 4.1] */

    float scales[] = {2.0f, 10.0f}; /* block_scale_size 2 */
    int64_t out_pos[] = {0};
    float out_val[] = {5.0f};       /* outlier overwrites index 0 BEFORE scaling */
    /* slrq stores one salient per block (b_count entries). block0 local-1 ->
     * flat 1, block1 local-1 -> flat 3. Both injected AFTER scaling. */
    int32_t sal_idx[] = {1, 1};
    float sal_val[] = {7.0f, 99.0f};
    float lr_a[] = {1.0f};          /* rows=1, rank=1 */
    float lr_b[] = {0.0f, 0.0f, 1.0f, 0.0f}; /* cols=4, rank=1 -> adds 1.0 to col2 */

    orka_vq_tensor t = {0};
    t.packed_values = 4;
    t.padded_values = 4;
    t.rows = 1;
    t.cols = 4;
    t.n_stages = 2;
    t.stages = stages;
    t.block_scale_size = 2;
    t.scale_count = 2;
    t.scales = scales;
    t.outlier_count = 1;
    t.outlier_pos = out_pos;
    t.outlier_val = out_val;
    t.salient_count = 2;
    t.salient_idx = sal_idx;
    t.salient_val = sal_val;
    t.lowrank_rank = 1;
    t.lowrank_a = lr_a;
    t.lowrank_b = lr_b;

    float out[4];
    int rc = orka_vq_dequantize(&t, out);
    if (rc != 0) {
        printf("FAIL: rc=%d\n", rc);
        return 1;
    }

    /* Expected:
     *   sum      = [1.2, 2.2, 3.1, 4.1]
     *   outlier  = [5.0, 2.2, 3.1, 4.1]   (idx0 = 5.0)
     *   scale    = [10.0, 22.0, 31.0, 41.0]  (block0 *2, block1 *10)
     *   salient  = [10.0, 7.0, 31.0, 99.0]   (flat1 = 7.0, flat3 = 99.0)
     *   lowrank  = [10.0, 7.0, 32.0, 99.0]   (+1.0 at col2)
     */
    const float expected[] = {10.0f, 7.0f, 32.0f, 99.0f};
    for (int i = 0; i < 4; ++i) {
        if (!approx(out[i], expected[i])) {
            printf("FAIL: out[%d]=%f expected %f\n", i, out[i], expected[i]);
            return 1;
        }
    }
    printf("PASS: full chain ordering correct\n");

    /* Scalar-stage smoke (group_size 1 second stage). */
    float scb[] = {0.5f, -0.5f};
    int32_t sidx[] = {0, 0, 1, 1};
    orka_vq_stage st2[2] = {{2, 2, cb0, idx0}, {1, 2, scb, sidx}};
    orka_vq_tensor t2 = {0};
    t2.packed_values = 4; t2.padded_values = 4; t2.rows = 1; t2.cols = 4;
    t2.n_stages = 2; t2.stages = st2;
    float out2[4];
    if (orka_vq_dequantize(&t2, out2) != 0) { printf("FAIL: scalar rc\n"); return 1; }
    /* stage0 [1,2,3,4] + scalar [0.5,0.5,-0.5,-0.5] = [1.5,2.5,2.5,3.5] */
    const float exp2[] = {1.5f, 2.5f, 2.5f, 3.5f};
    for (int i = 0; i < 4; ++i) {
        if (!approx(out2[i], exp2[i])) { printf("FAIL: scalar out[%d]=%f\n", i, out2[i]); return 1; }
    }
    printf("PASS: scalar stage correct\n");
    return 0;
}
