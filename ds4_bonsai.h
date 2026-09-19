#ifndef DS4_BONSAI_H
#define DS4_BONSAI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Native text-only Qwen3.5 dense graph used by Prism Ternary Bonsai 2.
 * Weights and signs are borrowed from the engine and outlive all sessions. */
#define DS4_BONSAI_LAYERS 64
#define DS4_BONSAI_METAL_PREFILL_CAP 32u
typedef struct {
    const void *data;
    uint64_t bytes;
    uint32_t type, cols, rows;
    const int32_t *signs; /* NULL: no activation rotation for this weight */
} ds4_bonsai_tensor;

typedef struct {
    ds4_bonsai_tensor norm, post_norm, gate, up, down;
    ds4_bonsai_tensor q, k, v, out, q_norm, k_norm;
    ds4_bonsai_tensor qkv, z, alpha, beta, conv, a, dt, ssm_norm;
} ds4_bonsai_layer;

typedef struct {
    uint32_t n_layer, n_embd, n_vocab, n_ff, n_head, n_kv_head;
    uint32_t head_dim, n_rot, n_k_head, n_v_head, ssm_dim, conv_width;
    uint32_t full_interval, context;
    float eps, rope_base;
    bool gdn_v_grouped; /* permute V only for Hadamard-folded ssm_out */
    ds4_bonsai_tensor embedding, output, output_norm;
    ds4_bonsai_layer layer[DS4_BONSAI_LAYERS];
    int32_t *signs[3];
    uint32_t sign_widths[3];
} ds4_bonsai_model;

typedef struct ds4_bonsai_cpu ds4_bonsai_cpu;
/* Validate the independent graph API's geometry, storage and rotation tables. */
bool ds4_bonsai_model_valid(const ds4_bonsai_model *m);
ds4_bonsai_cpu *ds4_bonsai_cpu_create(const ds4_bonsai_model *m, uint32_t ctx);
void ds4_bonsai_cpu_free(ds4_bonsai_cpu *s);
void ds4_bonsai_cpu_reset(ds4_bonsai_cpu *s);
bool ds4_bonsai_cpu_eval(ds4_bonsai_cpu *s, int token, float *logits);

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
typedef struct ds4_bonsai_metal ds4_bonsai_metal;
ds4_bonsai_metal *ds4_bonsai_metal_create(const ds4_bonsai_model *m, uint32_t ctx);
void ds4_bonsai_metal_free(ds4_bonsai_metal *s);
void ds4_bonsai_metal_reset(ds4_bonsai_metal *s);
/* logits==NULL skips the final vocabulary projection during prompt ingest. */
bool ds4_bonsai_metal_eval(ds4_bonsai_metal *s, int token, float *logits);
/* A causal layer-major prompt chunk, at most DS4_BONSAI_METAL_PREFILL_CAP
 * tokens. Only the last row's logits are computed when logits is non-NULL.
 * Failed GPU execution requires reset; no partial position is committed. */
bool ds4_bonsai_metal_prefill(ds4_bonsai_metal *s, const int *tokens,
                             uint32_t count, float *logits);
#endif
#endif
