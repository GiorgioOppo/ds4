/* Included once by ds4_bonsai.c. Both CPU and Metal session creation use this
 * validator; the GGUF binder additionally enforces the supported checkpoint. */
#include <math.h>

static bool bonsai_size_product(uint64_t a, uint64_t b, uint64_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool bonsai_tensor_valid(const ds4_bonsai_tensor *w, uint32_t cols,
                               uint32_t rows, bool f32_only, bool allow_signs) {
    if (!w->data || !cols || !rows || w->cols != cols || w->rows != rows) return false;
    uint64_t row_bytes, bytes;
    if (f32_only && w->type != 0) return false;
    switch (w->type) {
        case 0:
            if ((uintptr_t)w->data % _Alignof(float)) return false;
            row_bytes = (uint64_t)cols * 4u;
            break;
        case 1: case 30: row_bytes = (uint64_t)cols * 2u; break;
        case 142: case 143:
            if (cols % 128u) return false;
            row_bytes = (uint64_t)(cols / 128u) * (w->type == 142 ? 34u : 28u);
            break;
        default: return false;
    }
    if (!bonsai_size_product(row_bytes, rows, &bytes) || w->bytes != bytes) return false;
    if (w->signs) {
        if (!allow_signs || cols % 1024u || (uintptr_t)w->signs % _Alignof(int32_t)) return false;
        for (uint32_t i = 0; i < cols; i++) if (w->signs[i] != 1 && w->signs[i] != -1) return false;
    }
    return true;
}

bool ds4_bonsai_model_valid(const ds4_bonsai_model *m) {
    if (!m || !m->n_layer || m->n_layer > DS4_BONSAI_LAYERS ||
        !m->n_embd || !m->n_vocab || m->n_vocab > INT32_MAX || !m->n_ff || !m->context ||
        !m->full_interval || !m->n_head || !m->n_kv_head || m->n_head % m->n_kv_head ||
        !m->head_dim || !m->n_rot || m->n_rot > m->head_dim || m->n_rot % 2u ||
        !m->n_k_head || !m->n_v_head || m->n_v_head % m->n_k_head ||
        !m->ssm_dim || m->conv_width < 2u ||
        !isfinite(m->eps) || m->eps <= 0 || !isfinite(m->rope_base) || m->rope_base <= 0) return false;
    if (m->n_head > UINT32_MAX / 2u / m->head_dim ||
        2ull * m->n_k_head + m->n_v_head > UINT32_MAX / m->ssm_dim) return false;
    const uint64_t q_rows = 2ull * m->n_head * m->head_dim;
    const uint64_t kv_rows = (uint64_t)m->n_kv_head * m->head_dim;
    const uint64_t v_rows = (uint64_t)m->n_v_head * m->ssm_dim;
    const uint64_t qkv_rows = (2ull * m->n_k_head + m->n_v_head) * m->ssm_dim;
    /* These products also appear in uint32_t dispatch/scratch expressions. */
    if (q_rows > UINT32_MAX || kv_rows > UINT32_MAX || v_rows > UINT32_MAX || qkv_rows > UINT32_MAX) return false;
    /* Convolution/history indexing uses channel * width in uint32_t. */
    if (qkv_rows * m->conv_width > UINT32_MAX) return false;
    uint64_t count, bytes;
    if (!bonsai_size_product(v_rows, m->ssm_dim, &count) || !bonsai_size_product(count, sizeof(float), &bytes) ||
        !bonsai_size_product(qkv_rows, m->conv_width - 1u, &count) || !bonsai_size_product(count, sizeof(float), &bytes) ||
        !bonsai_size_product(kv_rows, m->context, &count) || !bonsai_size_product(count, sizeof(float), &bytes)) return false;

#define VALID(w,c,r) bonsai_tensor_valid(&(w),(uint32_t)(c),(uint32_t)(r),false,true)
#define F32(w,c,r) bonsai_tensor_valid(&(w),(uint32_t)(c),(uint32_t)(r),true,false)
    if (!VALID(m->embedding,m->n_embd,m->n_vocab) || !VALID(m->output,m->n_embd,m->n_vocab) ||
        !F32(m->output_norm,m->n_embd,1)) return false;
    for (uint32_t i = 0; i < m->n_layer; i++) {
        const ds4_bonsai_layer *l = m->layer + i;
        if (!F32(l->norm,m->n_embd,1) || !F32(l->post_norm,m->n_embd,1) ||
            !VALID(l->gate,m->n_embd,m->n_ff) || !VALID(l->up,m->n_embd,m->n_ff) ||
            !VALID(l->down,m->n_ff,m->n_embd)) return false;
        if ((i + 1u) % m->full_interval) {
            if (!VALID(l->qkv,m->n_embd,qkv_rows) || !VALID(l->z,m->n_embd,v_rows) ||
                !VALID(l->alpha,m->n_embd,m->n_v_head) || !VALID(l->beta,m->n_embd,m->n_v_head) ||
                !VALID(l->out,v_rows,m->n_embd) || !F32(l->conv,m->conv_width,qkv_rows) ||
                !F32(l->a,m->n_v_head,1) || !F32(l->dt,m->n_v_head,1) || !F32(l->ssm_norm,m->ssm_dim,1)) return false;
        } else {
            if (!VALID(l->q,m->n_embd,q_rows) || !VALID(l->k,m->n_embd,kv_rows) || !VALID(l->v,m->n_embd,kv_rows) ||
                !VALID(l->out,q_rows / 2u,m->n_embd) || !F32(l->q_norm,m->head_dim,1) || !F32(l->k_norm,m->head_dim,1)) return false;
        }
    }
#undef F32
#undef VALID
    return true;
}
