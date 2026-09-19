/* Native, scalar reference graph for Prism Ternary Bonsai 2. Keep this path
 * independent of the Metal reductions so it can serve as a correctness oracle.
 * PQ2/PTQ weights remain packed; only one matrix row is expanded at a time. */
#include "ds4_bonsai.h"
#include "bonsai_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_bonsai_validate.inc"

struct ds4_bonsai_cpu {
    const ds4_bonsai_model *m;
    uint32_t ctx, pos;
    float *state[DS4_BONSAI_LAYERS], *history[DS4_BONSAI_LAYERS];
    float *keys[DS4_BONSAI_LAYERS], *values[DS4_BONSAI_LAYERS];
    float *x, *norm, *tmp, *rot, *row, *q, *k, *v, *z, *qkv;
    float *alpha, *beta, *gate, *up, *scores;
};

static size_t row_bytes(const ds4_bonsai_tensor *w) {
    if (w->type == 142) return (size_t)w->cols / 128 * 34;
    if (w->type == 143) return (size_t)w->cols / 128 * 28;
    return (size_t)w->cols * (w->type == 0 ? 4 : 2);
}

static bool unpack(const ds4_bonsai_tensor *w, uint32_t row, float *out) {
    const uint8_t *p = (const uint8_t *)w->data + row_bytes(w) * row;
    if (w->type == 142 || w->type == 143)
        return ds4_bonsai_dequantize_row(w->type, p, out, w->cols) != 0;
    if (w->type == 0) { memcpy(out, p, w->cols * sizeof(float)); return true; }
    for (uint32_t i = 0; i < w->cols; i++) {
        if (w->type == 30) {
            uint32_t bits = (uint32_t)(p[2*i] | p[2*i+1] << 8) << 16;
            memcpy(out+i, &bits, sizeof(bits));
        } else if (w->type == 1) out[i] = ds4_bonsai_load_f16_le(p+2*i);
        else return false;
    }
    return true;
}

static void matvec(ds4_bonsai_cpu *s, const ds4_bonsai_tensor *w, const float *x, float *y) {
    if (w->signs) {
        ds4_bonsai_hadamard_forward(s->rot, x, w->signs, w->cols);
        x = s->rot;
    }
    for (uint32_t r = 0; r < w->rows; r++) {
        unpack(w, r, s->row);
        double sum = 0;
        for (uint32_t c = 0; c < w->cols; c++) sum += (double)s->row[c] * x[c];
        y[r] = (float)sum;
    }
}

static void rms(float *out, const float *x, const ds4_bonsai_tensor *w, uint32_t n, float eps) {
    double sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum / n) + eps);
    const float *gamma = w->data;
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * gamma[i];
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x * sigmoid(x); }
static float softplus(float x) { return fmaxf(x, 0) + log1pf(expf(-fabsf(x))); }

static void rope(float *x, uint32_t n, uint32_t pos, float base) {
    for (uint32_t j = 0; j < n/2; j++) {
        const float angle = pos * powf(base, -2.0f * j / n);
        const float c = cosf(angle), s = sinf(angle), a = x[j], b = x[j+n/2];
        x[j] = a*c - b*s; x[j+n/2] = a*s + b*c;
    }
}

static void linear_attention(ds4_bonsai_cpu *s, uint32_t il) {
    const ds4_bonsai_model *m = s->m;
    const ds4_bonsai_layer *l = m->layer + il;
    const uint32_t D=m->ssm_dim, H=m->n_v_head, K=m->n_k_head;
    const uint32_t C=(2*K+H)*D, W=m->conv_width;
    matvec(s, &l->qkv, s->norm, s->qkv);
    matvec(s, &l->z, s->norm, s->z);
    matvec(s, &l->alpha, s->norm, s->alpha);
    matvec(s, &l->beta, s->norm, s->beta);
    const float *conv = l->conv.data;
    float *history = s->history[il];
    for (uint32_t c = 0; c < C; c++) {
        float a = s->qkv[c] * conv[c*W+W-1];
        for (uint32_t t = 0; t < W-1; t++) a += history[c*(W-1)+t] * conv[c*W+t];
        for (uint32_t t = 0; t+1 < W-1; t++) history[c*(W-1)+t] = history[c*(W-1)+t+1];
        history[c*(W-1)+W-2] = s->qkv[c];
        s->qkv[c] = silu(a);
    }
    for (uint32_t h = 0; h < 2*K; h++) {
        float *x = s->qkv+h*D;
        double sum = 0;
        for (uint32_t d = 0; d < D; d++) sum += (double)x[d]*x[d];
        float scale = 1.0f / fmaxf(sqrtf((float)sum), m->eps);
        for (uint32_t d = 0; d < D; d++) x[d] *= scale;
    }
    for (uint32_t h = 0; h < H; h++) {
        const float *q=s->qkv+(h%K)*D, *k=s->qkv+(K+h%K)*D;
        const float decay=expf(((const float *)l->a.data)[h] * softplus(s->alpha[h]+((const float *)l->dt.data)[h]));
        const float b=sigmoid(s->beta[h]);
        for (uint32_t v = 0; v < D; v++) {
            float *state=s->state[il]+((size_t)h*D+v)*D;
            double dot=0;
            for (uint32_t d = 0; d < D; d++) { state[d] *= decay; dot += (double)state[d]*k[d]; }
            const float delta=(s->qkv[2*K*D+h*D+v]-(float)dot)*b;
            double out=0;
            for (uint32_t d = 0; d < D; d++) { state[d] += delta*k[d]; out += (double)state[d]*q[d]; }
            s->v[h*D+v]=(float)out/sqrtf((float)D);
        }
        rms(s->v+h*D, s->v+h*D, &l->ssm_norm, D, m->eps);
        for (uint32_t d = 0; d < D; d++) s->v[h*D+d] *= silu(s->z[h*D+d]);
    }
    const float *input=s->v;
    if (m->gdn_v_grouped && l->out.signs) {
        for (uint32_t h = 0; h < H; h++)
            memcpy(s->z+((h%K)*(H/K)+h/K)*D, s->v+h*D, D*sizeof(float));
        input=s->z;
    }
    matvec(s, &l->out, input, s->tmp);
}

static void full_attention(ds4_bonsai_cpu *s, uint32_t il) {
    const ds4_bonsai_model *m=s->m;
    const ds4_bonsai_layer *l=m->layer+il;
    const uint32_t D=m->head_dim, H=m->n_head, K=m->n_kv_head;
    matvec(s, &l->q, s->norm, s->q);
    matvec(s, &l->k, s->norm, s->k);
    matvec(s, &l->v, s->norm, s->v);
    for (uint32_t h = 0; h < H; h++) {
        rms(s->q+2*h*D, s->q+2*h*D, &l->q_norm, D, m->eps);
        rope(s->q+2*h*D, m->n_rot, s->pos, m->rope_base);
    }
    for (uint32_t h = 0; h < K; h++) {
        rms(s->k+h*D, s->k+h*D, &l->k_norm, D, m->eps);
        rope(s->k+h*D, m->n_rot, s->pos, m->rope_base);
    }
    memcpy(s->keys[il]+(size_t)s->pos*K*D, s->k, K*D*sizeof(float));
    memcpy(s->values[il]+(size_t)s->pos*K*D, s->v, K*D*sizeof(float));
    for (uint32_t h = 0; h < H; h++) {
        const uint32_t kh=h/(H/K);
        float max=-INFINITY;
        for (uint32_t t = 0; t <= s->pos; t++) {
            const float *k=s->keys[il]+((size_t)t*K+kh)*D;
            double dot=0;
            for (uint32_t d = 0; d < D; d++) dot += (double)k[d]*s->q[2*h*D+d];
            s->scores[t]=(float)dot/sqrtf((float)D);
            max=fmaxf(max, s->scores[t]);
        }
        double sum=0;
        for (uint32_t t = 0; t <= s->pos; t++) { s->scores[t]=expf(s->scores[t]-max); sum += s->scores[t]; }
        for (uint32_t d = 0; d < D; d++) {
            double val=0;
            for (uint32_t t = 0; t <= s->pos; t++) val += s->scores[t]*(double)s->values[il][((size_t)t*K+kh)*D+d];
            s->z[h*D+d]=(float)(val/sum)*sigmoid(s->q[(2*h+1)*D+d]);
        }
    }
    matvec(s, &l->out, s->z, s->tmp);
}

ds4_bonsai_cpu *ds4_bonsai_cpu_create(const ds4_bonsai_model *m, uint32_t ctx) {
    if (!ds4_bonsai_model_valid(m) || !ctx || ctx > m->context) return NULL;
    ds4_bonsai_cpu *s=calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m=m; s->ctx=ctx;
    uint32_t work=m->n_ff;
    if (work < 2*m->n_head*m->head_dim) work=2*m->n_head*m->head_dim;
    if (work < (2*m->n_k_head+m->n_v_head)*m->ssm_dim) work=(2*m->n_k_head+m->n_v_head)*m->ssm_dim;
    if (work < m->n_embd) work=m->n_embd;
#define ALLOC(name,n) do { s->name=calloc((size_t)(n),sizeof(float)); if (!s->name) goto fail; } while(0)
    ALLOC(x,m->n_embd); ALLOC(norm,m->n_embd); ALLOC(tmp,m->n_embd);
    ALLOC(rot,work); ALLOC(row,work); ALLOC(q,work); ALLOC(k,work);
    ALLOC(v,work); ALLOC(z,work); ALLOC(qkv,work);
    ALLOC(alpha,m->n_v_head); ALLOC(beta,m->n_v_head);
    ALLOC(gate,m->n_ff); ALLOC(up,m->n_ff); ALLOC(scores,ctx);
    for (uint32_t l=0; l<m->n_layer; l++) {
        if ((l+1)%m->full_interval) {
            ALLOC(state[l],(size_t)m->n_v_head*m->ssm_dim*m->ssm_dim);
            ALLOC(history[l],(size_t)(2*m->n_k_head+m->n_v_head)*m->ssm_dim*(m->conv_width-1));
        } else {
            ALLOC(keys[l],(size_t)ctx*m->n_kv_head*m->head_dim);
            ALLOC(values[l],(size_t)ctx*m->n_kv_head*m->head_dim);
        }
    }
#undef ALLOC
    return s;
fail:
    ds4_bonsai_cpu_free(s); return NULL;
}

void ds4_bonsai_cpu_free(ds4_bonsai_cpu *s) {
    if (!s) return;
    for (uint32_t l=0;l<DS4_BONSAI_LAYERS;l++) {
        free(s->state[l]); free(s->history[l]); free(s->keys[l]); free(s->values[l]);
    }
    free(s->x); free(s->norm); free(s->tmp); free(s->rot); free(s->row);
    free(s->q); free(s->k); free(s->v); free(s->z); free(s->qkv);
    free(s->alpha); free(s->beta); free(s->gate); free(s->up); free(s->scores); free(s);
}

void ds4_bonsai_cpu_reset(ds4_bonsai_cpu *s) {
    const ds4_bonsai_model *m=s->m;
    s->pos=0;
    for (uint32_t l=0;l<m->n_layer;l++) if (s->state[l]) {
        memset(s->state[l],0,(size_t)m->n_v_head*m->ssm_dim*m->ssm_dim*sizeof(float));
        memset(s->history[l],0,(size_t)(2*m->n_k_head+m->n_v_head)*m->ssm_dim*(m->conv_width-1)*sizeof(float));
    }
}

bool ds4_bonsai_cpu_eval(ds4_bonsai_cpu *s, int token, float *logits) {
    if (!s || token<0 || (uint32_t)token>=s->m->n_vocab || s->pos>=s->ctx) return false;
    const ds4_bonsai_model *m=s->m;
    if (!unpack(&m->embedding,(uint32_t)token,s->x)) return false;
    if (m->embedding.signs) ds4_bonsai_hadamard_inverse(s->x,s->x,m->embedding.signs,m->n_embd);
    for (uint32_t il=0;il<m->n_layer;il++) {
        const ds4_bonsai_layer *l=m->layer+il;
        rms(s->norm,s->x,&l->norm,m->n_embd,m->eps);
        if ((il+1)%m->full_interval) linear_attention(s,il); else full_attention(s,il);
        for (uint32_t i=0;i<m->n_embd;i++) s->x[i] += s->tmp[i];
        rms(s->norm,s->x,&l->post_norm,m->n_embd,m->eps);
        matvec(s,&l->gate,s->norm,s->gate);
        matvec(s,&l->up,s->norm,s->up);
        for (uint32_t i=0;i<m->n_ff;i++) s->gate[i]=silu(s->gate[i])*s->up[i];
        matvec(s,&l->down,s->gate,s->tmp);
        for (uint32_t i=0;i<m->n_embd;i++) s->x[i] += s->tmp[i];
    }
    if (logits) {
        rms(s->norm,s->x,&m->output_norm,m->n_embd,m->eps);
        matvec(s,&m->output,s->norm,logits);
        for (uint32_t i=0;i<m->n_vocab;i++) if (!isfinite(logits[i])) return false;
    }
    s->pos++;
    return true;
}
