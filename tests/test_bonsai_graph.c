/* Model-free graph validation:
 * cc -std=c11 -O2 -Wall -Wextra -Werror -DDS4_NO_GPU -I. \
 *   tests/test_bonsai_graph.c ds4_bonsai.c -lm -o /tmp/test_bonsai_graph
 * Define DS4_BONSAI_TEST_METAL and link the Metal backend to also compare it
 * with this scalar CPU graph. No GGUF or running model server is needed. */
#include "ds4_bonsai.h"
#include "bonsai_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { EMB = 32, FF = 64, VOCAB = 32, CTX = 8, LAYERS = 4 };
typedef struct {
    ds4_bonsai_model m;
    void *allocation[128];
    size_t bytes[128], count;
} fixture;

static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "Bonsai graph: %s failed\n", what); exit(1); }
}
static void test_score_capacity(void) {
    need(ds4_bonsai_metal_score_capacity(0,24)==0 &&
         ds4_bonsai_metal_score_capacity(4096,0)==0,"score workspace rejects zero dimensions");
    need(ds4_bonsai_metal_score_capacity(1,24)==1 &&
         ds4_bonsai_metal_score_capacity(3,24)==3 &&
         ds4_bonsai_metal_score_capacity(4096,24)==8,"score workspace bounded by available queries");
    // Eight rows fit immediately below this boundary, but only seven above it.
    need(ds4_bonsai_metal_score_capacity(43690,24)==8 &&
         ds4_bonsai_metal_score_capacity(43691,24)==7,"score workspace 32 MiB boundary");
    need(ds4_bonsai_metal_score_capacity(300000,4)==6 &&
         ds4_bonsai_metal_score_capacity(262144,24)==1,"long-context score workspace capacity");
    need(ds4_bonsai_metal_score_capacity(UINT32_MAX,UINT32_MAX)==1,
         "oversized single-query workspace retains one query without overflow");
}
static uint32_t rng = 0x512090u;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static void *own(fixture *f, size_t bytes) {
    need(f->count < 128, "fixture allocation capacity");
    void *p = calloc(1, bytes); need(p != NULL, "fixture allocation");
    f->allocation[f->count] = p; f->bytes[f->count++] = bytes;
    return p;
}
static ds4_bonsai_tensor tensor(fixture *f, uint32_t cols, uint32_t rows, float scale) {
    const size_t n = (size_t)cols * rows;
    float *data = own(f, n * sizeof(float));
    for (size_t i = 0; i < n; i++) data[i] = ((int)(random_u32() % 31) - 15) * scale;
    return (ds4_bonsai_tensor){.data=data,.bytes=n*sizeof(float),.type=0,.cols=cols,.rows=rows};
}
static ds4_bonsai_tensor norm(fixture *f, uint32_t n) {
    ds4_bonsai_tensor t = tensor(f, n, 1, 1.f/1024.f);
    for (uint32_t i = 0; i < n; i++) ((float *)t.data)[i] += 1.f;
    return t;
}
static void fixture_init_width(fixture *f, uint32_t emb, uint32_t d, uint32_t h, uint32_t k) {
    memset(f, 0, sizeof(*f)); rng = 0x512090u;
    f->m = (ds4_bonsai_model){.n_layer=LAYERS,.n_embd=emb,.n_vocab=VOCAB,.n_ff=FF,
        .n_head=4,.n_kv_head=2,.head_dim=8,.n_rot=4,.n_k_head=k,.n_v_head=h,.ssm_dim=d,
        .conv_width=4,.full_interval=4,.context=CTX,.eps=1e-6f,.rope_base=10000000.f};
    ds4_bonsai_model *m = &f->m;
    m->embedding = tensor(f, emb, VOCAB, 1.f/32.f);
    m->output = tensor(f, emb, VOCAB, 1.f/128.f);
    m->output_norm = norm(f, emb);
    for (uint32_t i = 0; i < LAYERS; i++) {
        ds4_bonsai_layer *l = m->layer + i;
        l->norm = norm(f, emb); l->post_norm = norm(f, emb);
        l->gate = tensor(f, emb, FF, 1.f/512.f);
        l->up = tensor(f, emb, FF, 1.f/256.f);
        l->down = tensor(f, FF, emb, 1.f/512.f);
        if ((i + 1) % 4) {
            const uint32_t qkv = (2*k+h)*d;
            l->qkv = tensor(f, emb, qkv, 1.f/128.f);
            l->z = tensor(f, emb, h*d, 1.f/128.f);
            l->alpha = tensor(f, emb, h, 1.f/256.f);
            l->beta = tensor(f, emb, h, 1.f/256.f);
            l->out = tensor(f, h*d, emb, 1.f/512.f);
            l->conv = tensor(f, 4, qkv, 1.f/64.f);
            l->a = tensor(f, h, 1, 0);
            l->dt = tensor(f, h, 1, 1.f/64.f);
            l->ssm_norm = norm(f, d);
            for (uint32_t j = 0; j < h; j++) ((float *)l->a.data)[j] = -0.125f * (j + 1);
            for (uint32_t j = 0; j < qkv; j++) ((float *)l->conv.data)[4*j+3] += 1.f;
        } else {
            l->q = tensor(f, emb, 2*4*8, 1.f/128.f);
            l->k = tensor(f, emb, 2*8, 1.f/128.f);
            l->v = tensor(f, emb, 2*8, 1.f/128.f);
            l->out = tensor(f, 4*8, emb, 1.f/256.f);
            l->q_norm = norm(f, 8); l->k_norm = norm(f, 8);
        }
    }
    need(ds4_bonsai_model_valid(m), "well-formed synthetic graph");
}
static void fixture_init(fixture *f, uint32_t d, uint32_t h, uint32_t k) {
    fixture_init_width(f, EMB, d, h, k);
}
static void fixture_free(fixture *f) {
    for (size_t i = 0; i < f->count; i++) free(f->allocation[i]);
}
static uint64_t fingerprint(const fixture *f) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t b = 0; b < f->count; b++) {
        const uint8_t *p = f->allocation[b];
        for (size_t i = 0; i < f->bytes[b]; i++) { hash ^= p[i]; hash *= 1099511628211ull; }
    }
    return hash;
}
static float difference(const float *a, const float *b, size_t n) {
    float maximum = 0;
    for (size_t i = 0; i < n; i++) maximum = fmaxf(maximum, fabsf(a[i] - b[i]));
    return maximum;
}
static void close_logits(const float *actual, const float *expected, size_t n,
                         float atol, float rtol, const char *what) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            fabsf(actual[i] - expected[i]) > atol + rtol*fabsf(expected[i])) {
            fprintf(stderr, "Bonsai graph: %s at %zu: %.9g != %.9g\n", what, i, actual[i], expected[i]);
            exit(1);
        }
    }
}
static void run_cpu(const ds4_bonsai_model *m, const int *tokens, size_t n, float *logits) {
    ds4_bonsai_cpu *s = ds4_bonsai_cpu_create(m, CTX);
    need(s != NULL, "CPU graph create");
    for (size_t t = 0; t < n; t++) {
        for (unsigned v = 0; v < VOCAB; v++) logits[t*VOCAB+v] = NAN;
        need(ds4_bonsai_cpu_eval(s, tokens[t], logits+t*VOCAB), "CPU autoregressive eval");
        for (unsigned v = 0; v < VOCAB; v++) need(isfinite(logits[t*VOCAB+v]), "CPU writes finite logits");
    }
    ds4_bonsai_cpu_free(s);
}
#ifdef DS4_BONSAI_TEST_METAL
static void check_metal(const ds4_bonsai_model *m, const int *tokens, size_t n, const float *expected) {
    ds4_bonsai_metal *s = ds4_bonsai_metal_create(m, CTX);
    need(s != NULL, "Metal graph create");
    float logits[VOCAB], serial[CTX*VOCAB];
    for (unsigned repeat = 0; repeat < 2; repeat++) {
        ds4_bonsai_metal_reset(s);
        for (size_t t = 0; t < n; t++) {
            for (unsigned v = 0; v < VOCAB; v++) logits[v] = NAN;
            need(ds4_bonsai_metal_eval(s, tokens[t], logits), "Metal autoregressive eval");
            memcpy(serial+t*VOCAB,logits,sizeof(logits));
            close_logits(logits, expected+t*VOCAB, VOCAB, 3e-5f, 3e-4f, "Metal/CPU logits and reset");
        }
    }
    ds4_bonsai_metal_reset(s);
    for (size_t t = 0; t < n; t++)
        need(ds4_bonsai_metal_eval(s, tokens[t], t + 1 == n ? logits : NULL), "Metal prompt ingest without logits");
    close_logits(logits, expected+(n-1)*VOCAB, VOCAB, 3e-5f, 3e-4f, "Metal skipped vocabulary projections");
    // Layer-major execution must preserve both the frontier and recurrent
    // state for every split, including a prefix followed by serial decode.
    for (size_t split=1;split<=n;++split) {
        ds4_bonsai_metal_reset(s);
        need(ds4_bonsai_metal_prefill(s,tokens,(uint32_t)split,logits),"Metal batched prefix");
        need(!memcmp(logits,serial+(split-1)*VOCAB,sizeof(logits)),"batch/serial frontier bit parity");
        for (size_t t=split;t<n;++t) {
            need(ds4_bonsai_metal_eval(s,tokens[t],logits),"decode after batched prefix");
            need(!memcmp(logits,serial+t*VOCAB,sizeof(logits)),"batch/serial recurrent state bit parity");
        }
    }
    ds4_bonsai_metal_free(s);
}

static void test_prefill_chunks(void) {
    fixture f; fixture_init(&f,8,6,2);
    enum { LONG_CTX=259 };
    f.m.context=LONG_CTX;
    int tokens[LONG_CTX]; float serial[LONG_CTX*VOCAB],logits[VOCAB];
    for (unsigned i=0;i<LONG_CTX;++i) tokens[i]=(int)((i*7+3)%VOCAB);
    ds4_bonsai_metal *s=ds4_bonsai_metal_create(&f.m,LONG_CTX);
    need(s!=NULL,"long synthetic Metal context");
    for (unsigned i=0;i<LONG_CTX;++i)
        need(ds4_bonsai_metal_eval(s,tokens[i],serial+i*VOCAB),"serial long synthetic sequence");
    const uint32_t chunks[]={1,3,4,5,31,32,33,63,64,65,127,128};
    for (unsigned c=0;c<sizeof(chunks)/sizeof(chunks[0]);++c) {
        ds4_bonsai_metal_reset(s);
        unsigned pos=0;
        while(pos<LONG_CTX-2) {
            unsigned n=chunks[c]; if(n>LONG_CTX-2-pos) n=LONG_CTX-2-pos;
            need(ds4_bonsai_metal_prefill(s,tokens+pos,n,logits),"multiple prefill chunks");
            pos+=n;
            need(!memcmp(logits,serial+(pos-1)*VOCAB,sizeof(logits)),"chunk frontier bit parity");
        }
        float saved_batch[VOCAB]; memcpy(saved_batch,logits,sizeof(saved_batch));
        need(!ds4_bonsai_metal_prefill(s,tokens,3,logits) &&
             !memcmp(saved_batch,logits,sizeof(saved_batch)),
             "multi-token context overflow leaves output and state intact");
        for(;pos<LONG_CTX;++pos) {
            need(ds4_bonsai_metal_eval(s,tokens[pos],logits),"decode after multiple chunks");
            need(!memcmp(logits,serial+pos*VOCAB,sizeof(logits)),"decode state after chunks bit parity");
        }
        float saved[VOCAB]; memcpy(saved,logits,sizeof(saved));
        need(!ds4_bonsai_metal_prefill(s,tokens,1,logits) && !memcmp(saved,logits,sizeof(saved)),
             "prefill context overflow leaves output intact");
    }
    ds4_bonsai_metal_reset(s);
    int bad[]={2,VOCAB};
    need(!ds4_bonsai_metal_prefill(s,NULL,1,logits) &&
         !ds4_bonsai_metal_prefill(s,tokens,0,logits) &&
         !ds4_bonsai_metal_prefill(s,tokens,DS4_BONSAI_METAL_PREFILL_CAP+1,logits) &&
         !ds4_bonsai_metal_prefill(s,bad,2,logits),"invalid batch rejected before state advance");
    need(ds4_bonsai_metal_prefill(s,tokens,5,NULL),"batched prefix skipping head");
    need(ds4_bonsai_metal_prefill(s,tokens+5,3,logits),"append after skipped batched head");
    need(!memcmp(logits,serial+7*VOCAB,sizeof(logits)),"invalid input and skipped head preserve state");
    // Large batches also begin at nonaligned positions. Interleave short
    // tails and a skipped vocabulary projection, then continue with decode.
    ds4_bonsai_metal_reset(s);
    const uint32_t mixed[]={3,128,5,64,32,25};
    unsigned pos=0;
    for(unsigned c=0;c<sizeof(mixed)/sizeof(mixed[0]);++c) {
        if(c==1) {
            float saved[VOCAB];memcpy(saved,logits,sizeof(saved));
            need(!ds4_bonsai_metal_prefill(s,tokens+pos,DS4_BONSAI_METAL_PREFILL_CAP+1,logits) &&
                 !memcmp(saved,logits,sizeof(saved)),"oversized batch preserves nonzero-position state");
        }
        const bool skip_head=c==2;
        need(ds4_bonsai_metal_prefill(s,tokens+pos,mixed[c],skip_head?NULL:logits),
             "mixed-size prefill at nonaligned positions");
        pos+=mixed[c];
        if(!skip_head)need(!memcmp(logits,serial+(pos-1)*VOCAB,sizeof(logits)),
                          "mixed-size frontier bit parity");
    }
    need(pos==LONG_CTX-2,"mixed-size fixture leaves two decode tokens");
    for(;pos<LONG_CTX;++pos) {
        need(ds4_bonsai_metal_eval(s,tokens[pos],logits),"decode after mixed-size prefill");
        need(!memcmp(logits,serial+pos*VOCAB,sizeof(logits)),"mixed-size recurrent state bit parity");
    }
    ds4_bonsai_metal_free(s); fixture_free(&f);
}

/* With four attention heads this context reduces the score workspace to
 * six queries. Exercise host dispatch offsets and a partial query batch,
 * using a short actual sequence so the fixture stays inexpensive. */
static void test_attention_workspace(void) {
    fixture f; fixture_init(&f,8,6,2);
    enum { CAPACITY=300000, TOKENS=35 };
    f.m.context=CAPACITY;
    ds4_bonsai_metal *s=ds4_bonsai_metal_create(&f.m,CAPACITY);
    need(s!=NULL,"bounded attention workspace context");
    int tokens[TOKENS]; float serial[TOKENS*VOCAB],logits[VOCAB];
    for (unsigned i=0;i<TOKENS;++i) {
        tokens[i]=(int)((i*7+3)%VOCAB);
        need(ds4_bonsai_metal_eval(s,tokens[i],serial+i*VOCAB),"workspace serial reference");
    }
    ds4_bonsai_metal_reset(s);
    need(ds4_bonsai_metal_prefill(s,tokens,17,logits),"workspace partial query batch");
    need(!memcmp(logits,serial+16*VOCAB,sizeof(logits)),"workspace first frontier bit parity");
    need(ds4_bonsai_metal_prefill(s,tokens+17,17,logits),"workspace reuse and nonzero position");
    need(!memcmp(logits,serial+33*VOCAB,sizeof(logits)),"workspace reused frontier bit parity");
    need(ds4_bonsai_metal_eval(s,tokens[34],logits),"workspace decode continuation");
    need(!memcmp(logits,serial+34*VOCAB,sizeof(logits)),"workspace decode bit parity");
    ds4_bonsai_metal_free(s); fixture_free(&f);
}
#else
#define check_metal(m,t,n,e) ((void)0)
#define test_prefill_chunks() ((void)0)
#define test_attention_workspace() ((void)0)
#endif

static void test_sequence(fixture *f) {
    const int tokens[CTX] = {2,7,3,11,5,19,0,31};
    float expected[CTX*VOCAB], logits[VOCAB], other[2*VOCAB];
    const uint64_t before = fingerprint(f);
    run_cpu(&f->m, tokens, CTX, expected);
    ds4_bonsai_cpu *s = ds4_bonsai_cpu_create(&f->m, CTX);
    need(s != NULL, "second CPU session");
    for (unsigned repeat = 0; repeat < 2; repeat++) {
        ds4_bonsai_cpu_reset(s);
        need(!ds4_bonsai_cpu_eval(s, -1, logits) && !ds4_bonsai_cpu_eval(s, VOCAB, logits), "reject invalid token without advancing");
        for (unsigned t = 0; t < CTX; t++) {
            need(ds4_bonsai_cpu_eval(s, tokens[t], logits), "CPU repeat after reset");
            need(memcmp(logits, expected+t*VOCAB, sizeof(logits)) == 0, "CPU reset is bitwise reproducible");
        }
        memcpy(other, logits, sizeof(logits));
        need(!ds4_bonsai_cpu_eval(s, 2, logits) && !memcmp(other, logits, sizeof(logits)), "context bound leaves logits untouched");
    }
    ds4_bonsai_cpu_reset(s);
    for (unsigned t = 0; t < CTX; t++)
        need(ds4_bonsai_cpu_eval(s, tokens[t], t + 1 == CTX ? logits : NULL), "CPU ingest without logits");
    need(!memcmp(logits, expected+(CTX-1)*VOCAB, sizeof(logits)), "skipped output projection preserves state");
    ds4_bonsai_cpu_free(s);
    const int alternative[] = {9,7};
    run_cpu(&f->m, alternative, 2, other);
    need(difference(other+VOCAB, expected+VOCAB, VOCAB) > 1e-4f, "second token depends on earlier context");
    run_cpu(&f->m, tokens+1, 1, other);
    need(difference(other, expected+VOCAB, VOCAB) > 1e-4f, "warm second token differs from fresh token");
    check_metal(&f->m, tokens, CTX, expected);
    float grouped[CTX*VOCAB];
    f->m.gdn_v_grouped = true;
    run_cpu(&f->m, tokens, CTX, grouped);
    need(!memcmp(grouped, expected, sizeof(grouped)), "grouped metadata only affects folded output weights");
    check_metal(&f->m, tokens, CTX, expected);
    f->m.gdn_v_grouped = false;
    need(fingerprint(f) == before, "graph never mutates borrowed weights");
}

static void fill_ones(ds4_bonsai_tensor *w) {
    for (uint32_t i = 0; i < w->cols*w->rows; i++) ((float *)w->data)[i] = 1.f;
}
static float silu_ref(float x) { return x / (1.f + expf(-x)); }
static void test_silu(void) {
    fixture f; fixture_init(&f, 8, 6, 2);
    for (size_t i = 0; i < f.count; i++) memset(f.allocation[i], 0, f.bytes[i]);
    fill_ones(&f.m.embedding); fill_ones(&f.m.output_norm);
    for (unsigned i = 0; i < LAYERS; i++) {
        fill_ones(&f.m.layer[i].norm); fill_ones(&f.m.layer[i].post_norm);
        if (i < 3) fill_ones(&f.m.layer[i].ssm_norm);
        else { fill_ones(&f.m.layer[i].q_norm); fill_ones(&f.m.layer[i].k_norm); }
    }
    ds4_bonsai_layer *l = f.m.layer;
    for (unsigned i = 0; i < EMB; i++) {
        ((float *)l->out.data)[i*l->out.cols+i] = 1.f;
        ((float *)f.m.output.data)[i*EMB+i] = 1.f;
    }
    for (unsigned c = 0; c < 80; c++) ((float *)l->conv.data)[c*4+3] = 1.f;
    for (unsigned h = 0; h < 4; h++) for (unsigned d = 0; d < 8; d++)
        ((float *)l->qkv.data)[(h*8+d)*EMB] = 0.5f + 0.0625f*d;
    for (unsigned h = 0; h < 6; h++) for (unsigned d = 0; d < 8; d++) {
        ((float *)l->qkv.data)[(32+h*8+d)*EMB] = 0.15f + 0.03f*h + 0.045f*d;
        ((float *)l->z.data)[(h*8+d)*EMB] = -1.6f + 0.13f*h + 0.07f*d;
    }
    float actual[VOCAB], expected[VOCAB], wrong[VOCAB];
    const int token = 0;
    run_cpu(&f.m, &token, 1, actual);
    const float norm_x = 1.f/sqrtf(1.f+f.m.eps);
    for (unsigned mode = 0; mode < 2; mode++) {
        float hidden[EMB];
        for (unsigned h = 0; h < 4; h++) {
            float value[8]; double sum = 0;
            for (unsigned d = 0; d < 8; d++) {
                value[d] = 0.5f*silu_ref((0.15f+0.03f*h+0.045f*d)*norm_x)/sqrtf(8.f);
                sum += (double)value[d]*value[d];
            }
            const float r = 1.f/sqrtf((float)(sum/8)+f.m.eps);
            for (unsigned d = 0; d < 8; d++) {
                const float z = (-1.6f+0.13f*h+0.07f*d)*norm_x;
                const float gate = mode ? 1.f/(1.f+expf(-z)) : silu_ref(z);
                hidden[h*8+d] = 1.f+value[d]*r*gate;
            }
        }
        double sum = 0;
        for (unsigned i = 0; i < EMB; i++) sum += (double)hidden[i]*hidden[i];
        const float r = 1.f/sqrtf((float)(sum/EMB)+f.m.eps);
        for (unsigned i = 0; i < VOCAB; i++) (mode ? wrong : expected)[i] = hidden[i]*r;
    }
    close_logits(actual, expected, VOCAB, 2e-6f, 2e-6f, "analytic first-token GDN SiLU gate");
    need(difference(actual, wrong, VOCAB) > 0.03f, "fixture discriminates SiLU from sigmoid");
    check_metal(&f.m, &token, 1, expected);
    fixture_free(&f);
}

static void test_grouped(void) {
    /* A 2048-wide GDN output makes the real signed Hadamard path available;
     * K2/H16 gives a non-self-inverse transpose, unlike K2/H4. */
    fixture f; fixture_init(&f, 128, 16, 2);
    const int tokens[] = {2,7,3,11,5};
    float expected[5*VOCAB], actual[5*VOCAB];
    run_cpu(&f.m, tokens, 5, expected);
    int32_t *signs = own(&f, 2048*sizeof(int32_t));
    for (unsigned i = 0; i < 2048; i++) signs[i] = (i*7u+i/11u)%3u ? 1 : -1;
    float reordered[2048], transformed[2048];
    for (unsigned il = 0; il < 3; il++) {
        ds4_bonsai_tensor *w = &f.m.layer[il].out;
        float *data = (float *)w->data;
        for (unsigned row = 0; row < EMB; row++) {
            for (unsigned h = 0; h < 16; h++) for (unsigned d = 0; d < 128; d++)
                reordered[((h%2)*8+h/2)*128+d] = data[row*2048+h*128+d];
            need(ds4_bonsai_hadamard_forward(transformed, reordered, signs, 2048), "fold grouped output weight");
            memcpy(data+row*2048, transformed, sizeof(transformed));
        }
        w->signs = signs;
    }
    f.m.gdn_v_grouped = true;
    need(ds4_bonsai_model_valid(&f.m), "grouped signed F32 synthetic model");
    run_cpu(&f.m, tokens, 5, actual);
    close_logits(actual, expected, 5*VOCAB, 3e-5f, 3e-5f, "grouped permutation plus signed folded output preserves graph");
    check_metal(&f.m, tokens, 5, expected);
    f.m.gdn_v_grouped = false;
    run_cpu(&f.m, tokens, 5, actual);
    need(difference(actual, expected, 5*VOCAB) > 1e-3f, "fixture requires grouped permutation");
    fixture_free(&f);
}

static void test_packed_graph(void) {
    /* Quantized lookup + inverse rotation, and quantized vocabulary matmul +
     * forward rotation, compared with equivalent unfolded F32 matrices.
     * The trunk remains small; only the embedding width grows to 1024. */
    for (unsigned variant = 0; variant < 2; variant++) {
        fixture f; fixture_init_width(&f, 1024, 8, 6, 2);
        ds4_bonsai_model quantized = f.m;
        int32_t *signs = own(&f, 1024*sizeof(int32_t));
        for (unsigned i = 0; i < 1024; i++) signs[i] = (i+i/7u)%3u ? 1 : -1;
        ds4_bonsai_tensor *plain[2] = {&f.m.embedding, &f.m.output};
        ds4_bonsai_tensor *packed[2] = {&quantized.embedding, &quantized.output};
        for (unsigned which = 0; which < 2; which++) {
            const uint32_t type = 142u + ((which+variant)%2u);
            const size_t stride = type == 142 ? 34 : 28, row_bytes = 8*stride;
            uint8_t *bytes = own(&f, VOCAB*row_bytes);
            *packed[which] = (ds4_bonsai_tensor){.data=bytes,.bytes=VOCAB*row_bytes,
                .type=type,.cols=1024,.rows=VOCAB,.signs=signs};
            float latent[1024];
            for (unsigned row = 0; row < VOCAB; row++) {
                for (unsigned b = 0; b < 8; b++) {
                    uint8_t *block = bytes+row*row_bytes+b*stride;
                    for (size_t j = 0; j < stride; j++) block[j] = (uint8_t)random_u32();
                    const size_t scale_offset = type == 142 ? 0 : 26;
                    block[scale_offset] = 0;
                    block[scale_offset+1] = (uint8_t)(0x18 + b%4); /* Finite dyadic fp16 scales. */
                }
                need(ds4_bonsai_dequantize_row(type, bytes+row*row_bytes, latent, 1024), "packed graph row decode");
                need(ds4_bonsai_hadamard_inverse((float *)plain[which]->data+row*1024, latent, signs, 1024),
                     "unfold packed graph oracle weights");
            }
        }
        need(ds4_bonsai_model_valid(&quantized), "packed signed synthetic graph validation");
        const int32_t saved_sign = signs[1023];
        signs[1023] = 0;
        need(!ds4_bonsai_model_valid(&quantized) && !ds4_bonsai_cpu_create(&quantized, 1),
             "reject invalid sign at the end of a correctly sized table");
        signs[1023] = saved_sign;
        const int tokens[] = {2,7,3};
        float expected[3*VOCAB], actual[3*VOCAB];
        run_cpu(&f.m, tokens, 3, expected);
        run_cpu(&quantized, tokens, 3, actual);
        close_logits(actual, expected, 3*VOCAB, 3e-5f, 3e-5f, "packed signed graph matches unfolded F32");
        check_metal(&quantized, tokens, 3, actual);
        fixture_free(&f);
    }
}

static void test_sibling_rotations(void) {
    fixture f; fixture_init_width(&f,1024,8,6,2);
    const int tokens[]={2,7,3,11,5};
    float expected[5*VOCAB], actual[5*VOCAB], rotated[1024];
    run_cpu(&f.m,tokens,5,expected);
    int32_t *shared=own(&f,1024*sizeof(int32_t));
    int32_t *different=own(&f,1024*sizeof(int32_t));
    for (unsigned i=0;i<1024;++i) {
        shared[i]=(i%3) ? 1 : -1;
        different[i]=(i%5) ? -1 : 1;
    }
    for (unsigned il=0;il<LAYERS;++il) {
        ds4_bonsai_layer *l=&f.m.layer[il];
        ds4_bonsai_tensor *siblings[]={&l->gate,&l->up,
            il==3 ? &l->q : &l->qkv, il==3 ? &l->k : &l->z,
            il==3 ? &l->v : NULL};
        for (unsigned j=0;j<5 && siblings[j];++j) {
            ds4_bonsai_tensor *t=siblings[j];
            // Same-width siblings can have different signs. The BF16/F32
            // alpha/beta projections remain unfolded and use the raw input.
            t->signs=(il==1 && j==3) || (il==3 && j==3) ? different : shared;
            for (unsigned row=0;row<t->rows;++row) {
                float *data=(float *)t->data+row*1024;
                need(ds4_bonsai_hadamard_forward(rotated,data,t->signs,1024),"fold sibling projection");
                memcpy(data,rotated,sizeof(rotated));
            }
        }
    }
    run_cpu(&f.m,tokens,5,actual);
    close_logits(actual,expected,5*VOCAB,3e-5f,3e-5f,"sibling folds preserve graph");
    check_metal(&f.m,tokens,5,actual);
    fixture_free(&f);
}

static void test_packed_projection_batch(void) {
    /* Unlike the packed embedding/head fixture, these weights enter bonsai_mm
     * during prefill. Five tokens exercise both a full tile and its tail after
     * per-row signed Hadamard rotation; alpha/beta stay on the unrotated input.
     * The CPU graph independently decodes every row and accumulates in double. */
    fixture f; fixture_init_width(&f,1024,8,6,2);
    int32_t *signs=own(&f,1024*sizeof(int32_t));
    for (unsigned i=0;i<1024;++i) signs[i]=(i+i/13u)%3u ? 1 : -1;
    for (unsigned il=0;il<LAYERS;++il) {
        ds4_bonsai_layer *l=&f.m.layer[il];
        ds4_bonsai_tensor *weights[]={&l->gate,&l->up,
            il<3 ? &l->qkv : NULL,il<3 ? &l->z : NULL};
        for (unsigned wi=0;wi<4 && weights[wi];++wi) {
            ds4_bonsai_tensor *w=weights[wi];
            const size_t blocks=w->cols/128u,row_bytes=blocks*34u;
            uint8_t *packed=own(&f,(size_t)w->rows*row_bytes);
            for (uint32_t row=0;row<w->rows;++row) for (size_t b=0;b<blocks;++b) {
                uint8_t *p=packed+(size_t)row*row_bytes+b*34u;
                const uint16_t scale=(uint16_t)(0x1400u+((row+b+wi+il)%4u)*0x400u);
                p[0]=(uint8_t)scale; p[1]=(uint8_t)(scale>>8);
                for (unsigned j=0;j<32;++j) {
                    uint8_t byte=0;
                    for (unsigned lane=0;lane<4;++lane)
                        byte|=(uint8_t)(((row*13u+b*17u+j*7u+lane*5u+wi+il)%3u)<<(2u*lane));
                    p[j+2u]=byte;
                }
            }
            *w=(ds4_bonsai_tensor){.data=packed,.bytes=(uint64_t)w->rows*row_bytes,
                .type=142,.cols=w->cols,.rows=w->rows,.signs=signs};
        }
    }
    need(ds4_bonsai_model_valid(&f.m),"PQ2 folded projection fixture validation");
    const int tokens[]={2,7,3,11,5};
    float expected[5*VOCAB];
    run_cpu(&f.m,tokens,5,expected);
    check_metal(&f.m,tokens,5,expected);
    fixture_free(&f);
}

static void reject_model(const ds4_bonsai_model *m, const char *what) {
    need(!ds4_bonsai_model_valid(m), what);
    ds4_bonsai_cpu *s = ds4_bonsai_cpu_create(m, 1);
    need(s == NULL, "CPU create rejects malformed model");
}
static void test_invalid(fixture *f) {
    ds4_bonsai_model bad;
#define REJECT(field,value) do { bad=f->m; bad.field=(value); reject_model(&bad,#field); } while (0)
    reject_model(NULL, "NULL model");
    REJECT(n_layer,0); REJECT(n_layer,DS4_BONSAI_LAYERS+1); REJECT(full_interval,0);
    REJECT(n_embd,0); REJECT(n_vocab,0); REJECT(n_ff,0); REJECT(context,0);
    REJECT(n_head,3); REJECT(n_kv_head,0); REJECT(head_dim,0); REJECT(n_rot,9); REJECT(n_rot,3); REJECT(n_rot,0);
    REJECT(n_k_head,0); REJECT(n_v_head,5); REJECT(ssm_dim,0); REJECT(conv_width,1); REJECT(conv_width,0);
    REJECT(eps,0); REJECT(eps,NAN); REJECT(rope_base,INFINITY); REJECT(rope_base,0);
    REJECT(n_head,UINT32_MAX); REJECT(n_k_head,UINT32_MAX); REJECT(conv_width,UINT32_MAX);
    REJECT(output.data,NULL); REJECT(output.bytes,1); REJECT(output.rows,VOCAB+1); REJECT(output.type,42);
    REJECT(layer[0].norm.type,1); REJECT(layer[0].conv.type,30); REJECT(layer[0].qkv.cols,EMB+1);
    REJECT(layer[0].norm.data,(const uint8_t *)f->m.layer[0].norm.data+1);
    REJECT(layer[3].q.rows,32); REJECT(layer[3].k_norm.cols,7); REJECT(embedding.type,142);
    int32_t sign = 1;
    REJECT(layer[0].out.signs,&sign); /* Width 48 cannot use a 1024 rotation. */
#undef REJECT
    need(!ds4_bonsai_cpu_create(&f->m,0) && !ds4_bonsai_cpu_create(&f->m,CTX+1), "context creation bounds");
    need(!ds4_bonsai_cpu_eval(NULL,0,NULL), "NULL session eval");
    ds4_bonsai_cpu_free(NULL);
}

int main(void) {
    test_score_capacity();
    fixture f; fixture_init(&f, 8, 6, 2);
    test_invalid(&f); test_sequence(&f); fixture_free(&f);
    test_silu(); test_grouped(); test_packed_graph(); test_sibling_rotations();
    test_packed_projection_batch(); test_prefill_chunks(); test_attention_workspace();
    puts("PASS Bonsai graph: 3 GDN + 1 full attention, causal state/reset, SiLU, grouped output, packed Hadamard, validation");
    return 0;
}
