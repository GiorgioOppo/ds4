#include "../ds4.c"
#include <assert.h>

#ifdef DS4_HAS_QWEN4_METAL
/* Compare cache-only prefix preparation with the existing complete MTP
 * layer on exactly the same trunk rows. This uses real Metal/model weights;
 * it never runs the full CPU model or substitutes a mock dequantizer. */
enum { TEST_CTX = 128, TEST_CHUNK = 8, TEST_PREFIX = 13, TEST_APPEND = 16, MAX_FRAMES = 4 };
static char error[256];

static void need(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "Qwen MTP prefill: %s: %s\n", what, error);
        exit(1);
    }
}

static void *read_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    void *p = malloc(bytes ? (size_t)bytes : 1u);
    need(p != NULL, "host comparison allocation");
    need(!bytes || ds4_gpu_tensor_read(t, 0, p, bytes), "read comparison tensor");
    return p;
}

static void exact(const char *what, const void *a, const void *b, size_t bytes) {
    if (!memcmp(a, b, bytes)) return;
    const unsigned char *x = a, *y = b;
    size_t first = 0;
    while (first < bytes && x[first] == y[first]) first++;
    fprintf(stderr, "Qwen MTP prefill: %s differs at byte %zu/%zu\n", what, first, bytes);
    exit(1);
}

typedef struct {
    uint32_t rows, mtp_rows, residual_rows;
    uint64_t kv_bytes, ik_bytes, block_bytes, residual_bytes;
    void *k, *v, *ik, *block, *tail, *residual;
    float *logits;
} frontier;

static frontier capture(ds4_session *s, uint32_t residual_rows) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
    frontier f = {.rows = g->pos, .mtp_rows = g->mtp_pos, .residual_rows = residual_rows};
    f.kv_bytes = qwen4_payload_kv_bytes(f.mtp_rows);
    f.ik_bytes = qwen4_payload_ik_bytes(f.mtp_rows);
    f.block_bytes = qwen4_payload_block_key_bytes(f.mtp_rows);
    f.residual_bytes = residual_rows * row_bytes;
    f.k = read_bytes(g->layer_k_cache[il], f.kv_bytes);
    f.v = read_bytes(g->layer_v_cache[il], f.kv_bytes);
    f.ik = read_bytes(g->layer_ik_cache[il], f.ik_bytes);
    f.block = read_bytes(g->layer_block_key[il], f.block_bytes);
    f.tail = read_bytes(g->mtp_tail_R, row_bytes);
    f.residual = read_bytes(g->R, f.residual_bytes);
    f.logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    need(f.logits != NULL, "logits comparison allocation");
    memcpy(f.logits, s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
    return f;
}

static void free_frontier(frontier *f) {
    free(f->k); free(f->v); free(f->ik); free(f->block);
    free(f->tail); free(f->residual); free(f->logits);
    memset(f, 0, sizeof(*f));
}

static void compare_frontier(ds4_session *s, const frontier *want, bool residual) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    need(s->checkpoint_valid && g->pos == want->rows &&
         ds4_session_pos(s) == (int)want->rows, "target frontier position");
    need(g->mtp_pos == want->mtp_rows, "contiguous MTP prefix position");
    need(g->mtp_tail_valid && g->mtp_tail_pos + 1u == g->pos, "last trunk row retained");
    frontier got = capture(s, residual ? want->residual_rows : 0);
    exact("nextn K", want->k, got.k, want->kv_bytes);
    exact("nextn V", want->v, got.v, want->kv_bytes);
    exact("nextn indexer keys", want->ik, got.ik, want->ik_bytes);
    exact("nextn pooled keys", want->block, got.block, want->block_bytes);
    exact("retained trunk row", want->tail, got.tail,
          (size_t)DS4_N_EMBD * DS4_N_HC * sizeof(float));
    if (residual) exact("unmodified trunk residual", want->residual, got.residual, want->residual_bytes);
    exact("unmodified target logits", want->logits, got.logits, (size_t)DS4_N_VOCAB * sizeof(float));
    free_frontier(&got);
}

static void poison_nextn(ds4_session *s, unsigned char value) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    ds4_gpu_tensor *t[] = {g->layer_k_cache[il], g->layer_v_cache[il],
                          g->layer_ik_cache[il], g->layer_block_key[il]};
    for (unsigned i = 0; i < 4; i++) {
        const uint64_t bytes = ds4_gpu_tensor_bytes(t[i]);
        void *p = malloc((size_t)bytes);
        need(p != NULL, "poison allocation"); memset(p, value, (size_t)bytes);
        need(ds4_gpu_tensor_write(t[i], 0, p, bytes), "poison old nextn cache");
        free(p);
    }
}

/* Full MTP is the oracle. With a known lookahead it also fills the last row
 * of an intermediate chunk, so graph_forward_tokens never needs to repair
 * that boundary through the new cache-only helper. */
static unsigned reference_sync(ds4_session *s, const ds4_tokens *tokens,
                               uint32_t start, frontier out[MAX_FRAMES]) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    ds4_engine *e = s->engine;
    unsigned n = 0;
    if (!start) {
        ds4_session_invalidate(s); qwen4_graph_reset(g); poison_nextn(s, 0x55);
    } else if (g->mtp_pos < start) {
        need(g->mtp_pos + 1u == start && g->mtp_tail_valid, "reference append boundary");
        ds4_gpu_tensor *saved = g->R;
        g->R = g->mtp_tail_R;
        const bool ok = qwen4_graph_mtp_step(g, &e->model, &e->weights,
            0, tokens->v[start], start - 1u, false, NULL, NULL);
        g->R = saved;
        need(ok, "full MTP append boundary oracle");
    }
    for (uint32_t pos = start; pos < (uint32_t)tokens->len;) {
        uint32_t T = (uint32_t)tokens->len - pos;
        if (T > TEST_CHUNK) T = TEST_CHUNK;
        need(qwen4_graph_forward_tokens(g, &e->model, &e->weights,
            tokens->v + pos, T, s->logits, false), "reference trunk forward");
        uint32_t rows = T;
        if (pos + T == (uint32_t)tokens->len) rows--;
        for (uint32_t row = 0; row < rows;) {
            const uint32_t count = rows - row < 3u ? rows - row : 3u;
            need(qwen4_graph_mtp_steps(g, &e->model, &e->weights, row,
                tokens->v + pos + row + 1u, count, pos + row, false, NULL, NULL),
                "full MTP prefix oracle");
            row += count;
        }
        for (uint32_t row = 0; row < T; row++) token_vec_push(&s->checkpoint, tokens->v[pos + row]);
        s->checkpoint_valid = true; s->qwen4_rewound = false;
        need(n < MAX_FRAMES, "bounded expected frontiers");
        out[n++] = capture(s, T);
        pos += T;
    }
    return n;
}

typedef struct { ds4_session *s; const frontier *frames; unsigned n, seen; } progress_check;
static void check_progress(void *opaque, const char *phase, int completed, int total) {
    progress_check *p = opaque;
    if (strcmp(phase, "prefill_chunk")) return;
    need(p->seen < p->n, "unexpected prefill callback");
    need(completed == (int)p->frames[p->seen].rows && completed <= total, "chunk callback frontier");
    compare_frontier(p->s, p->frames + p->seen++, true);
}

static void checked_sync(ds4_session *s, const ds4_tokens *tokens, frontier *f, unsigned n) {
    progress_check p = {.s = s, .frames = f, .n = n};
    ds4_session_set_progress(s, check_progress, &p);
    need(ds4_session_sync(s, tokens, error, sizeof(error)) == 0, "candidate prefix sync");
    ds4_session_set_progress(s, NULL, NULL);
    need(p.seen == n, "every completed chunk checked");
    need(s->qwen4_graph.mtp_last_rows == 0, "cache-only prefill does not publish predictor output");
    compare_frontier(s, f + n - 1u, true);
}

static void close_logits(const float *got, const float *want) {
    /* The existing 2/3-row verifier and one-row target have distinct
     * reductions; use the established MTP batch tolerance, not text alone. */
    float worst = 0.0f;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float d = fabsf(got[i] - want[i]);
        if (!isfinite(got[i]) || !isfinite(want[i]) || d > 2e-3f + 1e-4f * fabsf(want[i])) {
            fprintf(stderr, "Qwen MTP target mismatch id=%u got=%g want=%g error=%g\n",
                    i, (double)got[i], (double)want[i], (double)d);
            exit(1);
        }
        if (d > worst) worst = d;
    }
    printf("  all %u target logits: max error %.6g\n", DS4_N_VOCAB, (double)worst);
}

static void check_cycles(ds4_session *live, ds4_session *target,
                         const ds4_session_snapshot *snapshot, unsigned mode) {
    need(ds4_session_load_snapshot(live, snapshot, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(target, snapshot, error, sizeof(error)) == 0, "restore cycle frontier");
    const bool forced = mode == 1u, stochastic = mode == 2u;
    need(setenv("DS4_QWEN4_MTP_DEPTH", mode ? "3" : "2", 1) == 0, "set depth case");
    need((forced ? setenv("DS4_QWEN4_SPEC_FORCE_ACCEPT", "1", 1) :
                   unsetenv("DS4_QWEN4_SPEC_FORCE_ACCEPT")) == 0, "set acceptance case");
    live->engine->dspark_exact_sampling = stochastic;
    uint64_t rng = 0x3456789abcdefull;
    bool saw_verify = false, saw_three = false;
    for (unsigned step = 0; step < 3u; step++) {
        const int first = ds4_session_sample(target, stochastic ? 0.8f : 0.0f,
                                             0, 0.95f, 0.0f, &rng);
        int accepted[3] = {-1, -1, -1};
        const int before = ds4_session_pos(live);
        const int n = ds4_session_eval_speculative(live, first, 3, -1,
            stochastic ? 0.8f : 0.0f, 0, 0.95f, 0.0f, &rng, accepted, 3, error, sizeof(error));
        need(n >= 1 && n <= (stochastic ? 2 : 3), "bounded accepted count");
        need(accepted[0] == first && ds4_session_pos(live) == before + n, "committed prefix count");
        saw_verify |= n > 1; saw_three |= n == 3;
        for (int i = 0; i < n; i++) {
            need(accepted[i] >= 0 && accepted[i] < (int)DS4_N_VOCAB, "accepted token range");
            if (!forced && !stochastic)
                need(accepted[i] == ds4_session_argmax(target), "greedy proposal agrees with target");
            need(ds4_session_eval(target, accepted[i], error, sizeof(error)) == 0,
                 "independent one-row target continuation");
        }
        close_logits(live->logits, target->logits);
    }
    if (forced) need(saw_verify && saw_three, "depth-three verifier actually exercised");
    printf("PASS MTP continuation %s, independent target logits\n",
           forced ? "forced depth3 state fixture" : stochastic ? "exact sampling depth2" : "greedy depth2");
}

static void check_payload_modes(ds4_session *live, ds4_session *reference,
                                const ds4_tokens *prefix, const ds4_session_snapshot *snapshot) {
    ds4_engine *e = live->engine;
    /* Share one model mapping, but allocate the third graph exactly as a
     * session created with MTP disabled. Ordinary eval uses that graph only. */
    ds4_session *off = NULL;
    const bool enabled = e->glm_mtp;
    e->glm_mtp = false;
    const int create_rc = ds4_session_create(&off, e, TEST_CTX);
    e->glm_mtp = enabled;
    need(create_rc == 0 && off && !off->qwen4_graph.mtp_R, "allocate MTP-disabled graph");
    need(ds4_session_load_snapshot(off, snapshot, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(reference, snapshot, error, sizeof(error)) == 0,
         "MTP-on payload loads with MTP disabled");
    exact("MTP-disabled restored target logits", off->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
    const int next = (ds4_session_argmax(reference) + 1) % (int)DS4_N_VOCAB;
    need(ds4_session_eval(off, next, error, sizeof(error)) == 0 &&
         ds4_session_eval(reference, next, error, sizeof(error)) == 0,
         "MTP-disabled continuation from MTP-on payload");
    exact("MTP-disabled target continuation", off->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));

    need(ds4_session_load_snapshot(off, snapshot, error, sizeof(error)) == 0,
         "restore MTP-disabled snapshot source");
    ds4_session_snapshot plain = {0};
    need(ds4_session_save_snapshot(off, &plain, error, sizeof(error)) == 0,
         "save actual MTP-disabled payload");
    ds4_session_free(off);
    poison_nextn(live, 0x77);
    need(ds4_session_load_snapshot(live, &plain, error, sizeof(error)) == 0,
         "MTP-off payload loads with MTP enabled");
    need(live->qwen4_graph.mtp_pos == 0 && !live->qwen4_graph.mtp_tail_valid,
         "MTP-off payload does not claim a predictor prefix");
    frontier frames[MAX_FRAMES] = {{0}};
    const unsigned n = reference_sync(reference, prefix, 0, frames);
    checked_sync(live, prefix, frames, n);
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    ds4_session_snapshot_free(&plain);

    /* The old tag cannot describe the deferred final row. Reject it before
     * any tensor writes instead of interpreting its high-water mark as KV. */
    ds4_session_snapshot old = {.len = snapshot->len, .cap = snapshot->len};
    old.ptr = malloc((size_t)old.len);
    need(old.ptr != NULL, "old payload allocation");
    memcpy(old.ptr, snapshot->ptr, (size_t)old.len);
    uint32_t tag = 0;
    memcpy(&tag, old.ptr + 12u * sizeof(uint32_t), sizeof(tag));
    need(tag == DS4_QWEN4_PAYLOAD_TAG, "payload family tag location");
    tag = 0x51573802u;
    memcpy(old.ptr + 12u * sizeof(uint32_t), &tag, sizeof(tag));
    need(ds4_session_load_snapshot(live, &old, error, sizeof(error)) != 0,
         "reject legacy nextn payload without a retained row");
    error[0] = '\0';
    ds4_session_snapshot_free(&old);
    puts("PASS MTP on/off payload compatibility, prefix replay and legacy-tag rejection");
}

int main(int argc, char **argv) {
    const bool streaming = argc == 3 && !strcmp(argv[2], "--ssd-streaming");
    if (argc != 2 && !streaming) {
        fprintf(stderr, "usage: %s QWEN_GGUF [--ssd-streaming]\n", argv[0]); return 2;
    }
    const char *names[] = {"DS4_QWEN4_SPEC_FORCE_ACCEPT", "DS4_QWEN4_MTP_DEPTH"};
    char *saved[2] = {NULL, NULL};
    for (unsigned i = 0; i < 2; i++) {
        const char *v = getenv(names[i]); saved[i] = v ? strdup(v) : NULL;
        need(!v || saved[i], "save test environment"); need(unsetenv(names[i]) == 0, "clear test override");
    }
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = TEST_CTX, .prefill_chunk = TEST_CHUNK, .glm_mtp = true,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 1024u : 0u};
    ds4_engine *engine = NULL;
    need(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen4(engine), "open Qwen model");
    ds4_session *live = NULL, *reference = NULL;
    need(ds4_session_create(&live, engine, TEST_CTX) == 0 &&
         ds4_session_create(&reference, engine, TEST_CTX) == 0, "allocate two bounded sessions");
    need(live->qwen4_graph.cap_tokens == TEST_CHUNK, "requested chunk size");
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten and explain the pattern.", DS4_THINK_NONE, &prompt);
    need(prompt.len >= TEST_APPEND, "enough real prompt token IDs");
    ds4_tokens prefix = prompt; prefix.len = 1;
    frontier frames[MAX_FRAMES] = {{0}};
    unsigned n = reference_sync(reference, &prefix, 0, frames);
    poison_nextn(live, 0xaa);
    checked_sync(live, &prefix, frames, n);
    need(live->qwen4_graph.mtp_pos == 0 && !live->qwen4_graph.mtp_tail_cached,
         "one-token prefix has only a deferred row");
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    ds4_session_invalidate(live);
    prefix.len = TEST_PREFIX;
    n = reference_sync(reference, &prefix, 0, frames);
    poison_nextn(live, 0xaa);
    checked_sync(live, &prefix, frames, n);
    need(live->qwen4_graph.mtp_pos + 1u == TEST_PREFIX, "last unknown token is not guessed");
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    puts("PASS cache-only prefix against full MTP: chunk lookahead and final retained row");

    /* Force the deferred row to use a real next input, not a guessed argmax. */
    prompt.v[TEST_PREFIX] = (ds4_session_argmax(reference) + 1) % (int)DS4_N_VOCAB;
    prefix.len = TEST_APPEND;
    n = reference_sync(reference, &prefix, TEST_PREFIX, frames);
    checked_sync(live, &prefix, frames, n);
    ds4_session_snapshot snapshot = {0};
    need(ds4_session_save_snapshot(live, &snapshot, error, sizeof(error)) == 0, "save prepared prefix");
    /* Load into a graph containing unrelated nextn rows and a different tail. */
    ds4_tokens other = {0};
    for (int i = 0; i < TEST_PREFIX; i++) ds4_tokens_push(&other, prompt.v[(i + 2) % TEST_APPEND]);
    need(ds4_session_sync(live, &other, error, sizeof(error)) == 0, "reuse with unrelated prefix");
    poison_nextn(live, 0x33);
    need(ds4_session_load_snapshot(live, &snapshot, error, sizeof(error)) == 0, "restore over stale cache");
    compare_frontier(live, frames + n - 1u, false);
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    puts("PASS append boundary and checkpoint restore over reused state");

    check_payload_modes(live, reference, &prefix, &snapshot);
    for (unsigned mode = 0; mode < 3; mode++) check_cycles(live, reference, &snapshot, mode);
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens_free(&other); ds4_tokens_free(&prompt);
    ds4_session_free(reference); ds4_session_free(live); ds4_engine_close(engine);
    for (unsigned i = 0; i < 2; i++) {
        need((saved[i] ? setenv(names[i], saved[i], 1) : unsetenv(names[i])) == 0, "restore test environment");
        free(saved[i]);
    }
    puts("Qwen MTP prefill: all checks passed");
    return 0;
}
#else
int main(void) { fputs("Qwen MTP prefill test requires Metal\n", stderr); return 2; }
#endif
