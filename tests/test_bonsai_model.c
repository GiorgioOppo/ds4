/* Real-model test of ds4's public Bonsai engine/session API.
 * Compile against the normal core objects, including ds4_bonsai.o and (on
 * macOS) ds4_bonsai_metal.o. Do not include ds4.c in this driver.
 *
 * Usage: test_bonsai_model --model MODEL (--tokens IDS | --text TEXT)
 *   [--backend cpu|metal] [--decode N] [--ctx N] [--chunk N] [--threads N]
 *   [--chat] [--ids-out FILE] [--ids-only] [--logits FILE] [--all-logits]
 *   [--lifecycle]
 *
 * IDS contains whitespace-separated decimal IDs. Raw text uses
 * ds4_tokenize_text; --chat uses the public chat formatter with THINK_NONE.
 * N defaults to zero; generation evaluates exactly N greedy tokens, even EOG.
 * FILE logits are native float32, with FILE.rows.tsv matching the reference
 * harness: row, zero-based position, input_id, argmax. Without --all-logits,
 * capture the prefill frontier and each decoded token; with it capture every
 * prompt token too. --ids-out writes plain prompt IDs for tokenizer parity.
 *
 * --lifecycle is opt-in and uses at most the first four supplied prompt IDs
 * plus one appended token. It checks same-prefix sync, append, changed-prefix
 * sync, invalidate, rewind+sync+eval replay and rewind+sync against fresh sessions
 * on the same engine/backend. Same-prefix sync must preserve logits bitwise;
 * fresh-session comparisons require finite logits, identical argmax, max
 * absolute error <= 1e-4 and RMSE <= 1e-5. Different prompt/decode kernels can
 * use different FP32 accumulation orders. Cross-backend parity is not imposed.
 */
#include "../ds4.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) abort();
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static char api_error[512];
static const double lifecycle_max_abs = 1e-4;
static const double lifecycle_max_rmse = 1e-5;

static void require(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "Bonsai model test: %s%s%s\n", message,
            api_error[0] ? ": " : "", api_error);
    exit(1);
}

static int number(const char *s, int minimum) {
    char *end = NULL;
    errno = 0;
    long value = strtol(s, &end, 10);
    require(!errno && *s && end && !*end && value >= minimum && value <= INT_MAX,
            "invalid numeric argument or token ID");
    return (int)value;
}

static FILE *open_file(const char *path, const char *mode) {
    FILE *file = fopen(path, mode);
    if (!file) snprintf(api_error, sizeof(api_error), "%s: %s", path, strerror(errno));
    require(file != NULL, "open file");
    return file;
}

static void read_ids(const char *path, ds4_tokens *tokens) {
    FILE *file = open_file(path, "r");
    char token[64];
    while (fscanf(file, "%63s", token) == 1) ds4_tokens_push(tokens, number(token, 0));
    require(!ferror(file), "read token file");
    require(fclose(file) == 0, "close token file");
}

static void copy_logits(ds4_session *session, float *logits, int vocab) {
    require(ds4_session_copy_logits(session, logits, vocab) == vocab, "copy complete vocabulary logits");
    for (int i = 0; i < vocab; ++i) {
        if (!isfinite(logits[i])) {
            snprintf(api_error, sizeof(api_error), "token %d: %g", i, logits[i]);
            require(0, "nonfinite vocabulary logit");
        }
    }
}

static void check_tokens(ds4_session *session, const ds4_tokens *expected) {
    const ds4_tokens *actual = ds4_session_tokens(session);
    require(ds4_session_pos(session) == expected->len && actual && actual->len == expected->len,
            "session position and token count");
    require(!memcmp(actual->v, expected->v, (size_t)expected->len * sizeof(int)),
            "exact session token timeline");
}

static void compare_logits(const char *label, const float *actual, const float *expected, int vocab) {
    if (!memcmp(actual, expected, (size_t)vocab * sizeof(float))) return;
    int first = -1, changed = 0;
    double worst = 0;
    for (int i = 0; i < vocab; ++i) {
        if (memcmp(actual + i, expected + i, sizeof(float))) {
            if (first < 0) first = i;
            ++changed;
            const double error = fabs((double)actual[i] - expected[i]);
            if (error > worst) worst = error;
        }
    }
    snprintf(api_error, sizeof(api_error),
             "%s: changed=%d/%d first=%d actual=%.9g fresh=%.9g max_abs=%.9g",
             label, changed, vocab, first, actual[first], expected[first], worst);
    require(0, "same-backend session logits differ");
}

static void compare_fresh(ds4_engine *engine, ds4_session *subject, const ds4_tokens *prompt,
                          int context, int vocab, float *actual, float *expected, const char *label) {
    ds4_session *fresh = NULL;
    require(ds4_session_create(&fresh, engine, context) == 0, "create fresh reference session");
    api_error[0] = 0;
    require(ds4_session_sync(fresh, prompt, api_error, sizeof(api_error)) == 0, "sync fresh reference");
    check_tokens(subject, prompt);
    check_tokens(fresh, prompt);
    copy_logits(subject, actual, vocab);
    copy_logits(fresh, expected, vocab);
    double max_abs = 0.0, squared_error = 0.0;
    int actual_top = 0, expected_top = 0;
    for (int i = 0; i < vocab; ++i) {
        const double delta = (double)actual[i] - (double)expected[i];
        if (fabs(delta) > max_abs) max_abs = fabs(delta);
        squared_error += delta * delta;
        if (actual[i] > actual[actual_top]) actual_top = i;
        if (expected[i] > expected[expected_top]) expected_top = i;
    }
    const double rmse = sqrt(squared_error / (double)vocab);
    const bool pass = actual_top == expected_top && max_abs <= lifecycle_max_abs &&
                      rmse <= lifecycle_max_rmse;
    ds4_session_free(fresh);
    fprintf(stderr, "%s lifecycle %s: tokens=%d finite_logits=%d argmax=%d fresh_argmax=%d max_abs=%.9g rmse=%.9g\n",
            pass ? "PASS" : "FAIL", label, prompt->len, vocab, actual_top, expected_top, max_abs, rmse);
    require(pass, "fresh-session logits exceed fixed lifecycle criteria");
}

static void lifecycle(ds4_engine *engine, const ds4_tokens *input, int context, int vocab) {
    fprintf(stderr, "LIFECYCLE criteria: fresh-session logits must all be finite, argmax equal, "
            "max_abs<=%.9g, RMSE<=%.9g; same-prefix sync must be bit-exact\n",
            lifecycle_max_abs, lifecycle_max_rmse);
    ds4_tokens prompt = {0}, extended = {0};
    for (int i = 0; i < input->len && i < 4; ++i) ds4_tokens_push(&prompt, input->v[i]);
    ds4_tokens_copy(&extended, &prompt);
    ds4_tokens_push(&extended, prompt.v[0]);
    ds4_session *subject = NULL;
    float *actual = malloc((size_t)vocab * sizeof(float));
    float *expected = malloc((size_t)vocab * sizeof(float));
    require(actual && expected, "allocate lifecycle logits");
    require(ds4_session_create(&subject, engine, context) == 0, "create lifecycle session");
    require(ds4_session_sync(subject, &prompt, api_error, sizeof(api_error)) == 0, "initial lifecycle sync");
    compare_fresh(engine, subject, &prompt, context, vocab, actual, expected, "initial");
    memcpy(expected, actual, (size_t)vocab * sizeof(float));
    require(ds4_session_sync(subject, &prompt, api_error, sizeof(api_error)) == 0, "same-prefix sync");
    check_tokens(subject, &prompt);
    copy_logits(subject, actual, vocab);
    compare_logits("same-prefix", actual, expected, vocab);
    fprintf(stderr, "PASS lifecycle same-prefix: %d logits bit-exact\n", vocab);

    require(ds4_session_sync(subject, &extended, api_error, sizeof(api_error)) == 0, "append sync");
    compare_fresh(engine, subject, &extended, context, vocab, actual, expected, "append");
    extended.v[0] = (extended.v[0] + 1) % vocab;
    require(ds4_session_sync(subject, &extended, api_error, sizeof(api_error)) == 0, "changed-prefix sync");
    compare_fresh(engine, subject, &extended, context, vocab, actual, expected, "changed-prefix");

    ds4_session_invalidate(subject);
    require(ds4_session_pos(subject) == 0, "invalidate clears checkpoint");
    require(ds4_session_sync(subject, &extended, api_error, sizeof(api_error)) == 0, "sync after invalidate");
    compare_fresh(engine, subject, &extended, context, vocab, actual, expected, "invalidate");

    ds4_session_rewind(subject, extended.len - 1);
    require(ds4_session_pos(subject) == extended.len - 1, "rewind keeps requested prefix");
    require(ds4_session_eval(subject, extended.v[extended.len - 1], api_error, sizeof(api_error)) != 0,
            "rewound checkpoint requires sync before eval");
    require(ds4_session_pos(subject) == extended.len - 1, "rejected eval preserves rewound prefix");
    api_error[0] = 0;
    ds4_tokens retained = extended;
    retained.len--;
    require(ds4_session_sync(subject, &retained, api_error, sizeof(api_error)) == 0,
            "sync retained prefix reconstructs recurrent state and logits");
    check_tokens(subject, &retained);
    require(ds4_session_eval(subject, extended.v[extended.len - 1], api_error, sizeof(api_error)) == 0,
            "eval after synchronizing rewound prefix");
    compare_fresh(engine, subject, &extended, context, vocab, actual, expected, "rewind-sync-eval");
    ds4_session_rewind(subject, 1);
    require(ds4_session_sync(subject, &extended, api_error, sizeof(api_error)) == 0, "sync after rewind");
    compare_fresh(engine, subject, &extended, context, vocab, actual, expected, "rewind-sync");
    ds4_session_rewind(subject, 0);
    require(ds4_session_sync(subject, &prompt, api_error, sizeof(api_error)) == 0, "sync after rewind to zero");
    compare_fresh(engine, subject, &prompt, context, vocab, actual, expected, "rewind-zero");
    ds4_session_free(subject);
    free(actual); free(expected);
    ds4_tokens_free(&prompt); ds4_tokens_free(&extended);
}

static int save_row(ds4_session *session, float *logits, int vocab, FILE *output, FILE *rows,
                    int *saved, int position, int input) {
    copy_logits(session, logits, vocab);
    int top = 0;
    for (int i = 1; i < vocab; ++i) if (logits[i] > logits[top]) top = i;
    require(ds4_session_argmax(session) == top, "public argmax matches complete vocabulary");
    if (output) {
        require(fwrite(logits, sizeof(float), (size_t)vocab, output) == (size_t)vocab, "write logits");
        require(fprintf(rows, "%d\t%d\t%d\t%d\n", (*saved)++, position, input, top) > 0,
                "write logits row metadata");
    }
    return top;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s --model MODEL (--tokens IDS | --text TEXT)\n"
            "  [--backend cpu|metal] [--decode N] [--ctx N] [--chunk N] [--threads N]\n"
            "  [--chat] [--ids-out FILE] [--ids-only] [--logits FILE] [--all-logits] [--lifecycle]\n", program);
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokens_file = NULL, *text = NULL, *logits_path = NULL, *ids_path = NULL;
    ds4_backend backend = DS4_BACKEND_CPU;
    int decode = 0, context = 0, chunk = 128, threads = 4;
    bool chat = false, all_logits = false, run_lifecycle = false, ids_only = false;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help")) { usage(argv[0]); return 0; }
        if (!strcmp(arg, "--chat")) { chat = true; continue; }
        if (!strcmp(arg, "--all-logits")) { all_logits = true; continue; }
        if (!strcmp(arg, "--lifecycle")) { run_lifecycle = true; continue; }
        if (!strcmp(arg, "--ids-only")) { ids_only = true; continue; }
        require(i + 1 < argc, "missing option value");
        const char *value = argv[++i];
        if (!strcmp(arg, "--model")) model = value;
        else if (!strcmp(arg, "--tokens")) tokens_file = value;
        else if (!strcmp(arg, "--text")) text = value;
        else if (!strcmp(arg, "--logits")) logits_path = value;
        else if (!strcmp(arg, "--ids-out")) ids_path = value;
        else if (!strcmp(arg, "--decode")) decode = number(value, 0);
        else if (!strcmp(arg, "--ctx")) context = number(value, 1);
        else if (!strcmp(arg, "--chunk")) chunk = number(value, 1);
        else if (!strcmp(arg, "--threads")) threads = number(value, 1);
        else if (!strcmp(arg, "--backend")) {
            require(!strcmp(value, "cpu") || !strcmp(value, "metal"), "backend must be cpu or metal");
            backend = !strcmp(value, "cpu") ? DS4_BACKEND_CPU : DS4_BACKEND_METAL;
        } else require(0, "unknown option");
    }
    require(model && ((tokens_file != NULL) != (text != NULL)), "model and exactly one tokens/text input required");
    require(!chat || text, "--chat requires --text");
    require(!all_logits || logits_path, "--all-logits requires --logits");
    require(!ids_only || (!decode && !run_lifecycle && !logits_path), "--ids-only cannot evaluate or save logits");
    ds4_engine_options options = {.model_path = model, .backend = backend,
        .n_threads = threads, .context_size = context ? context : 256,
        .prefill_chunk = (uint32_t)chunk, .glm_mtp = false};
    ds4_engine *engine = NULL;
    require(ds4_engine_open(&engine, &options) == 0 && ds4_engine_is_bonsai(engine), "open Bonsai engine");
    const int vocab = ds4_engine_vocab_size(engine);
    require(vocab > 1, "valid model vocabulary");
    require(ds4_engine_mtp_draft_tokens(engine) == 0, "MTP disabled");
    ds4_tokens prompt = {0}, timeline = {0};
    if (tokens_file) read_ids(tokens_file, &prompt);
    else if (chat) ds4_encode_chat_prompt(engine, NULL, text, DS4_THINK_NONE, &prompt);
    else ds4_tokenize_text(engine, text, &prompt);
    require(prompt.len > 0, "nonempty prompt");
    for (int i = 0; i < prompt.len; ++i)
        require(prompt.v[i] >= 0 && prompt.v[i] < vocab, "prompt token in vocabulary");
    printf("prompt_ids");
    for (int i = 0; i < prompt.len; ++i) printf(" %d", prompt.v[i]);
    putchar('\n');
    if (ids_path) {
        FILE *ids = open_file(ids_path, "w");
        for (int i = 0; i < prompt.len; ++i) require(fprintf(ids, "%d\n", prompt.v[i]) > 0, "write prompt IDs");
        require(fclose(ids) == 0, "close prompt IDs");
    }
    if (ids_only) {
        ds4_tokens_free(&prompt); ds4_engine_close(engine);
        return 0;
    }
    const uint64_t required = (uint64_t)prompt.len + (uint64_t)decode + 2u;
    require(required <= INT_MAX, "context size overflow");
    if (!context) context = required < 256u ? 256 : (int)required;
    require(context > prompt.len && (uint64_t)context > (uint64_t)prompt.len + (uint64_t)decode,
            "context must leave room after prompt/decode");
    require(!run_lifecycle || context > (prompt.len < 4 ? prompt.len : 4) + 1, "lifecycle append needs context room");
    FILE *output = NULL, *rows = NULL;
    if (logits_path) {
        output = open_file(logits_path, "wb");
        const size_t size = strlen(logits_path) + sizeof(".rows.tsv");
        char *rows_path = malloc(size);
        require(rows_path != NULL, "allocate output path");
        snprintf(rows_path, size, "%s.rows.tsv", logits_path);
        rows = open_file(rows_path, "w"); free(rows_path);
        require(fputs("row\tposition\tinput_id\targmax\n", rows) >= 0, "write row header");
    }
    float *logits = malloc((size_t)vocab * sizeof(float));
    require(logits != NULL, "allocate vocabulary buffer");
    ds4_session *session = NULL;
    require(ds4_session_create(&session, engine, context) == 0, "create public session");
    int next = -1, saved = 0;
    const double prefill_begin = wall_seconds();
    if (all_logits) {
        for (int i = 0; i < prompt.len; ++i) {
            ds4_tokens_push(&timeline, prompt.v[i]);
            require(ds4_session_sync(session, &timeline, api_error, sizeof(api_error)) == 0, "incremental prompt sync");
            next = save_row(session, logits, vocab, output, rows, &saved, i, prompt.v[i]);
        }
    } else {
        ds4_tokens_copy(&timeline, &prompt);
        require(ds4_session_sync(session, &prompt, api_error, sizeof(api_error)) == 0, "prompt sync");
        next = save_row(session, logits, vocab, output, rows, &saved, prompt.len - 1, prompt.v[prompt.len - 1]);
    }
    const double prefill_seconds = wall_seconds() - prefill_begin;
    double decode_seconds = 0;
    check_tokens(session, &timeline);
    printf("prefill_argmax %d\n", next);
    for (int step = 0; step < decode; ++step) {
        const int input = next;
        const double step_begin = wall_seconds();
        require(ds4_session_eval(session, input, api_error, sizeof(api_error)) == 0, "greedy public decode");
        decode_seconds += wall_seconds() - step_begin;
        ds4_tokens_push(&timeline, input);
        check_tokens(session, &timeline);
        next = save_row(session, logits, vocab, output, rows, &saved, prompt.len + step, input);
        printf("decode %d input_id %d argmax %d eog %d\n", step, input, next, ds4_token_is_stop(engine, input));
    }
    fprintf(stderr, "TIMING prefill_tokens=%d prefill_s=%.6f prefill_tps=%.3f decode_tokens=%d decode_s=%.6f decode_tps=%.3f\n",
            prompt.len, prefill_seconds, prompt.len / prefill_seconds,
            decode, decode_seconds, decode_seconds > 0 ? decode / decode_seconds : 0.0);
    ds4_session_free(session); free(logits);
    if (output) { require(fclose(output) == 0, "close logits"); require(fclose(rows) == 0, "close row metadata"); }
    if (run_lifecycle) lifecycle(engine, &prompt, context, vocab);
    fprintf(stderr, "PASS Bonsai API: backend=%s prompt=%d decode=%d vocab=%d saved_logit_rows=%d\n",
            backend == DS4_BACKEND_CPU ? "cpu" : "metal", prompt.len, decode, vocab, saved);
    ds4_tokens_free(&timeline); ds4_tokens_free(&prompt); ds4_engine_close(engine);
    return 0;
}
