// Experimental resident Q4_K attn_q_b -> F16 sidecars for ROCm prefill.
//
// The native Q4_K/Q8_K TILE8 path remains the unconditional fallback.  This
// cache is deliberately resident-only and opt-in: expanding every production
// attn_q_b matrix costs 64 MiB per layer, so admission happens once, before
// prefill, with both an explicit cache budget and device-memory headroom.

enum {
    DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES = 80u,
    DS4_ROCM_Q4_K_TYPE = 12u,
    DS4_ROCM_Q4_ATTN_Q_B_IN_DIM = 1024u,
    DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM = 32768u,
};

struct rocm_q4_attn_q_b_f16_cache_entry {
    const void *model_map;
    uint64_t model_size;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t weight_type;
    __half *device_ptr;
    uint64_t f16_bytes;
    int valid;
};

struct rocm_q4_attn_q_b_f16_arena {
    __half *device_ptr;
    uint64_t bytes;
};

static rocm_q4_attn_q_b_f16_cache_entry
    g_rocm_q4_attn_q_b_f16_entries[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES];
static rocm_q4_attn_q_b_f16_arena
    g_rocm_q4_attn_q_b_f16_arenas[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES];
static uint32_t g_rocm_q4_attn_q_b_f16_entry_count;
static uint32_t g_rocm_q4_attn_q_b_f16_arena_count;
static uint64_t g_rocm_q4_attn_q_b_f16_bytes;
static uint64_t g_rocm_q4_attn_q_b_f16_generation = 1u;
static uint64_t g_rocm_q4_attn_q_b_f16_lookups;
static uint64_t g_rocm_q4_attn_q_b_f16_hits;
static uint64_t g_rocm_q4_attn_q_b_f16_misses;
static uint64_t g_rocm_q4_attn_q_b_f16_builds;
static uint64_t g_rocm_q4_attn_q_b_f16_build_failures;
static uint64_t g_rocm_q4_attn_q_b_f16_candidate_calls;
static uint64_t g_rocm_q4_attn_q_b_f16_fallbacks;
static uint64_t g_rocm_q4_attn_q_b_f16_rejects;
static int g_rocm_q4_attn_q_b_f16_hard_failure;
static int g_rocm_q4_attn_q_b_f16_pending_evict;
static pthread_mutex_t g_rocm_q4_attn_q_b_f16_cache_mu =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_rocm_q4_attn_q_b_f16_build_mu =
    PTHREAD_MUTEX_INITIALIZER;

static int rocm_q4_attn_q_b_env_value_eq(
        const char *value, size_t n, const char *literal) {
    const size_t literal_n = strlen(literal);
    if (n != literal_n) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)value[i]) !=
            tolower((unsigned char)literal[i])) {
            return 0;
        }
    }
    return 1;
}

/* Return -1 when unset, 0 for an explicit false value, and 1 otherwise.
 * In particular, VAR=0 / false / no / off must not enable an experimental
 * path merely because the variable exists. */
static int rocm_q4_attn_q_b_env_bool(const char *name) {
    const char *value = getenv(name);
    if (!value) return -1;
    while (isspace((unsigned char)*value)) value++;
    size_t n = strlen(value);
    while (n != 0u && isspace((unsigned char)value[n - 1u])) n--;
    if (n == 0u) return 1;
    if (rocm_q4_attn_q_b_env_value_eq(value, n, "1") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "true") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "yes") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "on")) {
        return 1;
    }
    if (rocm_q4_attn_q_b_env_value_eq(value, n, "0") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "false") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "no") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "off")) {
        return 0;
    }
    /* Preserve the project's traditional presence-enables behavior for
     * unknown non-empty values while still handling conventional booleans. */
    return 1;
}

static uint64_t rocm_q4_attn_q_b_env_u64(
        const char *name,
        uint64_t fallback,
        uint64_t min_value,
        uint64_t max_value) {
    const char *value = getenv(name);
    if (!value) return fallback;
    while (isspace((unsigned char)*value)) value++;
    if (!*value) return fallback;

    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || errno == ERANGE) return fallback;
    while (isspace((unsigned char)*end)) end++;
    if (*end != '\0' || parsed < min_value) return fallback;
    const uint64_t result = (uint64_t)parsed;
    return result > max_value ? max_value : result;
}

static int rocm_q4_attn_q_b_f16_enabled(void) {
    return rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_ATTN_Q_B_F16_CACHE") == 1;
}

static int rocm_q4_attn_q_b_f16_required(void) {
    return rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_ATTN_Q_B_F16_CACHE") == 1;
}

static int rocm_q4_attn_q_b_f16_disabled(void) {
    return rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_ATTN_Q_B_F16_CACHE") == 1;
}

static uint64_t rocm_q4_attn_q_b_f16_min_tokens(void) {
    return rocm_q4_attn_q_b_env_u64(
        "DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS",
        512u, 32u, UINT32_MAX);
}

static uint64_t rocm_q4_attn_q_b_f16_budget_bytes(void) {
    const uint64_t budget_mib = rocm_q4_attn_q_b_env_u64(
        "DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MB",
        3072u, 1u, 65536u);
    return budget_mib * 1048576u;
}

static int rocm_q4_attn_q_b_f16_policy_allowed(void) {
    return (rocm_q4_attn_q_b_f16_enabled() ||
            rocm_q4_attn_q_b_f16_required()) &&
           !rocm_q4_attn_q_b_f16_disabled() &&
           !g_ssd_streaming_mode &&
           !g_quality_mode &&
           !g_q8_f16_disabled_for_multi_model;
}

static int rocm_q4_attn_q_b_f16_key_equal(
        const rocm_q4_attn_q_b_f16_cache_entry *entry,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t weight_type) {
    return entry->valid &&
           entry->model_map == model_map &&
           entry->model_size == model_size &&
           entry->weight_offset == weight_offset &&
           entry->weight_bytes == weight_bytes &&
           entry->in_dim == in_dim &&
           entry->out_dim == out_dim &&
           entry->weight_type == weight_type;
}

static rocm_q4_attn_q_b_f16_cache_entry *
rocm_q4_attn_q_b_f16_find_locked(
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t weight_type) {
    for (uint32_t i = 0;
         i < DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES;
         i++) {
        rocm_q4_attn_q_b_f16_cache_entry *entry =
            &g_rocm_q4_attn_q_b_f16_entries[i];
        if (rocm_q4_attn_q_b_f16_key_equal(
                entry, model_map, model_size, weight_offset, weight_bytes,
                in_dim, out_dim, weight_type)) {
            return entry;
        }
    }
    return NULL;
}

/* On success the cache mutex remains held until the caller has enqueued the
 * GEMM that consumes the returned pointer.  Release holds the same mutex
 * across device synchronization and frees, closing the lookup/free race. */
static const __half *rocm_q4_attn_q_b_f16_acquire(
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t weight_type) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    g_rocm_q4_attn_q_b_f16_lookups++;
    /* Recheck the circuit while holding the same mutex that protects the
     * entries.  A concurrent runtime failure may open it after the caller's
     * cheap policy check but before this lookup; refusing here also keeps a
     * partially freed arena unreachable while pending eviction is retried. */
    if (g_rocm_q4_attn_q_b_f16_hard_failure) {
        g_rocm_q4_attn_q_b_f16_misses++;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        return NULL;
    }
    rocm_q4_attn_q_b_f16_cache_entry *entry =
        rocm_q4_attn_q_b_f16_find_locked(
            model_map, model_size, weight_offset, weight_bytes,
            in_dim, out_dim, weight_type);
    if (!entry) {
        g_rocm_q4_attn_q_b_f16_misses++;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        return NULL;
    }
    g_rocm_q4_attn_q_b_f16_hits++;
    return entry->device_ptr;
}

static void rocm_q4_attn_q_b_f16_release_acquired(void) {
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
}

static void rocm_q4_attn_q_b_f16_note_candidate(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    g_rocm_q4_attn_q_b_f16_candidate_calls++;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
}

static int rocm_q4_attn_q_b_f16_try_runtime_evict(void);

static int rocm_q4_attn_q_b_f16_fallback(
        int required, int rejected, int build_failure) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    g_rocm_q4_attn_q_b_f16_fallbacks++;
    if (rejected) g_rocm_q4_attn_q_b_f16_rejects++;
    if (build_failure) {
        g_rocm_q4_attn_q_b_f16_build_failures++;
        /* A submission/launch failure is backend-wide for this optional
         * specialization. Fail closed for later layers instead of retrying
         * the same hipBLAS or kernel error dozens of times per prefill. */
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        g_rocm_q4_attn_q_b_f16_pending_evict = 1;
    }
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    /* All runtime callers reach this helper after releasing the cache mutex
     * used to pin the sidecar through GEMM submission, and none holds the
     * build mutex. Try the synchronized eviction now; if HIP cannot reach a
     * safe point, pending_evict keeps the cache disabled until a later
     * lifecycle/prewarm boundary can retry. */
    if (build_failure) (void)rocm_q4_attn_q_b_f16_try_runtime_evict();
    return required ? -1 : 0;
}

static int rocm_q4_attn_q_b_f16_circuit_open(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const int open = g_rocm_q4_attn_q_b_f16_hard_failure;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    return open;
}

__device__ __forceinline__ static void
rocm_q4_attn_q_b_get_scale_min(
        uint32_t group,
        const uint8_t *scales,
        uint8_t *scale,
        uint8_t *minimum) {
    if (group < 4u) {
        *scale = scales[group] & 63u;
        *minimum = scales[group + 4u] & 63u;
    } else {
        *scale = (scales[group + 4u] & 0x0fu) |
                 ((scales[group - 4u] >> 6u) << 4u);
        *minimum = (scales[group + 4u] >> 4u) |
                   ((scales[group] >> 6u) << 4u);
    }
}

/* One block expands one canonical 256-value GGUF Q4_K block.  The output is
 * row-major [out_dim, in_dim], the same storage consumed as W^T by hipBLAS. */
__global__ static void rocm_dequant_q4_K_attn_q_b_f16_kernel(
        __half *dst,
        const cuda_block_q4_K *src,
        uint64_t block_count) {
    const uint64_t block = (uint64_t)blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (block >= block_count || i >= CUDA_QK_K) return;

    const cuda_block_q4_K *xb = src + block;
    const uint32_t group = i >> 5u;
    const uint32_t within_group = i & 31u;
    const uint32_t byte_offset = (group >> 1u) * 32u + within_group;
    const uint32_t shift = (group & 1u) * 4u;
    const uint32_t q = (xb->qs[byte_offset] >> shift) & 0x0fu;
    uint8_t scale = 0;
    uint8_t minimum = 0;
    rocm_q4_attn_q_b_get_scale_min(
        group, xb->scales, &scale, &minimum);
    const float d = __half2float(
        __ushort_as_half((unsigned short)xb->d));
    const float dmin = __half2float(
        __ushort_as_half((unsigned short)xb->dmin));
    dst[block * CUDA_QK_K + i] =
        __float2half(d * (float)scale * (float)q -
                     dmin * (float)minimum);
}

static int rocm_q4_attn_q_b_f16_desc_valid(
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *desc,
        uint64_t model_size,
        uint64_t *f16_bytes) {
    if (!desc || desc->weight_type != DS4_ROCM_Q4_K_TYPE ||
        desc->in_dim != DS4_ROCM_Q4_ATTN_Q_B_IN_DIM ||
        desc->out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM) {
        return 0;
    }
    uint64_t row_bytes = 0;
    uint64_t expected_weight_bytes = 0;
    uint64_t elems = 0;
    uint64_t expanded_bytes = 0;
    if (!cuda_u64_mul_checked(desc->in_dim / CUDA_QK_K,
                              sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(desc->out_dim, row_bytes,
                              &expected_weight_bytes) ||
        !cuda_u64_mul_checked(desc->in_dim, desc->out_dim, &elems) ||
        !cuda_u64_mul_checked(elems, sizeof(__half), &expanded_bytes)) {
        return 0;
    }
    if (desc->weight_bytes != expected_weight_bytes ||
        !cuda_model_range_fits(model_size, desc->weight_offset,
                               desc->weight_bytes)) {
        return 0;
    }
    if (f16_bytes) *f16_bytes = expanded_bytes;
    return 1;
}

static int rocm_q4_attn_q_b_f16_memory_has_room(
        uint64_t sidecar_bytes,
        uint64_t working_set_reserve_bytes,
        uint64_t *free_bytes_out,
        uint64_t *total_bytes_out,
        uint64_t *reserve_bytes_out) {
    size_t free_b = 0;
    size_t total_b = 0;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess || total_b == 0u) {
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes =
        cuda_q8_f16_cache_reserve_bytes(total_bytes);
    uint64_t required_free = 0;
    if (!cuda_u64_add_checked(sidecar_bytes, reserve_bytes,
                              &required_free) ||
        !cuda_u64_add_checked(required_free, working_set_reserve_bytes,
                              &required_free)) {
        return 0;
    }
    if (free_bytes_out) *free_bytes_out = free_bytes;
    if (total_bytes_out) *total_bytes_out = total_bytes;
    if (reserve_bytes_out) *reserve_bytes_out = reserve_bytes;
    return required_free <= free_bytes;
}

static void rocm_q4_attn_q_b_f16_clear_locked(
        int reset_stats, int reset_circuit) {
    for (uint32_t i = 0;
         i < DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES;
         i++) {
        g_rocm_q4_attn_q_b_f16_entries[i] = {};
        g_rocm_q4_attn_q_b_f16_arenas[i] = {};
    }
    g_rocm_q4_attn_q_b_f16_entry_count = 0;
    g_rocm_q4_attn_q_b_f16_arena_count = 0;
    g_rocm_q4_attn_q_b_f16_bytes = 0;
    if (reset_circuit) g_rocm_q4_attn_q_b_f16_hard_failure = 0;
    g_rocm_q4_attn_q_b_f16_pending_evict = 0;
    g_rocm_q4_attn_q_b_f16_generation++;
    if (g_rocm_q4_attn_q_b_f16_generation == 0u) {
        g_rocm_q4_attn_q_b_f16_generation = 1u;
    }
    if (reset_stats) {
        g_rocm_q4_attn_q_b_f16_lookups = 0;
        g_rocm_q4_attn_q_b_f16_hits = 0;
        g_rocm_q4_attn_q_b_f16_misses = 0;
        g_rocm_q4_attn_q_b_f16_builds = 0;
        g_rocm_q4_attn_q_b_f16_build_failures = 0;
        g_rocm_q4_attn_q_b_f16_candidate_calls = 0;
        g_rocm_q4_attn_q_b_f16_fallbacks = 0;
        g_rocm_q4_attn_q_b_f16_rejects = 0;
    }
}

/* The caller owns build_mu. Keeping this separate lets make_room serialize
 * against an unpublished build without recursively locking through the public
 * release API. cache_mu remains held across synchronization and frees, so a
 * successful lookup cannot race the last reference to its arena. */
static int rocm_q4_attn_q_b_f16_release_with_build_lock(
        int reset_circuit) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);

    if (g_rocm_q4_attn_q_b_f16_arena_count != 0u) {
        const cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b F16 cache release sync failed: %s\n",
                    cudaGetErrorString(sync_err));
            (void)cudaGetLastError();
            g_rocm_q4_attn_q_b_f16_hard_failure = 1;
            g_rocm_q4_attn_q_b_f16_pending_evict = 1;
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
            return 0;
        }
    }
    int ok = 1;
    for (uint32_t i = 0; i < g_rocm_q4_attn_q_b_f16_arena_count; i++) {
        if (g_rocm_q4_attn_q_b_f16_arenas[i].device_ptr) {
            const cudaError_t free_err =
                cudaFree(g_rocm_q4_attn_q_b_f16_arenas[i].device_ptr);
            if (free_err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "Q4 attn_q_b F16 cache free failed: %s\n",
                        cudaGetErrorString(free_err));
                (void)cudaGetLastError();
                ok = 0;
            } else {
                /* A later retry must not double-free arenas already released
                 * before another arena reported an error. */
                g_rocm_q4_attn_q_b_f16_arenas[i].device_ptr = NULL;
            }
        }
    }
    if (!ok) {
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        g_rocm_q4_attn_q_b_f16_pending_evict = 1;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        return 0;
    }
    rocm_q4_attn_q_b_f16_clear_locked(0, reset_circuit);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    return 1;
}

static int rocm_q4_attn_q_b_f16_try_runtime_evict(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(0);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

extern "C" int ds4_gpu_release_q4_attn_q_b_f16_sidecars(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(1);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

extern "C" uint64_t ds4_gpu_q4_attn_q_b_f16_cache_generation(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const uint64_t generation = g_rocm_q4_attn_q_b_f16_generation;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    return generation;
}

extern "C" int ds4_gpu_make_room_for_q4_attn_q_b_f16_session(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const uint32_t entries = g_rocm_q4_attn_q_b_f16_entry_count;
    const uint64_t bytes = g_rocm_q4_attn_q_b_f16_bytes;
    const int needs_reset =
        entries != 0u || g_rocm_q4_attn_q_b_f16_arena_count != 0u ||
        g_rocm_q4_attn_q_b_f16_hard_failure ||
        g_rocm_q4_attn_q_b_f16_pending_evict;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    if (!needs_reset) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return 1;
    }
    if (entries != 0u) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "evicting %.2f GiB of resident Q4 attn_q_b F16 sidecars "
                "before allocating another live session\n",
                (double)bytes / 1073741824.0);
    }
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(1);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

extern "C" int ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
        const void *model_map,
        uint64_t model_size,
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *descs,
        uint32_t count,
        uint32_t max_prefill_rows,
        uint64_t working_set_reserve_bytes,
        uint64_t *prepared_bytes) {
    if (prepared_bytes) *prepared_bytes = 0;
    if (!model_map || !descs || count == 0u ||
        count > DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES ||
        max_prefill_rows < 32u) {
        return 0;
    }

    const int required = rocm_q4_attn_q_b_f16_required();
    const uint64_t min_tokens = rocm_q4_attn_q_b_f16_min_tokens();
    if ((uint64_t)max_prefill_rows < min_tokens) return 0;
    if (!rocm_q4_attn_q_b_f16_policy_allowed() || !g_cublas_ready ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size) {
        return required ? -1 : 0;
    }

    uint64_t desc_f16_bytes[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES] = {0};
    for (uint32_t i = 0; i < count; i++) {
        if (!rocm_q4_attn_q_b_f16_desc_valid(
                &descs[i], model_size, &desc_f16_bytes[i])) {
            return required ? -1 : 0;
        }
    }

    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const int pending_evict = g_rocm_q4_attn_q_b_f16_pending_evict;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    /* Explicit prewarm is a quiescent session boundary. Retry an eviction
     * that could not synchronize at the original runtime error, while
     * preserving the hard circuit so this session cannot rebuild and repeat
     * the failed specialization. */
    if (pending_evict &&
        !rocm_q4_attn_q_b_f16_release_with_build_lock(0)) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }
    if (!rocm_q4_attn_q_b_f16_policy_allowed() || !g_cublas_ready ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }

    uint32_t miss_indices[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES] = {0};
    uint32_t free_slots[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES] = {0};
    uint32_t miss_count = 0;
    uint32_t free_count = 0;
    uint64_t missing_bytes = 0;
    uint64_t cached_bytes = 0;
    int hard_failure = 0;
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    cached_bytes = g_rocm_q4_attn_q_b_f16_bytes;
    hard_failure = g_rocm_q4_attn_q_b_f16_hard_failure;
    for (uint32_t i = 0;
         i < DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES;
         i++) {
        if (!g_rocm_q4_attn_q_b_f16_entries[i].valid) {
            free_slots[free_count++] = i;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        if (rocm_q4_attn_q_b_f16_find_locked(
                model_map, model_size, descs[i].weight_offset,
                descs[i].weight_bytes, descs[i].in_dim,
                descs[i].out_dim, descs[i].weight_type)) {
            continue;
        }
        int duplicate_miss = 0;
        for (uint32_t mi = 0; mi < miss_count; mi++) {
            const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *prior =
                &descs[miss_indices[mi]];
            if (prior->weight_offset == descs[i].weight_offset &&
                prior->weight_bytes == descs[i].weight_bytes &&
                prior->in_dim == descs[i].in_dim &&
                prior->out_dim == descs[i].out_dim &&
                prior->weight_type == descs[i].weight_type) {
                duplicate_miss = 1;
                break;
            }
        }
        if (duplicate_miss) continue;
        if (!cuda_u64_add_checked(missing_bytes, desc_f16_bytes[i],
                                  &missing_bytes)) {
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
            return required ? -1 : 0;
        }
        miss_indices[miss_count++] = i;
    }
    const int entry_room =
        miss_count <= free_count &&
        g_rocm_q4_attn_q_b_f16_arena_count <
            DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);

    if (hard_failure) {
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        g_rocm_q4_attn_q_b_f16_rejects++;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }
    if (miss_count == 0u) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return 1;
    }
    const uint64_t budget_bytes = rocm_q4_attn_q_b_f16_budget_bytes();
    if (!entry_room ||
        missing_bytes > (uint64_t)SIZE_MAX ||
        missing_bytes > budget_bytes ||
        cached_bytes > budget_bytes - missing_bytes) {
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        g_rocm_q4_attn_q_b_f16_rejects++;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }

    const char *weight_ptrs[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES] = {NULL};
    for (uint32_t mi = 0; mi < miss_count; mi++) {
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *desc =
            &descs[miss_indices[mi]];
        weight_ptrs[mi] = cuda_model_range_ptr(
            model_map, desc->weight_offset, desc->weight_bytes,
            "Q4 attn_q_b sidecar source");
        if (!weight_ptrs[mi]) {
            pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
            g_rocm_q4_attn_q_b_f16_build_failures++;
            g_rocm_q4_attn_q_b_f16_hard_failure = 1;
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
            return required ? -1 : 0;
        }
    }

    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t reserve_bytes = 0;
    if (!rocm_q4_attn_q_b_f16_memory_has_room(
            missing_bytes, working_set_reserve_bytes,
            &free_bytes, &total_bytes, &reserve_bytes)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b F16 prewarm skipped for device headroom: "
                "sidecars %.2f GiB + future sessions %.2f GiB + reserve "
                "%.2f GiB, free %.2f GiB of %.2f GiB\n",
                (double)missing_bytes / 1073741824.0,
                (double)working_set_reserve_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        g_rocm_q4_attn_q_b_f16_rejects++;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }

    void *arena_raw = NULL;
    cudaError_t err = cudaMalloc(&arena_raw, (size_t)missing_bytes);
    if (err != cudaSuccess || !arena_raw) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b F16 sidecar allocation failed (%.2f GiB): %s\n",
                (double)missing_bytes / 1073741824.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        g_rocm_q4_attn_q_b_f16_build_failures++;
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }

    __half *arena = (__half *)arena_raw;
    __half *sidecar_ptrs[
        DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MAX_ENTRIES] = {NULL};
    uint64_t arena_offset = 0;
    (void)cudaGetLastError();
    int launch_ok = 1;
    for (uint32_t mi = 0; mi < miss_count; mi++) {
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *desc =
            &descs[miss_indices[mi]];
        sidecar_ptrs[mi] = (__half *)((char *)arena + arena_offset);
        const uint64_t block_count =
            (desc->in_dim / CUDA_QK_K) * desc->out_dim;
        rocm_dequant_q4_K_attn_q_b_f16_kernel<<<
            (uint32_t)block_count, CUDA_QK_K>>>(
                sidecar_ptrs[mi],
                (const cuda_block_q4_K *)weight_ptrs[mi],
                block_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b F16 dequant launch failed at layer %u: %s\n",
                    desc->layer, cudaGetErrorString(err));
            launch_ok = 0;
            break;
        }
        arena_offset += desc_f16_bytes[miss_indices[mi]];
    }
    if (launch_ok) {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b F16 dequant synchronization failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            launch_ok = 0;
        }
    } else {
        (void)cudaDeviceSynchronize();
    }
    if (!launch_ok) {
        (void)cudaFree(arena);
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        g_rocm_q4_attn_q_b_f16_build_failures++;
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return required ? -1 : 0;
    }

    /* Publish only after every dequantization completed.  Until this point no
     * lookup can observe any part of the new batch. */
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const uint32_t arena_slot = g_rocm_q4_attn_q_b_f16_arena_count++;
    g_rocm_q4_attn_q_b_f16_arenas[arena_slot] = {arena, missing_bytes};
    uint32_t published = 0;
    for (uint32_t mi = 0; mi < miss_count; mi++) {
        const uint32_t slot = free_slots[mi];
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *desc =
            &descs[miss_indices[mi]];
        g_rocm_q4_attn_q_b_f16_entries[slot] = {
            model_map,
            model_size,
            desc->weight_offset,
            desc->weight_bytes,
            desc->in_dim,
            desc->out_dim,
            desc->weight_type,
            sidecar_ptrs[mi],
            desc_f16_bytes[miss_indices[mi]],
            1,
        };
        published++;
    }
    g_rocm_q4_attn_q_b_f16_entry_count += published;
    g_rocm_q4_attn_q_b_f16_bytes += missing_bytes;
    g_rocm_q4_attn_q_b_f16_builds += published;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);

    if (prepared_bytes) *prepared_bytes = missing_bytes;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "prewarmed %u resident Q4 attn_q_b F16 sidecars "
            "(%.2f GiB; cache budget %llu MiB; min batch %llu tokens)\n",
            published,
            (double)missing_bytes / 1073741824.0,
            (unsigned long long)(budget_bytes / 1048576u),
            (unsigned long long)min_tokens);
    return 1;
}
