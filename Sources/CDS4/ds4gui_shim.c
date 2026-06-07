#include "ds4gui_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The 19 Metal kernel files the engine concatenates at runtime, keyed by the
 * DS4_METAL_*_SOURCE override the engine looks up (see ds4_metal.m). Keep this
 * list in sync with required_sources in ds4_gpu_full_source(). */
static const char *const k_metal_overrides[][2] = {
    {"DS4_METAL_FLASH_ATTN_SOURCE", "flash_attn.metal"},
    {"DS4_METAL_DENSE_SOURCE",      "dense.metal"},
    {"DS4_METAL_MOE_SOURCE",        "moe.metal"},
    {"DS4_METAL_DSV4_HC_SOURCE",    "dsv4_hc.metal"},
    {"DS4_METAL_UNARY_SOURCE",      "unary.metal"},
    {"DS4_METAL_DSV4_KV_SOURCE",    "dsv4_kv.metal"},
    {"DS4_METAL_DSV4_ROPE_SOURCE",  "dsv4_rope.metal"},
    {"DS4_METAL_DSV4_MISC_SOURCE",  "dsv4_misc.metal"},
    {"DS4_METAL_ARGSORT_SOURCE",    "argsort.metal"},
    {"DS4_METAL_CPY_SOURCE",        "cpy.metal"},
    {"DS4_METAL_CONCAT_SOURCE",     "concat.metal"},
    {"DS4_METAL_GET_ROWS_SOURCE",   "get_rows.metal"},
    {"DS4_METAL_SUM_ROWS_SOURCE",   "sum_rows.metal"},
    {"DS4_METAL_SOFTMAX_SOURCE",    "softmax.metal"},
    {"DS4_METAL_REPEAT_SOURCE",     "repeat.metal"},
    {"DS4_METAL_GLU_SOURCE",        "glu.metal"},
    {"DS4_METAL_NORM_SOURCE",       "norm.metal"},
    {"DS4_METAL_BIN_SOURCE",        "bin.metal"},
    {"DS4_METAL_SET_ROWS_SOURCE",   "set_rows.metal"},
};

void ds4gui_set_minimum_ram_mode(int enabled) {
    if (!enabled) return;
    /* These flags are honored by the engine as "set => active", value is
     * unused. Setting them with overwrite=1 makes the choice explicit even
     * when the user's shell defined them differently. */
    setenv("DS4_METAL_NO_RESIDENCY", "1", 1);
    setenv("DS4_METAL_NO_MODEL_WARMUP", "1", 1);
    /* Force the prefill to use the canonical layer-major path instead of the
     * "decode-style streamed prefill" that fires for short prompts. The
     * latter has tighter assumptions about pre-mapped non-routed weights and
     * surfaces "model range not covered by mapped model views" on tight-RAM
     * SSD streaming setups. The classical path lazy-maps each layer cleanly
     * via metal_graph_stream_map_layer() inside metal_graph_prefill_*. */
    setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);

    /* THE key tight-RAM decode setting. By default streamed decode uses a
     * "static decode map" that maps every layer's non-routed weights into
     * Metal model views at once (fast, but needs them all mappable). On a
     * 16 GB machine with a 100+ GB model those ranges cannot all be mapped,
     * so the first decode step hits "model range not covered by mapped model
     * views" and fails. Disabling it makes the engine map one layer at a time
     * during decode (metal_graph_stream_map_layer_decode), keeping the decode
     * resident set near a single layer. Slower per token, but it actually
     * runs. This is the decode-side analogue of the layer-major prefill above. */
    setenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP", "1", 1);
}

int ds4gui_set_metal_source_dir(const char *dir) {
    if (!dir || !dir[0]) return -1;

    size_t dirlen = strlen(dir);
    int has_slash = (dirlen > 0 && dir[dirlen - 1] == '/');
    size_t n = sizeof(k_metal_overrides) / sizeof(k_metal_overrides[0]);

    for (size_t i = 0; i < n; i++) {
        const char *var = k_metal_overrides[i][0];
        const char *file = k_metal_overrides[i][1];
        char path[4096];
        int written = snprintf(path, sizeof(path), "%s%s%s",
                               dir, has_slash ? "" : "/", file);
        if (written < 0 || (size_t)written >= sizeof(path)) return -1;
        setenv(var, path, 1);
    }
    return 0;
}
