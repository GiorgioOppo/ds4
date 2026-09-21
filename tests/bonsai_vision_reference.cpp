/* Standalone Prism CPU vision reference; not linked into ds4.
 * Uses Prism's internal mtmd headers. Reference revision used for validation:
 *   9a9394a895b96003ca842a6041cb28ac49a108f7
 * Build from the ds4 root after compiling the external Prism mtmd libraries:
 *   prism_ref=/private/tmp/ds4-bonsai-reference
 *   cmake --build "$prism_ref/build" --target mtmd -j4
 *   make ds4_image.o
 *   c++ -std=c++17 -O2 -Wall -Wextra -Werror -I. \
 *     -isystem "$prism_ref/include" -isystem "$prism_ref/ggml/include" \
 *     -isystem "$prism_ref/tools/mtmd" tests/bonsai_vision_reference.cpp ds4_image.o \
 *     -L"$prism_ref/build/bin" -Wl,-rpath,"$prism_ref/build/bin" \
 *     -lmtmd -lllama -lggml -lggml-base -lm \
 *     -o /private/tmp/bonsai-vision-reference
 *
 * Native Prism preprocessing (the default, including its padding policy):
 *   /private/tmp/bonsai-vision-reference MMPROJ IMAGE /tmp/prism-native.bin
 * Kernel comparison using exactly DS4's normalized image pixels instead:
 *   /private/tmp/bonsai-vision-reference --ds4-preprocess \
 *     MMPROJ IMAGE /tmp/prism-ds4-pixels.bin
 * Compare the latter with tests/test_qwen4_vision MMPROJ IMAGE OUT 64 64.
 * The flag changes only the supplied image pixels, not the Prism graph.
 * Native Prism and DS4 may resize/pad differently even with equal output grids;
 * do not interpret their default end-to-end difference as a kernel error.
 * Prism CPU also rounds some GEMM inputs and uses tanh GELU in the merger,
 * whereas DS4 keeps FP32 inputs and uses erf GELU there: no bitwise claim.
 *
 * Both preprocessing limits are fixed at 64 tokens to keep this test small;
 * rounding may produce fewer tokens (maple.png yields 54, grid 6x9). Results
 * above 64 tokens are rejected before encoding. No GPU inference or warmup.
 * Output matches test_qwen4_vision: four little-endian uint32 values
 * [tokens, dimension, grid_height, grid_width], then row-major float32 data.
 * Requires IEEE float32 and a little-endian host, as does the DS4 dump.
 */
#include "clip.h"
#include "clip-model.h"
#include "mtmd-image.h"
#include "ds4_image.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t token_limit = 64;
static constexpr uint32_t embedding_dim = 5120;
static constexpr uint32_t patch_side = 16;

static void need(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

struct image_owner {
    ds4_image value = {};
    ~image_owner() { ds4_image_free(&value); }
};

struct patches_owner {
    ds4_image_patches value = {};
    ~patches_owner() { ds4_image_patches_free(&value); }
};

static void finite_values(const std::vector<float> &values, const char *name) {
    for (size_t i = 0; i < values.size(); ++i)
        if (!std::isfinite(values[i]))
            throw std::runtime_error(std::string(name) + " contains a nonfinite value at " + std::to_string(i));
}

static void use_ds4_pixels(clip_image_f32 &input, const ds4_image &rgb) {
    patches_owner owned;
    auto &p = owned.value;
    char error[512] = {};
    const bool preprocessed = ds4_image_preprocess_qwen4(&p, &rgb, token_limit, token_limit, error, sizeof(error));
    need(preprocessed, std::string("DS4 preprocessing failed: ") + error);
    need(p.patches && p.grid_width && p.grid_height &&
         p.grid_width % 2 == 0 && p.grid_height % 2 == 0 &&
         static_cast<uint64_t>(p.grid_width) * p.grid_height == p.patch_count &&
         p.patch_count == static_cast<uint64_t>(p.image_token_count) * 4u && p.image_token_count <= token_limit,
         "invalid DS4 patch grid or token count above the 64-token test limit");
    need(p.content_width == p.grid_width * patch_side &&
         p.content_height == p.grid_height * patch_side &&
         p.content_width == static_cast<uint32_t>(input.nx()) &&
         p.content_height == static_cast<uint32_t>(input.ny()),
         "DS4 and Prism preprocessing dimensions differ; cannot compare kernels at the same grid");
    std::vector<float> canvas(input.n_elements());
    size_t src = 0;
    // Invert DS4's [window_y][window_x][inner_y][inner_x][channel][y][x]
    // patch layout into Prism's normalized interleaved RGB image buffer.
    for (uint32_t by = 0; by < p.grid_height / 2; ++by)
        for (uint32_t bx = 0; bx < p.grid_width / 2; ++bx)
            for (uint32_t my = 0; my < 2; ++my)
                for (uint32_t mx = 0; mx < 2; ++mx)
                    for (uint32_t c = 0; c < 3; ++c)
                        for (uint32_t y = 0; y < patch_side; ++y)
                            for (uint32_t x = 0; x < patch_side; ++x) {
                                const size_t row = (by * 2 + my) * patch_side + y;
                                const size_t col = (bx * 2 + mx) * patch_side + x;
                                canvas[(row * p.content_width + col) * 3 + c] = p.patches[src++];
                            }
    need(src == canvas.size(), "DS4 patch reconstruction size mismatch");
    finite_values(canvas, "DS4 pixels");
    const auto &native = input.get_ro_buf();
    double max_diff = 0, sum_sq = 0;
    for (size_t i = 0; i < canvas.size(); ++i) {
        const double d = static_cast<double>(canvas[i]) - native[i];
        max_diff = std::max(max_diff, std::abs(d));
        sum_sq += d * d;
    }
    std::fprintf(stderr, "PREPROCESS mode=ds4 same-pixel kernel comparison; versus Prism native max=%.9g rms=%.9g\n",
                 max_diff, std::sqrt(sum_sq / canvas.size()));
    input.cpy_buf(canvas);
}

static void usage(const char *program) {
    std::fprintf(stderr, "usage: %s [--ds4-preprocess] MMPROJ IMAGE OUTPUT\n"
                         "  default: native Prism preprocessing; optional flag: DS4 pixels for kernel comparison\n"
                         "  CPU only, min/max image tokens fixed at 64, output dimension 5120\n", program);
}

int main(int argc, char **argv) {
    try {
        bool ds4_preprocess = false;
        std::vector<const char *> paths;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
            if (!std::strcmp(argv[i], "--ds4-preprocess")) {
                need(!ds4_preprocess, "duplicate --ds4-preprocess option");
                ds4_preprocess = true;
            } else {
                need(std::strncmp(argv[i], "--", 2) != 0, std::string("unknown option: ") + argv[i]);
                paths.push_back(argv[i]);
            }
        }
        if (paths.size() != 3) { usage(argv[0]); return 2; }
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
        const uint32_t endian = 1;
        need(*reinterpret_cast<const unsigned char *>(&endian) == 1, "little-endian host required for DS4 dump format");

        ggml_backend_load_all();
        clip_context_params params = {};
        params.use_gpu = false;
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        params.image_min_tokens = token_limit;
        params.image_max_tokens = token_limit;
        params.warmup = false;
        const auto initialized = clip_init(paths[0], params);
        using context_ptr = std::unique_ptr<clip_ctx, decltype(&clip_free)>;
        context_ptr vision(initialized.ctx_v, clip_free);
        context_ptr audio(initialized.ctx_a, clip_free);
        context_ptr generated_audio(initialized.ctx_gen_a, clip_free);
        need(vision != nullptr, "Prism could not load a vision context");
        const auto *hp = clip_get_hparams(vision.get());
        need(hp && clip_get_projector_type(vision.get()) == PROJECTOR_TYPE_QWEN3VL &&
             hp->patch_size == 16 && hp->n_merge == 2 && hp->n_embd == 1152 &&
             hp->n_head == 16 && hp->n_layer == 27 &&
             clip_n_mmproj_embd(vision.get()) == static_cast<int>(embedding_dim),
             "expected Bonsai Qwen3-VL mmproj: patch16, merge2, embd1152, heads16, layers27, output5120");

        image_owner owned_image;
        auto &rgb = owned_image.value;
        char error[512] = {};
        const bool decoded = ds4_image_decode_file(&rgb, paths[1], error, sizeof(error));
        need(decoded, std::string("image decode failed: ") + error);
        need(rgb.rgb && rgb.width && rgb.height && rgb.width <= DS4_IMAGE_MAX_DIMENSION &&
             rgb.height <= DS4_IMAGE_MAX_DIMENSION &&
             static_cast<uint64_t>(rgb.width) * rgb.height <= DS4_IMAGE_MAX_PIXELS,
             "invalid decoded image dimensions");
        clip_image_u8 image;
        image.set_size({static_cast<int>(rgb.width), static_cast<int>(rgb.height)}, false);
        image.cpy_buf(std::vector<uint8_t>(rgb.rgb, rgb.rgb + static_cast<size_t>(rgb.width) * rgb.height * 3));
        mtmd_image_preprocessor_dyn_size preprocessor(vision.get());
        auto prepared = preprocessor.preprocess(image);
        need(prepared.entries.size() == 1, "expected exactly one preprocessed image");
        auto &input = prepared.entries[0];
        need(input.nx() > 0 && input.ny() > 0 && input.nx() % 32 == 0 && input.ny() % 32 == 0 &&
             static_cast<uint64_t>(input.nx()) * input.ny() <= token_limit * 32u * 32u &&
             input.get_ro_buf().size() == input.n_elements(),
             "invalid Prism image dimensions or image above the 64-token test limit");
        finite_values(input.get_ro_buf(), "Prism pixels");
        if (ds4_preprocess) use_ds4_pixels(input, rgb);
        else std::fprintf(stderr, "PREPROCESS mode=prism-native (Prism resize and padding)\n");

        const int tokens = clip_n_output_tokens(vision.get(), &input);
        const int gh = clip_n_output_tokens_y(vision.get(), &input);
        const int gw = clip_n_output_tokens_x(vision.get(), &input);
        need(tokens > 0 && tokens <= static_cast<int>(token_limit) && gh == input.ny() / 32 &&
             gw == input.nx() / 32 && static_cast<int64_t>(gh) * gw == tokens,
             "inconsistent Prism output token count/grid");
        std::vector<float> output(static_cast<size_t>(tokens) * embedding_dim);
        need(clip_image_encode(vision.get(), 4, &input, output), "Prism CPU encoding failed");
        need(output.size() == static_cast<size_t>(tokens) * embedding_dim, "Prism output size changed during encoding");
        finite_values(output, "Prism output");
        const uint32_t header[4] = {static_cast<uint32_t>(tokens), embedding_dim,
                                    static_cast<uint32_t>(gh), static_cast<uint32_t>(gw)};
        std::ofstream file(paths[2], std::ios::binary | std::ios::trunc);
        need(static_cast<bool>(file), std::string("cannot open output: ") + paths[2]);
        file.write(reinterpret_cast<const char *>(header), sizeof(header));
        file.write(reinterpret_cast<const char *>(output.data()),
                   static_cast<std::streamsize>(output.size() * sizeof(float)));
        need(static_cast<bool>(file), "cannot write complete output");
        file.close();
        need(static_cast<bool>(file), "cannot close output");
        std::fprintf(stderr, "PASS Prism CPU vision reference preprocess=%s tokens=%u dim=%u grid=%ux%u output=%s\n",
                     ds4_preprocess ? "ds4" : "prism-native", header[0], header[1], header[2], header[3], paths[2]);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "bonsai-vision-reference: %s\n", error.what());
        return 1;
    }
}
