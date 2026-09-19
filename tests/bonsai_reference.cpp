/* Standalone Prism llama.cpp oracle; intentionally not linked into ds4.
 * Build after the reference libraries have been compiled:
 *   ref=/private/tmp/ds4-bonsai-reference
 *   c++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$ref/include" \
 *     -I"$ref/ggml/include" tests/bonsai_reference.cpp -L"$ref/build/bin" \
 *     -Wl,-rpath,"$ref/build/bin" -lllama -lggml -lggml-base \
 *     -o /private/tmp/ds4-bonsai/bonsai-reference
 *
 * Example (loads one model; run only when GPU/model testing is coordinated):
 *   bonsai-reference --model MODEL --tokens prompt.ids --backend cpu \
 *     --decode 0 --logits /tmp/ref.f32 --all-logits --dump-prefix /tmp/ref \
 *     --dump model.input_embed --dump attn_norm-0 --dump z-0 \
 *     --dump linear_attn_qkv_mixed-0 --dump linear_attn_out-0 \
 *     --dump ffn_out-0 --dump l_out-0 --dump result_norm --dump result_output
 *
 * --tokens is whitespace-separated decimal token IDs, with no added tokens.
 * --text is literal text, with model special-token insertion enabled unless
 * --no-special is supplied. --decode N greedily evaluates exactly N tokens,
 * including EOG if selected; zero is valid. No sampler or MTP is used.
 * --logits writes native float32 rows plus PATH.rows.tsv (row, position,
 * input ID, argmax). Normally rows are the prefill frontier and N decodes;
 * --all-logits additionally captures every prompt position. --chunk controls
 * both logical and physical batch limits. Compare identical token sequences.
 * --dump requests exact graph tensor names and forces observation only for
 * those names. Each occurrence becomes PREFIX.callNNNN.NAME.K.f32; metadata
 * is PREFIX.tensors.tsv. Dumps are packed in ggml dimension-0-fastest order,
 * converting F16/BF16 to F32. Block 0 is GDN: its attention output is named
 * linear_attn_out-0, not attn_output-0 (full attention starts at block 3).
 */
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool ok, const std::string & why) {
    if (!ok) throw std::runtime_error(why);
}

static int number(const std::string & s, int minimum) {
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(s.c_str(), &end, 10);
    need(!errno && !s.empty() && end && !*end && n >= minimum && n <= INT32_MAX,
         "invalid integer: " + s);
    return static_cast<int>(n);
}

static void write_f32(std::ostream & file, const float * values, size_t count) {
    file.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(count * sizeof(float)));
    need(static_cast<bool>(file), "write float32 output failed");
}

struct capture {
    std::set<std::string> requested;
    std::map<std::string, size_t> seen;
    std::string prefix, error;
    std::ofstream metadata;
    int call = 0;

    static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
        auto & self = *static_cast<capture *>(opaque);
        const std::string name = ggml_get_name(tensor);
        if (ask) return self.requested.count(name) != 0;
        try {
            const size_t occurrence = self.seen[name]++;
            need(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 ||
                 tensor->type == GGML_TYPE_BF16, "non-floating capture: " + name);
            std::vector<unsigned char> raw(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
            std::vector<float> packed(static_cast<size_t>(ggml_nelements(tensor)));
            size_t dst = 0;
            for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3)
                for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2)
                    for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1)
                        for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                            const size_t offset = i0 * tensor->nb[0] + i1 * tensor->nb[1] +
                                                  i2 * tensor->nb[2] + i3 * tensor->nb[3];
                            float value;
                            if (tensor->type == GGML_TYPE_F32) {
                                need(offset + sizeof(value) <= raw.size(), "capture F32 view bounds");
                                std::memcpy(&value, raw.data() + offset, sizeof(value));
                            } else {
                                need(offset + 2 <= raw.size(), "capture F16/BF16 view bounds");
                                if (tensor->type == GGML_TYPE_F16) {
                                    ggml_fp16_t half;
                                    std::memcpy(&half, raw.data() + offset, sizeof(half));
                                    value = ggml_fp16_to_fp32(half);
                                } else {
                                    ggml_bf16_t bf;
                                    std::memcpy(&bf, raw.data() + offset, sizeof(bf));
                                    value = ggml_bf16_to_fp32(bf);
                                }
                            }
                            need(std::isfinite(value), "nonfinite capture: " + name);
                            packed[dst++] = value;
                        }
            std::string safe = name;
            for (char & c : safe)
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_')) c = '_';
            std::ostringstream path;
            path << self.prefix << ".call" << std::setw(4) << std::setfill('0') << self.call
                 << '.' << safe << '.' << occurrence << ".f32";
            std::ofstream file(path.str(), std::ios::binary);
            need(static_cast<bool>(file), "open tensor dump: " + path.str());
            write_f32(file, packed.data(), packed.size());
            file.close();
            need(static_cast<bool>(file), "close tensor dump: " + path.str());
            self.metadata << self.call << '\t' << name << '\t' << ggml_type_name(tensor->type);
            for (int d = 0; d < 4; ++d) self.metadata << '\t' << tensor->ne[d];
            for (int d = 0; d < 4; ++d) self.metadata << '\t' << tensor->nb[d];
            self.metadata << '\t' << path.str() << '\n';
            need(static_cast<bool>(self.metadata), "write tensor metadata");
            return true;
        } catch (const std::exception & e) {
            self.error = e.what();
            return false;
        }
    }
};

static void usage(const char * program) {
    std::cerr << "usage: " << program << " --model FILE (--tokens IDS | --text TEXT)\n"
        "  [--backend cpu|metal] [--decode N] [--chunk N] [--ctx N] [--threads N]\n"
        "  [--logits FILE] [--all-logits] [--no-special] [--kv f16|f32]\n"
        "  [--flash-attn auto|on|off] [--dump-prefix PREFIX --dump NAME ...]\n";
}

int main(int argc, char ** argv) {
    try {
        std::string model_path, token_path, text, backend = "cpu", logits_path;
        int decode = 0, chunk = 128, context = 0, threads = 4;
        bool have_text = false, all_logits = false, add_special = true;
        ggml_type kv_type = GGML_TYPE_F16;
        llama_flash_attn_type flash = LLAMA_FLASH_ATTN_TYPE_AUTO;
        capture dumps;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() { need(i + 1 < argc, "missing value for " + arg); return std::string(argv[++i]); };
            if (arg == "--help") { usage(argv[0]); return 0; }
            else if (arg == "--model") model_path = value();
            else if (arg == "--tokens") token_path = value();
            else if (arg == "--text") { text = value(); have_text = true; }
            else if (arg == "--backend") backend = value();
            else if (arg == "--decode") decode = number(value(), 0);
            else if (arg == "--chunk") chunk = number(value(), 1);
            else if (arg == "--ctx") context = number(value(), 1);
            else if (arg == "--threads") threads = number(value(), 1);
            else if (arg == "--logits") logits_path = value();
            else if (arg == "--all-logits") all_logits = true;
            else if (arg == "--no-special") add_special = false;
            else if (arg == "--dump-prefix") dumps.prefix = value();
            else if (arg == "--dump") dumps.requested.insert(value());
            else if (arg == "--kv") {
                const std::string kind = value();
                need(kind == "f16" || kind == "f32", "--kv expects f16 or f32");
                kv_type = kind == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
            } else if (arg == "--flash-attn") {
                const std::string mode = value();
                need(mode == "auto" || mode == "on" || mode == "off", "invalid flash attention mode");
                flash = mode == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                        mode == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else throw std::runtime_error("unknown argument: " + arg);
        }
        need(!model_path.empty() && (have_text != !token_path.empty()), "provide model and exactly one token/text input");
        need(backend == "cpu" || backend == "metal", "--backend expects cpu or metal");
        need(!all_logits || !logits_path.empty(), "--all-logits requires --logits");
        need(dumps.requested.empty() == dumps.prefix.empty(), "--dump and --dump-prefix must be used together");
        if (!dumps.prefix.empty()) {
            dumps.metadata.open(dumps.prefix + ".tensors.tsv");
            need(static_cast<bool>(dumps.metadata), "open tensor metadata");
            dumps.metadata << "call\tname\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tfile\n";
        }
        std::vector<llama_token> tokens;
        if (!token_path.empty()) {
            std::ifstream input(token_path);
            need(static_cast<bool>(input), "open token file: " + token_path);
            std::string token;
            while (input >> token) tokens.push_back(number(token, 0));
            need(input.eof(), "read token file failed");
        }

        ggml_backend_load_all();
        llama_backend_init();
        struct backend_guard { ~backend_guard() { llama_backend_free(); } } backend_lifetime;
        ggml_backend_dev_t devices[2] = {nullptr, nullptr};
        if (backend == "metal") {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t device = ggml_backend_dev_get(i);
                const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
                if (reg == "MTL" || reg == "Metal") { devices[0] = device; break; }
            }
            need(devices[0], "Metal backend unavailable");
        }
        llama_model_params mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = backend == "metal" ? -1 : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        need(static_cast<bool>(model), "load reference model");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        if (have_text) {
            need(text.size() <= INT32_MAX, "text exceeds tokenizer limit");
            const int count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                              nullptr, 0, add_special, true);
            need(count != INT32_MIN && count <= 0, "tokenizer sizing failed");
            tokens.resize(static_cast<size_t>(-count));
            const int actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                tokens.data(), static_cast<int32_t>(tokens.size()), add_special, true);
            need(actual >= 0 && actual <= static_cast<int>(tokens.size()), "tokenize prompt");
            tokens.resize(static_cast<size_t>(actual));
        }
        need(!tokens.empty() && tokens.size() <= INT32_MAX, "prompt must contain 1..INT32_MAX tokens");
        for (llama_token token : tokens) need(token < n_vocab, "token ID exceeds vocabulary");
        const uint64_t required = tokens.size() + static_cast<uint64_t>(decode);
        need(required <= INT32_MAX, "context size overflow");
        if (!context) context = static_cast<int>(std::max<uint64_t>(256, required));
        need(static_cast<uint64_t>(context) >= required, "context too short for prompt and decode");
        chunk = std::min(chunk, context);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(context);
        cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(chunk);
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = kv_type;
        cp.flash_attn_type = flash;
        cp.offload_kqv = cp.op_offload = backend == "metal";
        cp.no_perf = true;
        if (!dumps.requested.empty()) { cp.cb_eval = capture::callback; cp.cb_eval_user_data = &dumps; }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        need(static_cast<bool>(ctx), "create reference context");
        struct batch_guard {
            llama_batch batch;
            ~batch_guard() { llama_batch_free(batch); }
        } bg{llama_batch_init(chunk, 0, 1)};
        llama_batch & batch = bg.batch;
        need(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits, "allocate batch");
        std::ofstream logits, rows;
        if (!logits_path.empty()) {
            logits.open(logits_path, std::ios::binary);
            rows.open(logits_path + ".rows.tsv");
            need(logits && rows, "open logits output");
            rows << "row\tposition\tinput_id\targmax\n";
        }
        std::cout << "prompt_ids";
        for (llama_token token : tokens) std::cout << ' ' << token;
        std::cout << '\n';
        size_t saved = 0;
        auto save_row = [&](int index, int position, llama_token input) {
            const float * values = llama_get_logits_ith(ctx.get(), index);
            need(values, "missing logits at batch row " + std::to_string(index));
            int top = 0;
            for (int i = 0; i < n_vocab; ++i) {
                need(std::isfinite(values[i]), "nonfinite vocabulary logit");
                if (values[i] > values[top]) top = i;
            }
            if (logits.is_open()) {
                write_f32(logits, values, static_cast<size_t>(n_vocab));
                rows << saved++ << '\t' << position << '\t' << input << '\t' << top << '\n';
                need(static_cast<bool>(rows), "write logits row index");
            }
            return static_cast<llama_token>(top);
        };
        double decode_seconds = 0;
        auto evaluate = [&](const llama_token * input, int count, int position, bool prompt) {
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = input[i]; batch.pos[i] = position + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = all_logits || !prompt || position + i + 1 == static_cast<int>(tokens.size());
            }
            const auto begin = std::chrono::steady_clock::now();
            const int status = llama_decode(ctx.get(), batch);
            llama_synchronize(ctx.get());
            if (!prompt) decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            need(dumps.error.empty(), dumps.error);
            need(status == 0, "llama_decode status " + std::to_string(status));
            llama_token next = -1;
            for (int i = 0; i < count; ++i)
                if (batch.logits[i]) next = save_row(i, position + i, input[i]);
            ++dumps.call;
            return next;
        };
        llama_token next = -1;
        const auto prefill_begin = std::chrono::steady_clock::now();
        for (size_t pos = 0; pos < tokens.size();) {
            const int count = static_cast<int>(std::min<size_t>(chunk, tokens.size() - pos));
            next = evaluate(tokens.data() + pos, count, static_cast<int>(pos), true);
            pos += static_cast<size_t>(count);
        }
        const double prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_begin).count();
        std::cout << "prefill_argmax " << next << '\n';
        for (int step = 0; step < decode; ++step) {
            const llama_token input = next;
            next = evaluate(&input, 1, static_cast<int>(tokens.size()) + step, false);
            std::cout << "decode " << step << " input_id " << input << " argmax " << next
                      << " eog " << llama_vocab_is_eog(vocab, input) << '\n';
        }
        std::cerr << std::fixed << std::setprecision(6)
                  << "TIMING prefill_tokens=" << tokens.size() << " prefill_s=" << prefill_seconds
                  << " prefill_tps=" << tokens.size() / prefill_seconds
                  << " decode_tokens=" << decode << " decode_s=" << decode_seconds
                  << " decode_tps=" << (decode_seconds > 0 ? decode / decode_seconds : 0) << '\n';
        if (logits.is_open()) {
            logits.close(); rows.close();
            need(logits && rows, "close logits output");
        }
        for (const auto & name : dumps.requested)
            if (!dumps.seen.count(name)) std::cerr << "warning: requested tensor not observed: " << name << '\n';
        if (dumps.metadata.is_open()) {
            dumps.metadata.close(); need(static_cast<bool>(dumps.metadata), "close tensor metadata");
        }
        std::cerr << "PASS reference: backend=" << backend << " prompt=" << tokens.size()
                  << " decode=" << decode << " chunk=" << chunk << " vocab=" << n_vocab
                  << " saved_logit_rows=" << saved << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "bonsai-reference: " << e.what() << '\n';
        return 1;
    }
}
