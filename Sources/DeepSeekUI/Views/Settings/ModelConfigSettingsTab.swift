import SwiftUI
import Combine
import DeepSeekKit

/// Exposes every field of `ModelConfig` as an editable row. Changes
/// are stored as user-supplied overrides in
/// `~/Library/Application Support/<appName>/config-overrides.json`;
/// they take effect on the next model load (we don't hot-swap a
/// loaded Transformer).
///
/// Most fields are model-architectural and are auto-inferred from
/// the checkpoint at load time (`ModelConfig.inferred(from: loader)`),
/// so editing them rarely helps — but the user explicitly opted into
/// "expose all", and surfacing them is useful for benchmarking and
/// for cases where the on-disk config.json is partial.
struct ModelConfigSettingsTab: View {
    @StateObject private var model = ConfigOverridesViewModel()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Overrides apply on next model load.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset all") { model.reset() }
            }
            .padding(.horizontal)
            .padding(.top, 8)

            Form {
                Section("Memory & batching") {
                    int("max_batch_size",  $model.cfg.maxBatchSize)
                    int("max_seq_len",     $model.cfg.maxSeqLen)
                }
                Section("Quantization") {
                    string("dtype",        $model.cfg.dtype,        options: ["fp8", "bf16"])
                    optString("scale_fmt", $model.cfg.scaleFmt,     options: [nil, "ue8m0"])
                    optString("expert_dtype", $model.cfg.expertDtype, options: [nil, "fp4"])
                    string("scale_dtype",  $model.cfg.scaleDtype,   options: ["fp32", "fp8"])
                }
                Section("Vocabulary & shape") {
                    int("vocab_size",       $model.cfg.vocabSize)
                    int("dim",              $model.cfg.dim)
                    int("moe_inter_dim",    $model.cfg.moeInterDim)
                    int("n_layers",         $model.cfg.nLayers)
                    int("n_hash_layers",    $model.cfg.nHashLayers)
                    int("n_mtp_layers",     $model.cfg.nMtpLayers)
                    int("n_heads",          $model.cfg.nHeads)
                }
                Section("MoE") {
                    int("n_routed_experts",   $model.cfg.nRoutedExperts)
                    int("n_shared_experts",   $model.cfg.nSharedExperts)
                    int("n_activated_experts", $model.cfg.nActivatedExperts)
                    string("score_func",      $model.cfg.scoreFunc,
                            options: ["softmax", "sigmoid", "sqrtsoftplus"])
                    float("route_scale",      $model.cfg.routeScale)
                    float("swiglu_limit",     $model.cfg.swigluLimit)
                }
                Section("MLA / windows") {
                    int("q_lora_rank",   $model.cfg.qLoraRank)
                    int("head_dim",      $model.cfg.headDim)
                    int("rope_head_dim", $model.cfg.ropeHeadDim)
                    float("norm_eps",    $model.cfg.normEps)
                    int("o_groups",      $model.cfg.oGroups)
                    int("o_lora_rank",   $model.cfg.oLoraRank)
                    int("window_size",   $model.cfg.windowSize)
                    LabeledContent("compress_ratios") {
                        Text(model.cfg.compressRatios.map(String.init).joined(separator: ", "))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                Section("YaRN RoPE") {
                    float("compress_rope_theta", $model.cfg.compressRopeTheta)
                    int("original_seq_len",      $model.cfg.originalSeqLen)
                    float("rope_theta",          $model.cfg.ropeTheta)
                    float("rope_factor",         $model.cfg.ropeFactor)
                    int("beta_fast",             $model.cfg.betaFast)
                    int("beta_slow",             $model.cfg.betaSlow)
                }
                Section("Indexer") {
                    int("index_n_heads",   $model.cfg.indexNHeads)
                    int("index_head_dim",  $model.cfg.indexHeadDim)
                    int("index_topk",      $model.cfg.indexTopk)
                }
                Section("Hyper-Connections") {
                    int("hc_mult",           $model.cfg.hcMult)
                    int("hc_sinkhorn_iters", $model.cfg.hcSinkhornIters)
                    float("hc_eps",          $model.cfg.hcEps)
                }
            }
            .formStyle(.grouped)
        }
    }

    // ---- field helpers ----
    @ViewBuilder private func int(_ key: String, _ b: Binding<Int>) -> some View {
        LabeledContent(key) {
            TextField(key, value: b, format: .number)
                .multilineTextAlignment(.trailing)
                .frame(width: 110)
                .font(.system(.body, design: .monospaced))
        }
    }
    @ViewBuilder private func float(_ key: String, _ b: Binding<Float>) -> some View {
        LabeledContent(key) {
            TextField(key, value: b, format: .number.precision(.fractionLength(0...6)))
                .multilineTextAlignment(.trailing)
                .frame(width: 140)
                .font(.system(.body, design: .monospaced))
        }
    }
    @ViewBuilder private func string(_ key: String, _ b: Binding<String>,
                                      options: [String]) -> some View {
        LabeledContent(key) {
            Picker("", selection: b) {
                ForEach(options, id: \.self) { Text($0).tag($0) }
            }
            .labelsHidden()
            .frame(width: 140)
        }
    }
    @ViewBuilder private func optString(_ key: String, _ b: Binding<String?>,
                                         options: [String?]) -> some View {
        LabeledContent(key) {
            Picker("", selection: b) {
                ForEach(options, id: \.self) { opt in
                    Text(opt ?? "—").tag(opt)
                }
            }
            .labelsHidden()
            .frame(width: 140)
        }
    }
}

@MainActor
final class ConfigOverridesViewModel: ObservableObject {
    @Published var cfg: ModelConfig

    private let url: URL? = {
        try? PersistencePaths.conversationsDir()
            .deletingLastPathComponent()
            .appendingPathComponent("config-overrides.json")
    }()
    private var cancellable: AnyCancellable?

    init() {
        if let url = url, let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode(ModelConfig.self, from: data) {
            self.cfg = decoded
        } else {
            self.cfg = ModelConfig()
        }
        // Autosave with a half-second debounce so per-keystroke
        // typing in a TextField doesn't hammer the disk.
        cancellable = $cfg
            .dropFirst()
            .debounce(for: .milliseconds(500), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.save() }
    }

    func save() {
        guard let url = url else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(cfg) {
            try? data.write(to: url, options: .atomic)
        }
    }

    func reset() {
        cfg = ModelConfig()
    }
}
