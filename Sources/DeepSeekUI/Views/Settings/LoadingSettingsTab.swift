import SwiftUI
import AppKit
import DeepSeekKit

/// Pre-flight knobs: strategy override + force-load + recent dirs +
/// converter binary path. Changes affect the next model load /
/// conversion, not the current session.
struct LoadingSettingsTab: View {
    @AppStorage(AppSettingsKey.loadStrategy) private var loadStrategy: String = "auto"
    @AppStorage(AppSettingsKey.forceLoad)    private var forceLoad: Bool = false
    @AppStorage(AppSettingsKey.lastModelDir) private var lastModelDir: String = ""
    @AppStorage(AppSettingsKey.converterBinaryPath)
    private var converterPath: String = ""
    @AppStorage(AppSettingsKey.warmupOnLoad) private var warmupOnLoad: Bool = false
    @AppStorage(AppSettingsKey.lazyExpertLoad) private var lazyExpertLoad: Bool = true
    @AppStorage(AppSettingsKey.commonPrefixRewind)
    private var commonPrefixRewind: Bool = false
    @AppStorage(AppSettingsKey.useMapSharedWeights)
    private var useMapSharedWeights: Bool = false
    @AppStorage(AppSettingsKey.crossRestartKVCache)
    private var crossRestartKVCache: Bool = false
    @AppStorage(AppSettingsKey.kvCacheCompression)
    private var kvCacheCompression: String = "f32"
    @AppStorage(AppSettingsKey.precomputedToolPrefix)
    private var precomputedToolPrefix: Bool = true
    @AppStorage(AppSettingsKey.overrideActiveExperts)
    private var overrideActiveExperts: Bool = false
    @AppStorage(AppSettingsKey.activeExpertsPerToken)
    private var activeExpertsPerToken: Int = 8

    var body: some View {
        Form {
            Section("Strategy") {
                Picker("Load strategy", selection: $loadStrategy) {
                    Text("Auto").tag("auto")
                    Text("Preload").tag("preload")
                    Text("Mmap").tag("mmap")
                    Text("Streaming").tag("streaming")
                }
                .pickerStyle(.segmented)
                Toggle("Bypass RAM-safety refusals (--force-load)",
                        isOn: $forceLoad)
                Text("Auto picks streaming when the checkpoint is more than 10× the effective unified-memory budget — between-layer madvise hints keep the working set bounded so the OS doesn't freeze. Force-load skips the 50% shard cap entirely. Preload copies everything to a fresh MTLBuffer (rare on small Macs).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Performance tweaks (ds4-inspired)") {
                Toggle("Lazy-expert load (EXPERIMENTAL, streaming MoE only)",
                        isOn: $lazyExpertLoad)
                    .onChange(of: lazyExpertLoad) { _, newValue in
                        // Push the new value to DeepSeekKit so the
                        // change takes effect on the very next layer
                        // pread without needing a model reload.
                        StreamingPool.lazyExpertEnabled = newValue
                    }
                Text("⚠️ Sperimentale, default OFF. Su strategy " +
                     "`streaming` carica per token solo i tensor " +
                     "non-expert del layer (attention, norms, gate, " +
                     "shared expert) e poi pread on-demand degli " +
                     "expert attivi indicati dal gate (~8/256 in " +
                     "V4-Pro). Sulla carta taglia l'I/O per token " +
                     "di ~7-15× su checkpoint con oversubscription " +
                     "elevata (148 GB su 16 GB RAM) ed evita il " +
                     "watchdog `Impacting Interactivity`. " +
                     "**Regressione nota su V4-Pro**: produce logits " +
                     "degeneri (loop di `<|begin_of_sentence|>`). " +
                     "Attiva solo se vuoi aiutare a fare bisect. " +
                     "Override env: `DEEPSEEK_LAZY_EXPERT=1` forza ON.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Override active experts per token",
                        isOn: $overrideActiveExperts)
                    .onChange(of: overrideActiveExperts) { _, _ in
                        ModelConfig.activeExpertsOverride =
                            AppSettings.activeExpertsOverride
                    }
                if overrideActiveExperts {
                    Stepper("Experts per token: \(activeExpertsPerToken)",
                             value: $activeExpertsPerToken, in: 1...16)
                        .onChange(of: activeExpertsPerToken) { _, _ in
                            ModelConfig.activeExpertsOverride =
                                AppSettings.activeExpertsOverride
                        }
                }
                Text("Quanti expert il gate attiva per token sui layer " +
                     "a routing appreso (i primi n_hash_layers restano " +
                     "al K addestrato). Default 8, tetto del kernel 16. " +
                     "Si applica al prossimo caricamento del modello. " +
                     "`DEEPSEEK_TOPK_EXPERTS=N` ha la precedenza.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Warm pages on load",
                        isOn: $warmupOnLoad)
                Text("Pre-fault tutte le pagine dei weight shards al " +
                     "model load. Riduce il time-to-first-token. " +
                     "Skip automatico se model size > RAM × 1.5 " +
                     "(safe su Mac con memoria limitata).")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Common-prefix rewind (experimental)",
                        isOn: $commonPrefixRewind)
                Text("Riusa la KV cache anche quando l'utente edita " +
                     "il proprio ultimo messaggio (common-prefix " +
                     "match invece di strict-prefix). Reset esplicito " +
                     "di scoreState al window boundary (round-down a " +
                     "LCM dei compressRatio del modello). " +
                     "Default OFF; abilita solo dopo testing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("MAP_SHARED for Metal weights (experimental)",
                        isOn: $useMapSharedWeights)
                Text("Usa `MAP_SHARED` invece di `MAP_PRIVATE` per " +
                     "l'mmap dei weight shards. Su Apple Silicon + " +
                     "APFS dovrebbe permettere zero-copy MTLBuffer " +
                     "wrap (come ds4). Fallback automatico a " +
                     "MAP_PRIVATE se mmap fallisce. Default OFF " +
                     "per safety vs Darwin VM panic.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Cross-restart KV cache persistence",
                        isOn: $crossRestartKVCache)
                Text("Salva la KV cache su disco a 4 trigger " +
                     "(cold/continued/evict/shutdown). Al rientro " +
                     "in una conversation, ripristina dallo snapshot " +
                     "invece di fare cold prefill. Comporta I/O " +
                     "periodico durante decode (throttle a 128 token " +
                     "o 5s). Default OFF.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if crossRestartKVCache {
                    Picker("KV cache compression",
                            selection: $kvCacheCompression) {
                        Text("F32 (lossless)").tag("f32")
                        Text("F16 (2× compression, half precision)").tag("f16")
                        Text("BF16 (2× compression, F32 range)").tag("bf16")
                    }
                    .pickerStyle(.menu)
                    Text("F16 raccomandato: 2× file più piccolo " +
                         "senza perdita percettiva. BF16 utile se " +
                         "le attivazioni hanno range estremo (>65k).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Toggle("Precompute tool prefix",
                        isOn: $precomputedToolPrefix)
                Text("Precompila una volta sola, al model load, il " +
                     "prefisso deterministico delle chat (BOS + " +
                     "blocco tool): token + snapshot KV-cache, salvati " +
                     "in `tool-prefix/`. Ogni chat nuova salta la " +
                     "ri-tokenizzazione e il prefill di quel blocco. " +
                     "Costa qualche centinaio di MB (snapshot KV " +
                     "tenuto in RAM). Default ON; ha effetto al " +
                     "prossimo model load.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Last loaded model") {
                if lastModelDir.isEmpty {
                    Text("None yet.")
                        .foregroundStyle(.secondary)
                } else {
                    HStack {
                        Text(lastModelDir)
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Forget") { lastModelDir = "" }
                    }
                }
            }
            Section("Converter binary") {
                LabeledContent("Path") {
                    HStack {
                        TextField("Auto-detect", text: $converterPath)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                            .frame(minWidth: 240)
                        Button("Choose…") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = true
                            panel.canChooseDirectories = false
                            panel.allowsMultipleSelection = false
                            panel.prompt = "Select"
                            if panel.runModal() == .OK, let u = panel.url {
                                converterPath = u.path
                            }
                        }
                    }
                }
                Text("Used by Convert model… When empty, the runner searches Bundle Resources, the running executable's sibling dir, `.build/{debug,release}/converter`, and `/usr/local/bin/converter` in that order.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Recent model folders") {
                if recents.isEmpty {
                    Text("No recents.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(recents, id: \.self) { path in
                        HStack {
                            Text(path)
                                .font(.system(.body, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button("Forget") {
                                AppSettings.forgetRecentDir(path)
                                recents = AppSettings.recentDirs()
                            }
                            .controlSize(.small)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear { recents = AppSettings.recentDirs() }
    }

    @State private var recents: [String] = AppSettings.recentDirs()
}
