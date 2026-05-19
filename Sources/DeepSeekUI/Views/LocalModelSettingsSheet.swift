import SwiftUI
import Combine
import DeepSeekKit

/// Sheet aperto dal model picker della toolbar che espone i pochi
/// knob runtime che la maggior parte degli utenti vuole tunare quando
/// carica un modello locale: dimensione della KV cache (`max_seq_len`),
/// batch size, e la persistenza/compressione della cache cross-restart.
///
/// I campi architetturali (n_layers, dim, n_heads, …) restano nel tab
/// completo Settings → Model Config; qui si vede solo quello che
/// muove davvero il trade-off RAM/contesto a ogni load.
///
/// Le override di `max_seq_len` e `max_batch_size` vengono scritte
/// sullo stesso `config-overrides.json` usato da
/// `ModelConfigSettingsTab` (riusiamo il suo `ConfigOverridesViewModel`),
/// così i due posti rimangono in sync. Effetto: prossimo model load,
/// salvo che l'utente clicchi "Reload model now" qui sotto.
struct LocalModelSettingsSheet: View {
    @ObservedObject var modelState: ModelState

    @Environment(\.dismiss) private var dismiss
    @StateObject private var overrides = ConfigOverridesViewModel()

    @AppStorage(AppSettingsKey.crossRestartKVCache)
    private var crossRestartKVCache: Bool = false
    @AppStorage(AppSettingsKey.kvCacheCompression)
    private var kvCacheCompression: String = "f32"
    @AppStorage(AppSettingsKey.showPrefillTrace)
    private var showPrefillTrace: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header

            Form {
                Section("Context / KV cache") {
                    LabeledContent("max_seq_len (tokens)") {
                        TextField("max_seq_len",
                                   value: $overrides.cfg.maxSeqLen,
                                   format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 120)
                            .font(.system(.body, design: .monospaced))
                    }
                    LabeledContent("max_batch_size") {
                        TextField("max_batch_size",
                                   value: $overrides.cfg.maxBatchSize,
                                   format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 120)
                            .font(.system(.body, design: .monospaced))
                    }
                    LabeledContent("Estimated KV cache RAM") {
                        Text(formatBytes(overrides.cfg.projectedKVCacheBytes))
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Text("`max_seq_len` cappa la finestra di contesto " +
                         "per conversazione ed è il driver principale " +
                         "della memoria KV cache. Abbassalo su Mac con " +
                         "RAM limitata; alzalo (con modelli YaRN/mscale) " +
                         "per abilitare sessioni più lunghe. La stima " +
                         "qui sopra usa la formula di " +
                         "`ModelConfig.projectedKVCacheBytes` con i " +
                         "default architetturali — il valore esatto " +
                         "viene ricalcolato al load sui tensori reali.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("KV cache persistence") {
                    Toggle("Persisti la KV cache tra i restart",
                            isOn: $crossRestartKVCache)
                    if crossRestartKVCache {
                        Picker("Compression",
                                selection: $kvCacheCompression) {
                            Text("F32 (lossless)").tag("f32")
                            Text("F16 (2× compression)").tag("f16")
                            Text("BF16 (2× compression, range F32)").tag("bf16")
                        }
                        .pickerStyle(.menu)
                    }
                    Text("Salva la KV cache su disco a 4 trigger " +
                         "(cold/continued/evict/shutdown). Al rientro " +
                         "in una conversation ripristina dallo " +
                         "snapshot invece di fare cold prefill. " +
                         "Comporta I/O periodico durante decode " +
                         "(throttle a 128 token o 5s).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Diagnostics") {
                    Toggle("Mostra il prompt al modello (primo turn)",
                            isOn: $showPrefillTrace)
                    Text("Sul primo messaggio di ogni conversazione " +
                         "(cold prefill, KV cache vuota) inserisce " +
                         "fra il tuo messaggio e la risposta un " +
                         "blocco grigio collassabile che mostra in " +
                         "streaming il testo completo del prompt che " +
                         "il modello sta per leggere: system message, " +
                         "blocco tools, project context, history. " +
                         "Sui turn successivi il delta è solo il " +
                         "nuovo user message — il trace viene omesso " +
                         "per non rumoreggiare.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)

            footer
        }
        .padding(20)
        .frame(width: 540, height: 600)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Local model settings").font(.title3.bold())
                Text("Tuna KV cache size, batch size e persistenza. " +
                     "Si applica al prossimo model load.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Reset") { overrides.reset() }
                .help("Riporta gli override ai default; al load i campi " +
                      "architetturali vengono comunque ricavati dal " +
                      "checkpoint.")
        }
        .padding(.bottom, 8)
    }

    private var footer: some View {
        HStack {
            Text("Tutti i campi di ModelConfig vivono in " +
                 "Settings → Model Config.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
            if case .loaded(let ep, _) = modelState.status {
                Button("Reload model") {
                    Task {
                        await modelState.load(ep)
                        await MainActor.run { dismiss() }
                    }
                }
                .help("Ricarica il modello corrente per applicare " +
                      "subito gli override.")
            }
            Button("Done") { dismiss() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(.top, 8)
    }

    private func formatBytes(_ b: UInt64) -> String {
        let kb = Double(b) / 1024
        let mb = kb / 1024
        let gb = mb / 1024
        if gb >= 1 { return String(format: "%.2f GB", gb) }
        if mb >= 1 { return String(format: "%.1f MB", mb) }
        if kb >= 1 { return String(format: "%.0f KB", kb) }
        return "\(b) B"
    }
}
