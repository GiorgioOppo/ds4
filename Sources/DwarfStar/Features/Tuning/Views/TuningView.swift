import SwiftUI
import DS4Engine

/// Runtime-tuning panel: expert slot-cache configuration ("persistent + changing
/// experts") and the routing-usage profile (the "usage imatrix") that pre-warms
/// it. Weight-level fine-tuning is NOT possible on-device (see the note below).
struct TuningView: View {
    @Bindable var store: ChatStore

    @ViewBuilder var body: some View {
        if store.supportsExpertTuning {
            VStack(spacing: 0) {
            Form {
                Section {
                    Label("Weight fine-tuning is not possible on-device: the engine is inference-only (no backward pass), and quantized 2-bit weights are not trainable. This tab tunes runtime behavior: which experts stay resident in RAM and the usage profile that selects them.",
                          systemImage: "info.circle")
                        .font(.callout).foregroundStyle(.secondary)
                }

                Section("Expert Cache - Persistent + Dynamic") {
                    Stepper("Slots per layer: \(store.expertCacheSlots == 0 ? "off" : "\(store.expertCacheSlots)")",
                            value: $store.expertCacheSlots, in: 0...64, step: 8)
                    Text("Each slot keeps one expert resident on the GPU. On the 43-layer Flash 2-bit model it is about 7 MB/slot/layer (8 slots ≈ 2.4 GB wired); Pro has 61 layers and larger experts, so its cost is substantially higher. Hot experts stay in RAM (hit = zero copies); cold experts rotate out via LRU. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let info = store.tuningInfo, info.cacheHits + info.cacheMisses > 0 {
                        let rate = Double(info.cacheHits) / Double(info.cacheHits + info.cacheMisses) * 100
                        LabeledContent("Hit rate",
                                       value: String(format: "%.0f%%  (%d hit / %d miss)", rate, info.cacheHits, info.cacheMisses))
                        Text(rate < 15
                             ? "Low hit rate: routing is almost uniform for this workload, so the cache is not paying off. Consider turning it off."
                             : "Useful hit rate: resident experts are saving I/O.")
                            .font(.caption)
                            .foregroundStyle(rate < 15 ? .orange : .green)
                    }
                }

                Section("Expert Usage Profile (Usage Imatrix)") {
                    LabeledContent("Active agent", value: store.selectedAgent.name)
                    if let info = store.tuningInfo {
                        LabeledContent("Recorded routes", value: "\(info.totalRoutes)")
                    }
                    Text("Counts how often the router picks each expert in your real usage. The profile is per-agent (different roles route to different experts): switching agents warms the cache from that profile. Persisted across sessions.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button { store.refreshTuningInfo() } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        Button { store.saveExpertUsage() } label: {
                            Label("Save Profile", systemImage: "square.and.arrow.down")
                        }
                        Button(role: .destructive) { store.resetExpertUsage() } label: {
                            Label("Reset", systemImage: "trash")
                        }
                    }
                    .disabled(!store.isReady)
                    if !store.isReady {
                        Text("Load a model in Chat to collect the profile.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()

            // Per-layer concentration: the honest signal for cache viability.
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Per-layer concentration (share of routing captured by the top-8 experts; ~3% = uniform, high = cache-friendly)")
                        .font(.caption.bold()).foregroundStyle(.secondary)
                        .padding(.bottom, 4)
                    if let info = store.tuningInfo, !info.layerSummaries.isEmpty {
                        ForEach(info.layerSummaries, id: \.self) { line in
                            Text(line)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    } else {
                        Text("No data yet - generate a few responses and press Refresh.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
            }
            .background(Color.black.opacity(0.05))
            }
            .onAppear { store.refreshTuningInfo() }
        } else {
            ContentUnavailableView(
                "Tuning esperti non disponibile",
                systemImage: "slider.horizontal.3",
                description: Text("Il modello selezionato non espone routing/cache esperti in questo backend. I controlli DeepSeek-specifici restano nascosti.")
            )
        }
    }
}
