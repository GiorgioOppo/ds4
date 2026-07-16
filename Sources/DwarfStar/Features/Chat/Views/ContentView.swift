import SwiftUI
import DS4Engine

struct ContentView: View {
    @Bindable var store: ChatStore

    var body: some View {
        switch store.phase {
        case .needsModel, .failed:
            ModelLoadView(store: store)
        case .loading:
            VStack(spacing: 12) {
                ProgressView()
                Text("Loading model...")
                    .foregroundStyle(.secondary)
                Text("Maps the GGUF and compiles Metal kernels. This can take a few seconds.")
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready:
            ChatView(store: store)
        }
    }
}

/// Pre-load configuration: model selection, SSD streaming, context, system prompt.
struct ModelLoadView: View {
    @Bindable var store: ChatStore
    @State private var showDownload = false

    var body: some View {
        Form {
            Section {
                if store.discoveredModels.isEmpty {
                    Text("Nessun modello DeepSeek V4 Flash supportato trovato. Puoi scaricarne uno oppure scegliere manualmente un GGUF avanzato.")
                        .foregroundStyle(.secondary)
                        .font(.callout)
                } else {
                    ForEach(store.discoveredModels) { model in
                        Button {
                            store.selectCatalogModel(path: model.path)
                        } label: {
                            HStack {
                                Image(systemName: store.modelPath == model.path
                                      ? "largecircle.fill.circle" : "circle")
                                    .foregroundStyle(.tint)
                                VStack(alignment: .leading) {
                                    Text(model.name).lineLimit(1)
                                    Text(model.displaySize)
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            } header: {
                HStack {
                    Text("Available Models")
                    Spacer()
                    Button { store.scanModels() } label: { Image(systemName: "arrow.clockwise") }
                        .buttonStyle(.borderless)
                    Button { showDownload = true } label: { Label("Scarica…", systemImage: "arrow.down.circle") }
                        .buttonStyle(.borderless)
                }
            }

            Section("Automatic Configuration") {
                Button {
                    store.applyRecommendedPreset()
                } label: {
                    Label("Configure for your RAM (\(MemoryInfo.gib(MemoryInfo.physicalBytes)))",
                          systemImage: "wand.and.stars")
                }
                if let note = store.presetNote {
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Paths") {
                Text(store.modelPath.isEmpty ? "Nessun GGUF selezionato" : store.modelPath)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Button {
                    if let path = ModelPicker.pickGGUF() { store.selectPickedModel(path: path) }
                } label: {
                    Label("Browse...", systemImage: "folder")
                }
                Text("With App Sandbox enabled, select the file here: the chosen path remains accessible and is remembered on the next launch.")
                    .font(.caption).foregroundStyle(.secondary)
                // Metal kernels are embedded in the app — no folder to set.
            }

            Section("Memory") {
                Text("SSD streaming is always enabled: non-routed weights are no-copy mapped through the page cache, and each token reads only the 6 selected experts. If the model fits in RAM, the page cache keeps it resident automatically.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Disk KV (reuse prefixes across sessions)", isOn: $store.diskKVEnabled)
                if store.diskKVEnabled {
                    Stepper("Budget: \(store.diskKVBudgetKTok)k tokens (≈ \(String(format: "%.1f", Double(store.diskKVBudgetKTok) * 0.022)) GB)",
                            value: $store.diskKVBudgetKTok, in: 128...4096, step: 128)
                    Text("At the end of a response, the KV state is saved to disk; a new conversation or server request that starts with a known prefix restores it instead of redoing prefill. The budget counts TOTAL checkpointed tokens across conversations (the live context window is separate). Applies on the next model load.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                Toggle("Raw-KV ring (experimental): constant KV RAM", isOn: $store.rawRingEnabled)
                if store.rawRingEnabled {
                    Text("Keeps only the attention window (128 rows) in RAM instead of the full context, making raw KV RAM independent of context length. Experimental: verify outputs after a long context. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Toggle("Read ahead selected experts (madvise)", isOn: $store.willNeedEnabled)
                Text("Starts reading the 6 selected experts just before gather: reduces cold faults on low-RAM systems, no-op when hot. Advisory only; does not change outputs. Recommended ON. Applies on the next model load.")
                    .font(.caption).foregroundStyle(.tertiary)
                Toggle("Decode route/attn profile (diagnostic)", isOn: $store.profileRouteEnabled)
                if store.profileRouteEnabled {
                    Text("Splits route/attn into 5 phases (comp/q/kv/attn/out) and writes the report to the engine log at the end of the turn. Extra commits slow generation: use it to understand where time goes, not to measure speed. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.orange)
                }
            }

            Section("Agent (Role)") {
                Picker("Agent", selection: Binding(get: { store.selectedAgentId },
                                                    set: { store.selectAgent($0) })) {
                    ForEach(store.agents) { agent in
                        Label(agent.name, systemImage: agent.icon).tag(agent.id)
                    }
                }
                if !store.selectedAgent.systemPrompt.isEmpty {
                    Text(store.selectedAgent.systemPrompt)
                        .font(.caption).foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                Text("Defines the chat role and tools. View and edit prompts in the Agents tab.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Context and System Prompt") {
                Stepper("Context: \(store.contextSize) tokens",
                        value: $store.contextSize, in: 1024...1_000_000, step: 1024)
                Text("KV caches grow with context and take page cache away from expert streaming. On low RAM, a large context slows generation a lot. The default is RAM-aware (4096 on <=16 GB, like the demo); raise it only if you need a longer window.")
                    .font(.caption).foregroundStyle(.secondary)
                TextField("Additional system prompt (added to the agent role)",
                          text: $store.systemPrompt, axis: .vertical)
                    .lineLimit(2...6)
            }

            if let warning = MemoryInfo.loadWarning(modelPath: store.modelPath) {
                Section {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
            }

            if case .failed(let message) = store.phase {
                Section {
                    Text(message).foregroundStyle(.red).font(.callout)
                }
            }

            Section {
                Button {
                    store.load()
                } label: {
                    Label("Load Model", systemImage: "bolt.fill")
                }
                .keyboardShortcut(.return, modifiers: [.command])
                Text("System RAM: \(MemoryInfo.gib(MemoryInfo.physicalBytes))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear { store.restoreModelBookmark(); store.scanModels() }
        .sheet(isPresented: $showDownload) {
            DownloadView(store: store)
        }
    }
}
