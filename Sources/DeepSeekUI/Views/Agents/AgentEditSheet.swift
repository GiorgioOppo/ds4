import SwiftUI
import DeepSeekTools

/// Modal sheet for creating / editing one `AgentConfig`. Holds a
/// local editable copy so Cancel discards every change; Save
/// passes the mutated copy back through `onSave`.
struct AgentEditSheet: View {
    let initial: AgentConfig
    @ObservedObject var mcpPool: MCPClientPool
    let onSave: (AgentConfig) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var name: String = ""
    @State private var summary: String = ""
    @State private var systemPrompt: String = ""
    @State private var defaultMode: String = "chat"
    /// Coarse Plan/Build switch. Persists into AgentConfig.agentMode.
    @State private var agentMode: AgentMode = .build
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 1.0
    @State private var topK: Int = 0
    @State private var minP: Double = 0.0
    @State private var tailFree: Double = 1.0
    @State private var typical: Double = 1.0
    @State private var repPenalty: Double = 1.0
    @State private var frequencyPenalty: Double = 0.0
    @State private var presencePenalty: Double = 0.0
    @State private var mirostatTau: Double = 0.0
    @State private var mirostatEta: Double = 0.1
    @State private var showAdvancedSampling: Bool = false
    @State private var maxTokens: Int = 4096
    @State private var iconName: String = "person.crop.circle"
    @State private var tint: String = "blue"
    /// Toolbox: nil → all connected MCP tools allowed; non-nil →
    /// explicit allowlist of qualified names ("server__tool"). The
    /// edit UI carries this as a (mode + explicit-set) pair so
    /// "All tools" / "No tools" stay reachable even when the live
    /// pool is empty.
    @State private var toolMode: ToolMode = .all
    @State private var explicitTools: Set<String> = []

    enum ToolMode: String, Hashable {
        case all       // pass nil downstream
        case none_     // pass [] downstream
        case allowlist // pass explicitTools downstream
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(initial.name == "Untitled agent" && !initial.systemPrompt.isEmpty == false
                  ? "New agent"
                  : "Edit \(initial.name)")
                .font(.title3.bold())

            // Form is wrapped in a ScrollView so the prompt textarea
            // doesn't squash the sampling sliders on a small Settings
            // window.
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    identitySection
                    Divider()
                    promptSection
                    Divider()
                    samplingSection
                    Divider()
                    toolsSection
                }
                .padding(.vertical, 4)
            }

            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { commit() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 600, height: 640)
        .onAppear { hydrate() }
    }

    // MARK: - identity

    private var identitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Identity").font(.headline)
            TextField("Name", text: $name)
                .textFieldStyle(.roundedBorder)
            TextField("Summary (shown in pickers)", text: $summary)
                .textFieldStyle(.roundedBorder)
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("SF Symbol")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("person.crop.circle", text: $iconName)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 220)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Tint")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("", selection: $tint) {
                        ForEach(AgentTint.all, id: \.self) { t in
                            HStack {
                                Circle()
                                    .fill(AgentTint.color(for: t))
                                    .frame(width: 12, height: 12)
                                Text(t.capitalized)
                            }
                            .tag(t)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 160)
                }
                Spacer()
                Image(systemName: iconName)
                    .font(.system(size: 36))
                    .foregroundStyle(AgentTint.color(for: tint))
                    .frame(width: 48, height: 48)
            }
        }
    }

    // MARK: - prompt

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("System prompt").font(.headline)
            Text("Injected as the first system message of every chat that uses this agent. Plain text — no template wrapping required.")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $systemPrompt)
                .font(.body.monospaced())
                .frame(minHeight: 140, maxHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.secondary.opacity(0.3)))
        }
    }

    // MARK: - sampling

    private var samplingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sampling defaults").font(.headline)
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Agent mode")
                Picker("", selection: $agentMode) {
                    ForEach(AgentMode.allCases, id: \.self) { m in
                        Text(m.displayName).tag(m)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 180)
                .help(agentMode.summary)
                Spacer()
            }
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Thinking")
                Picker("", selection: $defaultMode) {
                    Text("Chat (no think)").tag("chat")
                    Text("Think High").tag("high")
                    Text("Think Max").tag("max")
                }
                .labelsHidden()
                .frame(width: 180)
                Spacer()
                Text("Max tokens")
                TextField("", value: $maxTokens, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }
            slider("Temperature",
                    value: $temperature,
                    range: 0.5...1.0, step: 0.05,
                    fmt: "%.2f")
            slider("Top-P",
                    value: $topP,
                    range: 0.0...1.0, step: 0.05,
                    fmt: "%.2f")
            HStack(alignment: .firstTextBaseline) {
                Text("Top-K (0 = off)")
                    .frame(width: 130, alignment: .leading)
                TextField("", value: $topK, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 60)
                Spacer()
            }
            slider("Repetition penalty",
                    value: $repPenalty,
                    range: 1.0...1.5, step: 0.01,
                    fmt: "%.2f")

            DisclosureGroup(isExpanded: $showAdvancedSampling) {
                VStack(alignment: .leading, spacing: 8) {
                    slider("Min-P",
                            value: $minP,
                            range: 0.0...1.0, step: 0.01,
                            fmt: "%.2f")
                    slider("Tail-free z",
                            value: $tailFree,
                            range: 0.0...1.0, step: 0.01,
                            fmt: "%.2f")
                    slider("Typical p",
                            value: $typical,
                            range: 0.0...1.0, step: 0.01,
                            fmt: "%.2f")
                    slider("Frequency pen.",
                            value: $frequencyPenalty,
                            range: 0.0...2.0, step: 0.05,
                            fmt: "%.2f")
                    slider("Presence pen.",
                            value: $presencePenalty,
                            range: 0.0...2.0, step: 0.05,
                            fmt: "%.2f")
                    slider("Mirostat τ (0 = off)",
                            value: $mirostatTau,
                            range: 0.0...10.0, step: 0.1,
                            fmt: "%.1f")
                    slider("Mirostat η",
                            value: $mirostatEta,
                            range: 0.01...0.5, step: 0.01,
                            fmt: "%.2f")
                    Text("Min-P, tail-free and typical alternate with top-K/top-P. Mirostat overrides every tail filter and runs its own dynamic top-k.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.top, 4)
            } label: {
                Text("Advanced sampling")
                    .font(.subheadline.bold())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func slider(_ label: String,
                         value: Binding<Double>,
                         range: ClosedRange<Double>,
                         step: Double,
                         fmt: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).frame(width: 130, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: fmt, value.wrappedValue))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 50, alignment: .trailing)
        }
    }

    // MARK: - tools

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Tools").font(.headline)
            Text("Restrict which MCP tools the model can call when running under this agent.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("", selection: $toolMode) {
                Text("All connected tools").tag(ToolMode.all)
                Text("No tools").tag(ToolMode.none_)
                Text("Selected tools…").tag(ToolMode.allowlist)
            }
            .pickerStyle(.segmented)

            if toolMode == .allowlist {
                let liveTools = mcpPool.allTools()
                if liveTools.isEmpty {
                    Text("No connected MCP servers right now. The agent's allowlist still applies once servers come online.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(liveTools, id: \.qualifiedName) { t in
                        Toggle(isOn: Binding(
                            get: { explicitTools.contains(t.qualifiedName) },
                            set: { on in
                                if on { explicitTools.insert(t.qualifiedName) }
                                else  { explicitTools.remove(t.qualifiedName) }
                            }))
                        {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(t.qualifiedName)
                                    .font(.callout.monospaced())
                                if !t.description.isEmpty {
                                    Text(t.description)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                        }
                        .toggleStyle(.checkbox)
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    // MARK: - lifecycle

    private func hydrate() {
        name = initial.name
        summary = initial.summary
        systemPrompt = initial.systemPrompt
        defaultMode = initial.defaultMode
        agentMode = initial.agentMode
        temperature = initial.temperature
        topP = initial.topP
        topK = initial.topK
        minP = initial.minP
        tailFree = initial.tailFree
        typical = initial.typical
        repPenalty = initial.repetitionPenalty
        frequencyPenalty = initial.frequencyPenalty
        presencePenalty = initial.presencePenalty
        mirostatTau = initial.mirostatTau
        mirostatEta = initial.mirostatEta
        // Auto-expand if any advanced field is non-default so the user
        // sees the actual configuration without having to click.
        showAdvancedSampling = (initial.minP != 0.0
            || initial.tailFree != 1.0
            || initial.typical != 1.0
            || initial.frequencyPenalty != 0.0
            || initial.presencePenalty != 0.0
            || initial.mirostatTau != 0.0)
        maxTokens = initial.maxTokens
        iconName = initial.iconName.isEmpty ? "person.crop.circle" : initial.iconName
        tint = AgentTint.all.contains(initial.tint) ? initial.tint : "blue"
        if let allowed = initial.allowedToolNames {
            if allowed.isEmpty {
                toolMode = .none_
                explicitTools = []
            } else {
                toolMode = .allowlist
                explicitTools = allowed
            }
        } else {
            toolMode = .all
            explicitTools = []
        }
    }

    private func commit() {
        var updated = initial
        updated.name = name.trimmingCharacters(in: .whitespaces)
        updated.summary = summary.trimmingCharacters(in: .whitespaces)
        updated.systemPrompt = systemPrompt
        updated.defaultMode = defaultMode
        updated.agentMode = agentMode
        updated.temperature = max(0.5, min(1.0, temperature))
        updated.topP = max(0.0, min(1.0, topP))
        updated.topK = max(0, topK)
        updated.minP = max(0.0, min(1.0, minP))
        updated.tailFree = max(0.0, min(1.0, tailFree))
        updated.typical = max(0.0, min(1.0, typical))
        updated.repetitionPenalty = max(1.0, repPenalty)
        updated.frequencyPenalty = max(0.0, frequencyPenalty)
        updated.presencePenalty = max(0.0, presencePenalty)
        updated.mirostatTau = max(0.0, mirostatTau)
        updated.mirostatEta = max(0.01, mirostatEta)
        updated.maxTokens = max(1, maxTokens)
        updated.iconName = iconName
        updated.tint = tint
        switch toolMode {
        case .all:       updated.allowedToolNames = nil
        case .none_:     updated.allowedToolNames = []
        case .allowlist: updated.allowedToolNames = explicitTools
        }
        onSave(updated)
        dismiss()
    }
}
