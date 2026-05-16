import Foundation
import SwiftUI

/// One named preset that pins down how a chat behaves before the
/// first turn fires: the system prompt to inject, which subset of
/// the live MCP tools the model is allowed to see, the default
/// thinking mode + sampling parameters, and a few UI affordances
/// (icon / tint) so the user can spot the right agent at a glance.
///
/// Conversations don't *contain* an AgentConfig — they reference
/// one by id (`Conversation.agentID`, added in step A2) so
/// editing an agent later changes the behaviour of every chat
/// that uses it without rewriting per-chat state.
///
/// Sampling fields mirror the ones in `SamplingOptions` but stored
/// as plain `Double`/`Int` (no Float) so JSON encoding doesn't
/// drift across architectures.
struct AgentConfig: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    /// One-liner shown when another agent considers delegating to
    /// this one (step A3). Also the visible subtitle in the
    /// Settings sidebar.
    var summary: String
    /// Full multi-line system block content. Plain text — the chat
    /// template will wrap it as needed.
    var systemPrompt: String
    /// Allowlist of MCP tool qualified names ("server__tool"). When
    /// `nil`, every connected tool is exposed; when an empty set,
    /// the agent runs without tools.
    var allowedToolNames: Set<String>?
    /// `ThinkingMode.rawValue` — "chat" / "high" / "max".
    var defaultMode: String
    /// Sampling defaults. The Generation Settings tab still wins
    /// when the user tweaks the global sliders; the agent's values
    /// are picked up on every fresh chat under that agent.
    var temperature: Double
    var topP: Double
    var topK: Int
    /// Filter tokens with `prob < minP × max_prob`. `0` disables.
    var minP: Double
    /// Tail-free sampling z-parameter. `1` disables.
    var tailFree: Double
    /// Locally-typical sampling mass. `1` disables.
    var typical: Double
    var repetitionPenalty: Double
    /// OpenAI-style frequency penalty (scales with token count).
    /// `0` disables.
    var frequencyPenalty: Double
    /// OpenAI-style presence penalty (binary in token presence).
    /// `0` disables.
    var presencePenalty: Double
    /// Mirostat v2 target surprise. `0` disables.
    var mirostatTau: Double
    /// Mirostat v2 learning rate. Default 0.1.
    var mirostatEta: Double
    var maxTokens: Int
    /// SF Symbol name + a tint identifier for the sidebar / picker.
    /// `tint` is one of: "blue" "purple" "pink" "red" "orange"
    /// "yellow" "green" "teal" "gray". Mapped to `Color` at render
    /// time so the on-disk format stays stable across SwiftUI
    /// revisions.
    var iconName: String
    var tint: String
    var createdAt: Date

    init(id: UUID = UUID(),
         name: String,
         summary: String = "",
         systemPrompt: String = "",
         allowedToolNames: Set<String>? = nil,
         defaultMode: String = "chat",
         temperature: Double = 0.7,
         topP: Double = 1.0,
         topK: Int = 0,
         minP: Double = 0.0,
         tailFree: Double = 1.0,
         typical: Double = 1.0,
         repetitionPenalty: Double = 1.0,
         frequencyPenalty: Double = 0.0,
         presencePenalty: Double = 0.0,
         mirostatTau: Double = 0.0,
         mirostatEta: Double = 0.1,
         maxTokens: Int = 4096,
         iconName: String = "person.crop.circle",
         tint: String = "blue",
         createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.summary = summary
        self.systemPrompt = systemPrompt
        self.allowedToolNames = allowedToolNames
        self.defaultMode = defaultMode
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.tailFree = tailFree
        self.typical = typical
        self.repetitionPenalty = repetitionPenalty
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.maxTokens = maxTokens
        self.iconName = iconName
        self.tint = tint
        self.createdAt = createdAt
    }

    // Backward-compatible decoder: agents persisted before the
    // advanced sampling fields existed get sensible defaults instead
    // of failing to decode.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        summary = try c.decodeIfPresent(String.self, forKey: .summary) ?? ""
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        allowedToolNames = try c.decodeIfPresent(Set<String>.self, forKey: .allowedToolNames)
        defaultMode = try c.decodeIfPresent(String.self, forKey: .defaultMode) ?? "chat"
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.7
        topP = try c.decodeIfPresent(Double.self, forKey: .topP) ?? 1.0
        topK = try c.decodeIfPresent(Int.self, forKey: .topK) ?? 0
        minP = try c.decodeIfPresent(Double.self, forKey: .minP) ?? 0.0
        tailFree = try c.decodeIfPresent(Double.self, forKey: .tailFree) ?? 1.0
        typical = try c.decodeIfPresent(Double.self, forKey: .typical) ?? 1.0
        repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1.0
        frequencyPenalty = try c.decodeIfPresent(Double.self, forKey: .frequencyPenalty) ?? 0.0
        presencePenalty = try c.decodeIfPresent(Double.self, forKey: .presencePenalty) ?? 0.0
        mirostatTau = try c.decodeIfPresent(Double.self, forKey: .mirostatTau) ?? 0.0
        mirostatEta = try c.decodeIfPresent(Double.self, forKey: .mirostatEta) ?? 0.1
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 4096
        iconName = try c.decodeIfPresent(String.self, forKey: .iconName) ?? "person.crop.circle"
        tint = try c.decodeIfPresent(String.self, forKey: .tint) ?? "blue"
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? .now
    }
}

/// Map the on-disk `tint` identifiers to SwiftUI Colors. Kept as a
/// free function so both the sidebar and the editor preview can
/// share it without forcing AgentConfig to import SwiftUI.
enum AgentTint {
    static let all: [String] = [
        "blue", "purple", "pink", "red", "orange",
        "yellow", "green", "teal", "gray"
    ]
    static func color(for tint: String) -> Color {
        switch tint {
        case "blue":   return .blue
        case "purple": return .purple
        case "pink":   return .pink
        case "red":    return .red
        case "orange": return .orange
        case "yellow": return .yellow
        case "green":  return .green
        case "teal":   return .teal
        case "gray":   return .gray
        default:       return .blue
        }
    }
}

@MainActor
final class AgentLibrary: ObservableObject {
    @Published private(set) var agents: [AgentConfig] = []

    init() {
        load()
    }

    func add(_ agent: AgentConfig) {
        agents.append(agent)
        save()
    }

    func update(_ agent: AgentConfig) {
        guard let idx = agents.firstIndex(where: { $0.id == agent.id }) else { return }
        agents[idx] = agent
        save()
    }

    func delete(_ id: UUID) {
        agents.removeAll { $0.id == id }
        save()
    }

    func agent(id: UUID) -> AgentConfig? {
        agents.first(where: { $0.id == id })
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.agentsConfigURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let entries = try? decoder.decode([AgentConfig].self, from: data) {
            agents = entries.sorted { $0.createdAt < $1.createdAt }
        }
    }

    private func save() {
        guard let url = try? PersistencePaths.agentsConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(agents) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
