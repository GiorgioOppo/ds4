import Foundation

/// An agent = a ROLE the model plays in a conversation: a system prompt, the
/// tools it may call, and — crucially for this engine — its OWN expert-usage
/// profile. Different roles route to different experts, so each agent's profile
/// pre-warms the expert slot-cache with the experts THAT role actually uses.
/// Switching agent starts a fresh conversation with its role.
public struct AgentProfile: Sendable, Identifiable, Codable, Equatable {
    public var id: String          // stable key (also the usage-profile file key)
    public var name: String
    public var icon: String        // SF Symbol
    public var systemPrompt: String
    public var toolNames: [String] // built-in tools this agent exposes ([] = none)

    public init(id: String, name: String, icon: String, systemPrompt: String, toolNames: [String]) {
        self.id = id; self.name = name; self.icon = icon
        self.systemPrompt = systemPrompt; self.toolNames = toolNames
    }

    public static let defaults: [AgentProfile] = [
        .init(id: "generale", name: "Generale", icon: "person",
              systemPrompt: "", toolNames: []),
        .init(id: "coding", name: "Coding", icon: "chevron.left.forwardslash.chevron.right",
              systemPrompt: "Sei un assistente di programmazione esperto. Rispondi con codice corretto e conciso; spiega solo l'essenziale.",
              toolNames: []),
        .init(id: "matematica", name: "Matematica", icon: "function",
              systemPrompt: "Sei un assistente matematico preciso. Usa gli strumenti di calcolo forniti per ogni operazione aritmetica.",
              toolNames: ["calculator", "add", "subtract", "multiply"]),
        .init(id: "scrittura", name: "Scrittura", icon: "pencil",
              systemPrompt: "Sei un editor e scrittore in italiano: tono naturale, frasi chiare, niente giri di parole.",
              toolNames: []),
    ]
}
