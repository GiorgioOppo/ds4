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
              systemPrompt: "Sei un assistente di programmazione esperto. Rispondi con codice corretto e conciso; spiega solo l'essenziale. Se è stato importato un progetto, esploralo con i tool project_list / project_search e leggi SOLO i file rilevanti con project_read prima di rispondere.",
              toolNames: ["project_list", "project_read", "project_search"]),
        .init(id: "code", name: "Code", icon: "terminal",
              systemPrompt: """
              Sei un agente di coding autonomo che lavora sul progetto importato. Per ogni richiesta segui questo metodo, un tool alla volta:
              1) ESPLORA: individua i file rilevanti con project_list e project_search.
              2) LEGGI: leggi le parti che ti servono con project_read PRIMA di toccare qualsiasi cosa. Non inventare mai contenuti di file che non hai letto.
              3) MODIFICA: applica modifiche piccole e mirate con project_edit (il testo 'find' deve essere ESATTO, indentazione inclusa, e unico nel file: includi le righe adiacenti). Usa project_write solo per file nuovi o riscritture complete.
              4) VERIFICA: rileggi la zona modificata con project_read, controlla la coerenza (import, chiamanti trovati con project_search) e ispeziona le modifiche con git (es. "diff --stat", "diff <file>").
              5) Se il repo è git e l'utente lo chiede, committa con git "commit -am <messaggio conciso>".
              Alla fine riassumi in 2-3 frasi cosa hai cambiato e dove (file:riga). Se il task è ambiguo o rischioso, fermati e chiedi.
              """,
              toolNames: ["project_list", "project_read", "project_search",
                          "project_write", "project_edit", "git"]),
        .init(id: "orchestratore", name: "Orchestratore", icon: "person.3.sequence",
              systemPrompt: "Delega i compiti su file/progetto a sub-agent: subagent_search per trovare i file, poi subagent_run(target, domanda focalizzata, tools minimi — sola lettura di default, edit/write solo se serve). Integra le risposte.",
              toolNames: ["subagent_search", "subagent_run", "project_list", "project_search"]),
        .init(id: "matematica", name: "Matematica", icon: "function",
              systemPrompt: "Sei un assistente matematico preciso. Usa gli strumenti di calcolo forniti per ogni operazione aritmetica.",
              toolNames: ["calculator", "add", "subtract", "multiply"]),
        .init(id: "scrittura", name: "Scrittura", icon: "pencil",
              systemPrompt: "Sei un editor e scrittore in italiano: tono naturale, frasi chiare, niente giri di parole.",
              toolNames: []),
    ]
}
