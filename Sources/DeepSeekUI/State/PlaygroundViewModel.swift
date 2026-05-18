import Foundation
import DeepSeekTools

/// State holder per il `PlaygroundSheet`. Istanzia un
/// `DemoEchoChat` (concrete subclass di `ChatBase` che usa
/// `DemoEchoAgent` come backend) connesso a un `PluginRegistry`
/// con un `DemoLoggerPlugin` osservatore, per dimostrare l'OOP
/// foundation di `Sources/DeepSeekTools/Abstractions/`.
///
/// Mirror del pattern ConvertViewModel: @MainActor, @Published,
/// Task per le operazioni async.
@MainActor
final class PlaygroundViewModel: ObservableObject {

    // ---- UI state ----
    @Published var inputText: String = ""
    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil

    /// Snapshot del `Chat` mantenuto da `DemoEchoChat`. Aggiornato
    /// dopo ogni `ask` completata. Reso copy così la UI non legge
    /// stato mutevole sotto attivazione concorrente.
    @Published var chatSnapshot: Chat = Chat(title: "Demo Echo Chat")

    /// Log di envelope osservati dal `DemoLoggerPlugin` registrato
    /// nel `PluginRegistry`.
    @Published var pluginEvents: [PluginEventLog] = []

    struct PluginEventLog: Identifiable {
        let id = UUID()
        let timestamp: Date
        let kind: String
        let index: Int
    }

    // ---- OOP demo wiring ----
    private let chat: DemoEchoChat
    private let pluginRegistry: PluginRegistry
    private let loggerPlugin: DemoLoggerPlugin
    private let host: DemoPluginHost
    /// Task di bootstrap async: registra il plugin nel registry.
    /// Le ask() aspettano questa task prima di eseguire, così la
    /// Question pubblicata al broadcast trova il logger già attivo.
    private var bootstrapTask: Task<Void, Error>? = nil

    init() {
        let registry = PluginRegistry()
        let logger = DemoLoggerPlugin()
        let host = DemoPluginHost(registry: registry)
        self.pluginRegistry = registry
        self.loggerPlugin = logger
        self.host = host
        self.chat = DemoEchoChat(plugins: registry)

        self.bootstrapTask = Task { [registry, logger, host] in
            try await registry.register(logger, host: host)
        }
    }

    var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && !isRunning
    }

    func send() {
        let prompt = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isRunning else { return }
        inputText = ""
        isRunning = true
        lastError = nil

        Task { [weak self] in
            guard let self else { return }
            do {
                // Attendi il bootstrap del plugin prima del primo ask
                // (subsequent ask non aspettano niente — la task
                // è già `.success`).
                try await self.bootstrapTask?.value

                _ = try await self.chat.ask(prompt)

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.chatSnapshot = self.chat.chat
                    self.refreshPluginEvents()
                    self.isRunning = false
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    self?.lastError = error.localizedDescription
                }
            }
        }
    }

    func clearAll() {
        chat.clear()
        chatSnapshot = chat.chat
        pluginEvents.removeAll()
    }

    private func refreshPluginEvents() {
        // Il DemoLoggerPlugin accumula in observedKinds (NSLock
        // protetto). Re-renderiamo l'intera lista — N piccolo
        // (un paio per turno: chat.question + chat.answer).
        let kinds = loggerPlugin.observedKinds
        pluginEvents = kinds.enumerated().map { idx, kind in
            PluginEventLog(timestamp: Date(),
                            kind: kind,
                            index: idx)
        }
    }
}
