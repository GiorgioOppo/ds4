import Foundation
import SwiftUI
import DeepSeekKit

/// One conversation's runtime state, kept separate from the persisted
/// `Conversation` model so we can drive SwiftUI bindings without
/// trampling the on-disk snapshot during streaming.
enum GenerationPhase: Equatable {
    case idle
    case streaming(buffer: String, status: String)
    case error(String)
}

/// In-memory store. Multi-chat + persistence in commit 4.
@MainActor
final class ChatStore: ObservableObject {
    @Published var conversation: Conversation
    @Published var phase: GenerationPhase = .idle

    let service: InferenceService

    init(modelDirPath: String, service: InferenceService) {
        self.conversation = Conversation(modelDirPath: modelDirPath)
        self.service = service
    }

    /// Append a user message, flip to streaming, and pump
    /// `service.generate` events into the assistant placeholder until
    /// the stream finishes (or errors).
    func send(text: String,
               mode: ThinkingMode,
               options: SamplingOptions,
               maxTokens: Int) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // Don't allow re-entrancy.
        if case .streaming = phase { return }

        conversation.messages.append(StoredMessage(role: .user, content: trimmed))
        conversation.retitleIfNeeded()
        phase = .streaming(buffer: "", status: "Encoding prompt…")

        // Snapshot history (excluding the assistant placeholder we're
        // about to add) for the kit's encoder.
        let history = conversation.messages.map { $0.asKitMessage() }

        // Reserve an assistant slot in the conversation; we'll mutate
        // its `content` as tokens arrive.
        let placeholderId = UUID()
        conversation.messages.append(
            StoredMessage(id: placeholderId, role: .assistant, content: ""))

        Task {
            do {
                for try await event in service.generate(
                    history: history, mode: mode,
                    options: options, maxTokens: maxTokens)
                {
                    switch event {
                    case .token(let piece):
                        guard case .streaming(let buffer, _) = phase else { return }
                        let newBuffer = buffer + piece
                        phase = .streaming(buffer: newBuffer, status: "")
                        if let idx = conversation.messages.firstIndex(
                            where: { $0.id == placeholderId }) {
                            conversation.messages[idx].content = newBuffer
                        }

                    case .status(let s):
                        if case .streaming(let buffer, _) = phase {
                            phase = .streaming(buffer: buffer, status: s)
                        }

                    case .done(let final):
                        if let idx = conversation.messages.firstIndex(
                            where: { $0.id == placeholderId }) {
                            conversation.messages[idx] = StoredMessage(
                                id: placeholderId,
                                role: .assistant,
                                content: final.content,
                                reasoningContent: final.reasoningContent,
                                toolCalls: final.toolCalls.map(StoredToolCall.init))
                        }
                        phase = .idle
                    }
                }
            } catch {
                phase = .error((error as? LocalizedError)?.errorDescription
                                ?? error.localizedDescription)
            }
        }
    }

    func cancel() {
        service.cancelCurrent()
    }

    func newChat() {
        conversation = Conversation(modelDirPath: conversation.modelDirPath)
        phase = .idle
    }
}
