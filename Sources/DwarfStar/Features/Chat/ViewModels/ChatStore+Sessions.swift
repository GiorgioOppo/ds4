import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    // MARK: - Sessions (create / switch / delete / rename / persist)

    func newChat() { startNewChat() }

    /// Make a fresh persisted chat with the current role active. Reuses the current
    /// chat if it's still empty (so flipping the agent before sending anything
    /// doesn't pile up blank chats).
    func startNewChat() {
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        contextUsed = 0
        enginePrimed = true
        if let i = sessions.firstIndex(where: { $0.id == activeSessionId }),
           messages.isEmpty, sessions[i].messages.isEmpty {
            sessions[i].agentId = selectedAgentId
            sessions[i].systemNote = systemPrompt
            ChatSessionStore.save(sessions[i])
        } else {
            persistActiveSession()
            let session = ChatSession(agentId: selectedAgentId, systemNote: systemPrompt,
                                      modelName: info?.name ?? "")
            sessions.insert(session, at: 0)
            activeSessionId = session.id
            messages.removeAll()
            ChatSessionStore.save(session)
        }
        applyAgent()                      // role + tools + usage profile + resetConversation
    }

    /// Switch to an existing chat: persist the current one, then restore the target.
    func switchSession(_ id: String) {
        guard id != activeSessionId else { return }
        persistActiveSession()
        activate(id)
    }

    func deleteSession(_ id: String) {
        let wasActive = (id == activeSessionId)
        ChatSessionStore.delete(id)
        sessions.removeAll { $0.id == id }
        guard wasActive else { return }
        if let next = sessions.first { activate(next.id) } else { startNewChat() }
    }

    func renameSession(_ id: String, to title: String) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        sessions[i].title = trimmed.isEmpty ? ChatSession.untitled : trimmed
        ChatSessionStore.save(sessions[i])
    }

    /// Restore a session into the live UI and reset the engine to its role (without
    /// persisting the previous one — callers do that first when needed). A non-empty
    /// chat must re-prime on the next send, since the engine no longer holds its KV.
    func activate(_ id: String) {
        guard let target = sessions.first(where: { $0.id == id }) else { return }
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        activeSessionId = id
        messages = target.messages.map { UIMessage(stored: $0) }
        contextUsed = 0
        systemPrompt = target.systemNote
        if target.agentId != selectedAgentId, agents.contains(where: { $0.id == target.agentId }) {
            selectedAgentId = target.agentId
        }
        enginePrimed = messages.isEmpty
        applyAgent()                      // reset engine to the role; first send re-primes
    }

    /// Snapshot the live transcript into the active session and write it to disk.
    /// Trailing/empty assistant placeholders are dropped so a chat interrupted
    /// mid-generation doesn't reopen with a blank bubble.
    func persistActiveSession() {
        guard let i = sessions.firstIndex(where: { $0.id == activeSessionId }) else { return }
        let kept = messages.filter {
            !($0.role == .assistant && $0.text.isEmpty && $0.reasoning.isEmpty
              && $0.toolCalls.isEmpty && $0.subAgent == nil)
        }
        sessions[i].messages = kept.map { StoredMessage(from: $0) }
        sessions[i].agentId = selectedAgentId
        sessions[i].systemNote = systemPrompt
        if let name = info?.name { sessions[i].modelName = name }
        sessions[i].updatedAt = Date()
        if sessions[i].title == ChatSession.untitled {
            sessions[i].title = Self.deriveTitle(from: messages)
        }
        ChatSessionStore.save(sessions[i])
    }

    private func clearTransientTurnState() {
        attachments = []
        attachmentNote = nil
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        toolRounds = 0
    }

    /// First non-empty user line, for an auto title.
    private static func deriveTitle(from messages: [UIMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user && !$0.text.isEmpty }) else {
            return ChatSession.untitled
        }
        let line = first.text.split(separator: "\n").first.map(String.init) ?? first.text
        return String(line.prefix(48))
    }

}
