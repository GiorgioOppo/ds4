import Foundation
import DS4Core

/// La superficie chat COMUNE dei backend: il contratto su cui ChatStore e
/// LocalServer programmano, implementato dall'`InferenceService` DeepSeek e
/// dal `GLM52ChatService`. Copre SOLO ciò che entrambi servono davvero — le
/// capacità specifiche (profili di decode, disk-KV DeepSeek, sub-agent,
/// distribuito) restano API dei tipi concreti, raggiunte con un cast
/// esplicito dove servono.
///
/// Nota firme: i metodi sincroni dell'actor testimoniano i requisiti
/// (anche quelli `async`) e restano `async` per i chiamanti cross-actor —
/// il contratto è identico da fuori per entrambi i backend.
public protocol ChatBackend: Actor {
    func modelInfo() -> ModelInfo
    @discardableResult
    func warmup() async -> Bool
    func quiesceForTeardown() async
    func setAgent(_ agent: AgentProfile, tools: [ToolSpec])
    func setTools(_ tools: [ToolSpec])
    func setCompactTools(_ on: Bool)
    func committedTokens() -> Int
    func send(userText: String, thinkMode: DS4ThinkMode,
              sampling: SamplingParams, maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error>
    func sendWithHistory(_ history: [ChatTurn], userText: String,
                         systemPrompt: String?, thinkMode: DS4ThinkMode,
                         sampling: SamplingParams, maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error>
    func provideToolResults(_ outputs: [ToolOutput],
                            thinkMode: DS4ThinkMode,
                            sampling: SamplingParams, maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error>
    func complete(turns: [ChatTurn], tools: [ToolSpec],
                  thinkMode: DS4ThinkMode, sampling: SamplingParams,
                  maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error>
}

extension InferenceService: ChatBackend {}
extension GLM52ChatService: ChatBackend {}
extension LagunaChatService: ChatBackend {}
