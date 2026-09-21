import Foundation

/// Single-driver contract for architecture-owned Swift decoders. The chat
/// service serializes access on its dedicated executor; a decoder never shares
/// mutable KV/recurrent state with another conversation or model.
public protocol SwiftModelDecoder: AnyObject {
    var contextCapacity: Int { get }
    var vocabularySize: Int { get }
    /// Number of tokens whose complete layer stack has been committed.
    var position: Int { get }

    /// Discard all KV, recurrent, routing and token-history state.
    func reset() throws

    /// Append a causal suffix and return vocabulary logits for its last token.
    /// Validate the whole input before dispatch. Cancellation or a GPU/I/O
    /// failure invalidates partial state; callers reset and replay the retained
    /// prefix before another evaluation. No empty suffix or implicit truncation.
    func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float]
}

public enum SwiftModelDecoderError: Error, CustomStringConvertible {
    case invalidInput(String)
    case invalidState
    case contextOverflow(requested: Int, capacity: Int)
    case cancelled
    case gpu(String)

    public var description: String {
        switch self {
        case .invalidInput(let message): return message
        case .invalidState: return "Stato del decoder incompleto: ripristinare il contesto prima di continuare."
        case .contextOverflow(let requested, let capacity):
            return "Il contesto richiede \(requested) token, ma il modello è configurato per \(capacity)."
        case .cancelled: return "Generazione interrotta."
        case .gpu(let message): return "Errore Metal: \(message)"
        }
    }
}
