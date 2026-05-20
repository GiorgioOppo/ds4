import Foundation

/// On-disk metadata for one chat, in the v2 lazy-loading layout. Lives
/// at `conversations/{chatID}/chat.json`. Carries everything the
/// sidebar + chat header need to render WITHOUT touching the per-turn
/// transcript: title, dates, endpoint, agent/project bindings,
/// cumulative cost, and the chronological list of turn IDs.
///
/// Heavy data — message text, reasoning, tool outputs, prefill traces
/// — lives in `TurnSummary` (medium) and `StoredRound` (full payload)
/// files under `chat.json`'s sibling `turns/` directory. The canonical
/// tokenized prompt stream (`encodedTokens` in the v1 `Conversation`)
/// is split out as a separate binary file `chat.tokens` so the manifest
/// stays in the kilobytes range regardless of conversation length.
///
/// `schemaVersion` starts at 2 for this layout; v1 is the legacy
/// single-file `{chatID}.json`. Chats persisted before this refactor
/// keep using the v1 layout indefinitely — `ChatStore` reads both.
struct ChatManifest: Codable, Identifiable, Hashable {
    let id: UUID
    var title: String
    var createdAt: Date
    /// Path of the local model loaded when the chat was created. Empty
    /// for remote chats. Used as the legacy fallback for
    /// `effectiveEndpoint` resolution; new chats also fill `endpoint`.
    var modelDirPath: String
    /// Typed endpoint. When set, takes precedence over `modelDirPath`.
    /// Nil for v1 chats that predated the typed field.
    var endpoint: ModelEndpoint?
    /// Optional reference to a `Project` in `ProjectLibrary`. Same
    /// semantics as the v1 `Conversation.projectID`.
    var projectID: UUID?
    /// Optional reference to an `AgentConfig` in `AgentLibrary`.
    var agentID: UUID?
    /// Sum of `usage.total_cost` reported by the remote provider, in
    /// USD, across every turn in this chat. Nil = never billed.
    var cumulativeCostUSD: Double?
    /// `ThinkingMode.rawValue` that produced the persisted
    /// `chat.tokens`. Invalidated when the user changes mode
    /// mid-chat, same as the v1 `lastEncodedMode`.
    var lastEncodedMode: String?
    /// Turn IDs in chronological order. The actual `TurnSummary` for
    /// each lives at `turns/{turnID}.json`. The full payload for each
    /// round lives at `turns/{turnID}/rounds/{roundID}.json`.
    var turnIDs: [UUID]
    /// On-disk schema version. 2 = this lazy layout. 1 = legacy.
    var schemaVersion: Int

    init(id: UUID = UUID(),
         title: String = "New Chat",
         createdAt: Date = .now,
         modelDirPath: String = "",
         endpoint: ModelEndpoint? = nil,
         projectID: UUID? = nil,
         agentID: UUID? = nil,
         cumulativeCostUSD: Double? = nil,
         lastEncodedMode: String? = nil,
         turnIDs: [UUID] = [],
         schemaVersion: Int = 2)
    {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.modelDirPath = modelDirPath
        self.endpoint = endpoint
        self.projectID = projectID
        self.agentID = agentID
        self.cumulativeCostUSD = cumulativeCostUSD
        self.lastEncodedMode = lastEncodedMode
        self.turnIDs = turnIDs
        self.schemaVersion = schemaVersion
    }

    /// Best-effort endpoint resolution mirroring the v1
    /// `Conversation.effectiveEndpoint` helper. Prefers the typed
    /// `endpoint` field; falls back to the legacy `modelDirPath` when
    /// non-empty.
    var effectiveEndpoint: ModelEndpoint? {
        if let endpoint = endpoint { return endpoint }
        if !modelDirPath.isEmpty {
            return .localDirectory(path: modelDirPath)
        }
        return nil
    }
}

/// Bitfield summarising which heavy disclosures a turn has, so the
/// `AssistantTurnView` can paint the right affordances (reasoning
/// badge, tool-call count chip, prefill trace caret) WITHOUT reading
/// any `StoredRound` file. Each flag corresponds to a non-empty field
/// across the turn's rounds.
struct TurnFlags: OptionSet, Codable, Hashable {
    let rawValue: Int

    static let hasReasoning    = TurnFlags(rawValue: 1 << 0)
    static let hasToolCalls    = TurnFlags(rawValue: 1 << 1)
    static let hasPrefillTrace = TurnFlags(rawValue: 1 << 2)
    static let hasDelegation   = TurnFlags(rawValue: 1 << 3)
    static let isRemote        = TurnFlags(rawValue: 1 << 4)
    /// Turn that carries only a system message (no user prompt, no
    /// assistant rounds). Drawn as a `systemBubble` instead of the
    /// user → assistant pair.
    static let isSystem        = TurnFlags(rawValue: 1 << 5)
}

/// Medium-weight per-turn summary loaded once when a chat is selected.
/// One file per turn at `turns/{turnID}.json`. Carries the user prompt
/// inline (small) plus a preview of the final assistant content (~2 KB)
/// so the visible bubble paints without any round IO. Heavy data —
/// reasoning content, tool calls, tool outputs, prefill trace — lives
/// only in the per-round files, decoded on demand when a disclosure
/// opens or when `finalContentIsTruncated` forces a full read.
///
/// A "turn" is the lifecycle of one user prompt + every assistant
/// generate pass that followed it (one pass per `roundIDs` entry).
/// Tool-call roundtrips do NOT start a new turn — they append rounds.
struct TurnSummary: Codable, Identifiable, Hashable {
    let id: UUID
    var createdAt: Date
    /// Identifier of the user (or system) message that opened the
    /// turn. Carried for cross-referencing with the v1
    /// `StoredMessage.id` ids that the inference engine still uses
    /// during generation.
    var userMessageID: UUID
    /// The prompt text. Always inline because it's typed by the user
    /// (or set by the chat host) — bounded by the composer's input.
    var userText: String
    var userTokenCount: Int?
    /// First ~2 KB of the final round's `content`. The visible bubble
    /// renders straight from this; full content is fetched only when
    /// `finalContentIsTruncated` and the bubble actually shows on
    /// screen.
    var finalContentPreview: String
    var finalContentIsTruncated: Bool
    /// Round IDs in chronological order. Empty for system-only turns
    /// (`flags.contains(.isSystem)`) and for fresh turns whose first
    /// round hasn't been persisted yet (still streaming into
    /// `pending.json` / the in-memory `activeRound` slot).
    var roundIDs: [UUID]
    var flags: TurnFlags
    /// Total number of `toolCalls` aggregated across rounds. Drives
    /// the "N tool calls" chip without loading any round file.
    var toolCallCount: Int
    /// Sum of `tokenCount` across rounds. Informational; UI shows it
    /// in the per-turn metrics line.
    var totalGeneratedTokens: Int
    /// USD cost of this turn as reported by the remote provider, when
    /// non-local. Nil for local generations and for providers that
    /// don't report cost.
    var turnCostUSD: Double?

    init(id: UUID = UUID(),
         createdAt: Date = .now,
         userMessageID: UUID,
         userText: String,
         userTokenCount: Int? = nil,
         finalContentPreview: String = "",
         finalContentIsTruncated: Bool = false,
         roundIDs: [UUID] = [],
         flags: TurnFlags = [],
         toolCallCount: Int = 0,
         totalGeneratedTokens: Int = 0,
         turnCostUSD: Double? = nil)
    {
        self.id = id
        self.createdAt = createdAt
        self.userMessageID = userMessageID
        self.userText = userText
        self.userTokenCount = userTokenCount
        self.finalContentPreview = finalContentPreview
        self.finalContentIsTruncated = finalContentIsTruncated
        self.roundIDs = roundIDs
        self.flags = flags
        self.toolCallCount = toolCallCount
        self.totalGeneratedTokens = totalGeneratedTokens
        self.turnCostUSD = turnCostUSD
    }
}

/// Full payload of one assistant generate pass. One file per round at
/// `turns/{turnID}/rounds/{roundID}.json`. Decoded only when needed:
/// disclosure opens, full-content fetch for a truncated preview, or
/// the send path materialising the conversation history for the kit.
struct StoredRound: Codable, Identifiable, Hashable {
    let id: UUID
    /// 0-based position of this round inside its turn. Stable across
    /// loads so the LRU cache and the UI's "round N of M" disclosure
    /// stay coherent if rounds are re-read.
    var roundIndex: Int
    var content: String
    var reasoningContent: String?
    var toolCalls: [StoredToolCall]
    var toolOutputs: [String]?
    var prefillTrace: String?
    var tokenCount: Int?

    init(id: UUID = UUID(),
         roundIndex: Int,
         content: String = "",
         reasoningContent: String? = nil,
         toolCalls: [StoredToolCall] = [],
         toolOutputs: [String]? = nil,
         prefillTrace: String? = nil,
         tokenCount: Int? = nil)
    {
        self.id = id
        self.roundIndex = roundIndex
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolOutputs = toolOutputs
        self.prefillTrace = prefillTrace
        self.tokenCount = tokenCount
    }
}

/// Composite key identifying one round on disk. Used by
/// `ChatPersistence` for routing and by the round LRU cache.
struct RoundKey: Hashable {
    let chatID: UUID
    let turnID: UUID
    let roundID: UUID
}

/// Crash-recovery snapshot written hot during streaming. Separated
/// from `ChatManifest` so per-token writes touch ~1 KB instead of
/// re-encoding the whole transcript. Tagged union covers both the
/// local token-cursor case (v1 `PendingTurn`) and the remote
/// re-issue-call case (v1 `RemotePendingTurn`); only one variant is
/// populated at a time.
struct PendingSnapshot: Codable, Hashable {
    enum Kind: String, Codable, Hashable { case local, remote }
    var kind: Kind
    var local: PendingTurn?
    var remote: RemotePendingTurn?

    static func local(_ pt: PendingTurn) -> PendingSnapshot {
        PendingSnapshot(kind: .local, local: pt, remote: nil)
    }

    static func remote(_ rt: RemotePendingTurn) -> PendingSnapshot {
        PendingSnapshot(kind: .remote, local: nil, remote: rt)
    }
}
