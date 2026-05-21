import Foundation
import SwiftUI

/// Owns the live state of the assistant round currently being
/// generated. The streaming bubble subscribes to THIS object
/// (`@ObservedObject`), so per-token mutations only invalidate the
/// streaming view — the rest of the transcript stays Equatable-stable
/// and skips re-render.
///
/// One controller per active chat; lives in
/// `ChatStore.streamingControllers[chatID]` from the moment the
/// placeholder assistant message is appended until `.done` finalises
/// the round and copies its payload back into the `StoredMessage`.
///
/// The streaming buffer (`round.content`), reasoning buffer, and
/// prefill trace all mutate through this controller. Token-level
/// disk persistence (the future `pending.json`) reads from here too,
/// so a crash mid-stream can rebuild the in-progress round from the
/// pending snapshot + the generated-token stream the store already
/// persists.
@MainActor
final class StreamingRoundController: ObservableObject {
    /// Live round being generated. The `content`, `reasoningContent`,
    /// and `prefillTrace` fields grow during streaming; the others
    /// stay at their initial values until `.done` finalises the
    /// round and the store copies the payload over.
    @Published var round: StoredRound

    /// Identifier of the turn this round belongs to. Used by the
    /// store + the future v2 lazy-loading path to route the round
    /// payload to the right `turns/{turnID}/rounds/{roundID}.json`
    /// once persistence lands on this controller (PR 5).
    let turnID: UUID

    init(round: StoredRound, turnID: UUID) {
        self.round = round
        self.turnID = turnID
    }

    /// Append `piece` to the round's `content`. Single mutation,
    /// single `objectWillChange` event — SwiftUI re-evaluates only
    /// the streaming bubble that has this controller as its
    /// `@ObservedObject`.
    func appendContent(_ piece: String) {
        round.content += piece
    }

    /// Append `piece` to the round's `prefillTrace`. Same isolation
    /// guarantee as `appendContent`.
    func appendPrefillTrace(_ piece: String) {
        var trace = round.prefillTrace ?? ""
        trace += piece
        round.prefillTrace = trace
    }

    /// Reset the prefill trace on a fresh prefill start. Mirrors
    /// the store's previous `messages[mIdx].prefillTrace = nil`
    /// behaviour for resumes.
    func clearPrefillTrace() {
        round.prefillTrace = nil
    }

    // MARK: - Prefill-token indicator

    /// Real BPE tokens of the current prefill delta, pushed one at a
    /// time by `.prefillTokenProcessed`. Drives the dedicated
    /// prefill-token indicator. Transient — kept off `round`, so it is
    /// never persisted; reset at every `.prefillStart`.
    @Published var prefillTokenSteps: [String] = []
    /// Token count of the current prefill delta, from `.prefillStart`;
    /// the indicator shows `prefillTokenSteps.count` out of this.
    @Published var prefillTokenTotal: Int = 0

    /// Reset the prefill-token indicator state for a new prefill.
    func beginPrefillTokens(total: Int) {
        prefillTokenSteps.removeAll(keepingCapacity: true)
        prefillTokenTotal = total
    }

    /// Append one decoded prefill token. Single `@Published` mutation
    /// → only the prefill indicator re-renders.
    func appendPrefillTokenStep(_ piece: String) {
        prefillTokenSteps.append(piece)
    }

    /// Live reasoning buffer for the round. Mutated by the remote
    /// path's `reasoning_content` delta events (local generation
    /// surfaces reasoning only at `.done`).
    func setReasoning(_ text: String) {
        round.reasoningContent = text
    }
}
