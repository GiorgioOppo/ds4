import Foundation

// Decode SELF-SPECULATIVE, Fase A: i primitivi (vedi docs/SELF-SPECULATIVE.md).
//
// Il decode paga un giro completo (route → readback → gather → FFN, ×43 layer)
// per OGNI token. Lo schema speculativo lo paga una volta ogni round: K token
// candidati generati con un draft ECONOMICO (activeExperts ridotto, ~1/3
// dell'I/O esperti), poi verificati in UN passo batch full-config; il prefisso
// confermato avanza j+1 token per round (greedy: parità bit-per-bit col decode
// normale, per costruzione).
//
// Questo file contiene i due primitivi che rendono possibile il loop:
//  - `setActiveExperts`: il draft/full switch (il dispatch legge già
//    d.activeExperts per-forward e rinormalizza i route weights);
//  - snapshot/restore LEGGERO dello stato ricorrente NSA — il vincolo di
//    rollback: la ricorrenza dei compressori avanza a ogni token e non si può
//    riavvolgere. Qui si salva SOLO l'accumulatore ([coff·ratio × width] f32 +
//    count, ~32 KB/layer; niente righe di cache né KV raw: le righe compresse
//    oltre il count ripristinato e le righe raw delle posizioni rifiutate
//    vengono RISCRITTE — emissione indicizzata da `count`, KV raw posizionale
//    (`pos % rawRows`) — prima di qualunque lettura).

/// Lo stato ricorrente minimo per riavvolgere una finestra speculativa
/// rifiutata: accumulatori dei compressori NSA (attn + indexer) per layer.
/// ~1-2 MB sul Flash (contro i ~22 KB/token del KVSnapshot dei checkpoint,
/// che serializza anche finestra raw e righe compresse).
public struct SpecRecurrentState {
    struct Comp {
        var count: Int
        var stateKv: [Float]
        var stateScore: [Float]
    }
    var layers: [(comp: Comp?, idx: Comp?)]
    /// Posizione (token nel KV) al momento dello snapshot — il chiamante
    /// riparte da qui dopo un restore.
    public let nKeys: Int
}

/// Outcome of one DSpark greedy proposal after target verification.
/// A completely accepted window normally retains the batched verifier frontier
/// (the speed path); partial acceptance is restored and replayed exactly.
public struct DSparkVerificationResult: Sendable {
    public let proposedCount: Int
    public let acceptedTokens: [Int]
    public let currentTarget: Int
    public let nextLogits: [Float]
    public let retainedVerifierFrontier: Bool

    public var acceptedCount: Int { acceptedTokens.count }
    public var fullyAccepted: Bool {
        proposedCount > 0 && acceptedCount == proposedCount
    }
}

extension StreamingDecoder {
    /// Esperti attivi per token correnti (route top-K del dispatch).
    public var activeExpertsNow: Int { d.activeExperts }

    /// Override runtime degli esperti attivi (draft self-speculative: 2;
    /// full: il valore di configurazione, che il chiamante salva prima).
    /// Chiamare SOLO tra un forward e l'altro, sul thread del decode — il
    /// dispatch legge d.activeExperts alla route di ogni layer e rinormalizza
    /// i pesi sul top-K ridotto.
    public func setActiveExperts(_ k: Int) {
        d.activeExperts = max(1, min(k, d.k))
    }

    /// Snapshot leggero dello stato ricorrente a `nKeys` token committati.
    /// Chiamare tra un forward e l'altro (nessun lavoro GPU in volo).
    public func specSnapshot(nKeys: Int) -> SpecRecurrentState {
        var layers: [(comp: SpecRecurrentState.Comp?, idx: SpecRecurrentState.Comp?)] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            layers.append((comp: Self.specComp(compStates[i]),
                           idx: Self.specComp(indexStates[i])))
        }
        return SpecRecurrentState(layers: layers, nKeys: nKeys)
    }

    /// Ripristina lo stato ricorrente dello snapshot. NON tocca cache righe
    /// né KV raw (vedi il contratto in testa al file): dopo il restore il
    /// chiamante DEVE ricalcolare le posizioni ≥ snapshot.nKeys full-config
    /// (il passo di verifica) prima di leggerne KV o righe compresse.
    public func specRestore(_ s: SpecRecurrentState) {
        precondition(s.layers.count == nLayers, "specRestore: layer count mismatch")
        for i in 0..<nLayers {
            Self.specRestoreComp(compStates[i], from: s.layers[i].comp)
            Self.specRestoreComp(indexStates[i], from: s.layers[i].idx)
        }
    }

    /// Commit a verified speculative prefix through the ordinary single-token
    /// decode path. Batch verification and decode use numerically different
    /// compressor kernels; retaining the verifier frontier can therefore alter
    /// later greedy tokens even when every draft token was accepted.
    @discardableResult
    public func specReplay(_ tokens: ArraySlice<Int>, startPos: Int) throws
        -> [Float] {
        precondition(!tokens.isEmpty, "specReplay requires an accepted prefix")
        var logits: [Float] = []
        for (offset, token) in tokens.enumerated() {
            let position = startPos + offset
            logits = try forward(token: token, pos: position,
                                 nKeys: position + 1)
        }
        return logits
    }

    /// Verify a DSpark proposal against the full target and commit exactly its
    /// matching greedy prefix.
    ///
    /// The verifier window is allowed to mutate raw KV and recurrent NSA state.
    /// On a full accept its last row is already the correct target frontier, so
    /// retaining it removes the otherwise unavoidable duplicate target decode.
    /// A partial accept restores the light snapshot and replays only its accepted
    /// prefix. `DS4_DSPARK_EXACT_REPLAY=1` restores the older always-replay mode
    /// for strict A/B parity diagnostics.
    public func dsparkVerifyAndCommit(proposal: [Int],
                                      currentLogits: [Float],
                                      startPos: Int) throws
        -> DSparkVerificationResult {
        precondition(!currentLogits.isEmpty,
                     "DSpark verify requires the current target logits")
        precondition(startPos > 0, "DSpark verify cannot start before the prompt")
        precondition(startPos + proposal.count <= maxKeys,
                     "DSpark verify window exceeds KV capacity")

        let currentTarget = Self.greedyArgmax(currentLogits)
        guard !proposal.isEmpty,
              proposal[0] == currentTarget else {
            return DSparkVerificationResult(
                proposedCount: proposal.count,
                acceptedTokens: [],
                currentTarget: currentTarget,
                nextLogits: currentLogits,
                retainedVerifierFrontier: false)
        }

        let snapshot = specSnapshot(nKeys: startPos)
        let rows: [[Float]]
        do {
            rows = try specVerifyStep(tokens: proposal, startPos: startPos)
        } catch {
            dsparkStage0Runtime?.abortBatchCapture()
            specRestore(snapshot)
            throw error
        }
        let verifiedNext = rows.map(Self.greedyArgmax)
        let acceptedCount = DSparkGreedyVerifier.acceptedPrefix(
            proposal: proposal,
            currentTarget: currentTarget,
            verifiedNext: verifiedNext)

        let retainFull = acceptedCount == proposal.count
            && acceptedCount > 0
            && ProcessInfo.processInfo.environment["DS4_DSPARK_EXACT_REPLAY"] != "1"
        if retainFull, let frontier = rows.last {
            return DSparkVerificationResult(
                proposedCount: proposal.count,
                acceptedTokens: proposal,
                currentTarget: currentTarget,
                nextLogits: frontier,
                retainedVerifierFrontier: true)
        }

        // The verifier captured all draft rows, including rejected ones. Discard
        // that seed before replay; ordinary forwards maintain the private ring
        // with target-confirmed rows only.
        dsparkStage0Runtime?.abortBatchCapture()
        specRestore(snapshot)
        let accepted = Array(proposal.prefix(acceptedCount))
        let nextLogits: [Float]
        if accepted.isEmpty {
            nextLogits = currentLogits
        } else {
            nextLogits = try specReplay(accepted[...], startPos: startPos)
        }
        return DSparkVerificationResult(
            proposedCount: proposal.count,
            acceptedTokens: accepted,
            currentTarget: currentTarget,
            nextLogits: nextLogits,
            retainedVerifierFrontier: false)
    }

    // MARK: helpers

    private static func specComp(_ c: CompressorState?) -> SpecRecurrentState.Comp? {
        guard let c else { return nil }
        let coff = c.ratio == 4 ? 2 : 1
        let stateLen = coff * c.ratio * c.width
        return SpecRecurrentState.Comp(count: c.count,
                                       stateKv: readBack(c.stateKv, count: stateLen),
                                       stateScore: readBack(c.stateScore, count: stateLen))
    }

    private static func specRestoreComp(_ c: CompressorState?, from snap: SpecRecurrentState.Comp?) {
        guard let c else { return }
        guard let snap else { return }
        let coff = c.ratio == 4 ? 2 : 1
        let stateLen = coff * c.ratio * c.width
        // Lo snapshot viene da QUESTO stesso decoder nella stessa sessione:
        // una discrepanza di forma è un bug del chiamante, non input esterno.
        precondition(snap.stateKv.count == stateLen && snap.stateScore.count == stateLen,
                     "specRestoreComp: state length mismatch")
        writeBack(snap.stateKv, into: c.stateKv)
        writeBack(snap.stateScore, into: c.stateScore)
        c.count = snap.count
    }

    private static func readBack(_ t: GPUTensor, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let p = t.buffer.contents().advanced(by: t.byteOffset)
            .bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    private static func writeBack(_ a: [Float], into t: GPUTensor) {
        guard !a.isEmpty else { return }
        a.withUnsafeBufferPointer {
            _ = memcpy(t.buffer.contents().advanced(by: t.byteOffset),
                       $0.baseAddress!, a.count * 4)
        }
    }

    private static func greedyArgmax(_ logits: [Float]) -> Int {
        var bestID = 0
        var best = -Float.infinity
        for (id, value) in logits.enumerated() where value > best {
            best = value
            bestID = id
        }
        return bestID
    }
}
