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
            memcpy(t.buffer.contents().advanced(by: t.byteOffset),
                   $0.baseAddress!, a.count * 4)
        }
    }
}
