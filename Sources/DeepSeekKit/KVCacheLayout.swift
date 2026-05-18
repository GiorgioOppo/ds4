import Foundation

/// Layout binario del KV cache di un Transformer V4. Mappa ogni
/// tensor di stato accumulativo (Attention.kvCache, Compressor.kvState,
/// Compressor.scoreState, Indexer.kvCache e i suoi compressor) a un
/// offset+length dentro un payload contiguo, page-aligned, gestibile
/// via `KVCacheFile.region(offset:length:)`.
///
/// Usato dal wiring fisico cross-restart (TODO step B):
///   1. `compute(config:)` dà il `totalBytes` da passare a
///      `KVCacheFile.init(url:payloadBytes:)`.
///   2. Per ogni layer, `LayerOffsets` indica dove vivono i suoi
///      tensori dentro il payload.
///   3. `Assembly.makeCompressor(..., kvBacking: (file, offset))`
///      costruisce Tensor wrapper sul backing buffer invece di
///      `Tensor.empty`.
///   4. Cross-restart: alla riapertura del modello, KVCacheFile è
///      mappato dal disco, i wrapper Tensor re-puntano agli stessi
///      offset → la KV cache è "automaticamente" ripristinata.
///
/// **Page alignment** (16 KB su Apple Silicon): ogni `LayerOffsets`
/// inizia a un multiplo di 16384 byte per non incrociare boundary
/// di paging del kernel. Costa fino a 16 KB di slack per layer (vs
/// totale di GB), accettabile.
public struct KVCacheLayout: Sendable {

    /// Offset (in byte) e dimensione di ogni tensor KV per un singolo
    /// layer. I campi con valore `nil` indicano che il layer non
    /// alloca quel tensor (es. ratio=0 non ha compressor, ratio!=4
    /// non ha indexer).
    public struct LayerOffsets: Sendable, Codable, Equatable {
        public let layerIndex: Int
        public let compressRatio: Int

        /// Attention.kvCache: sempre presente.
        public let attnKVCache: Region

        /// Compressor.kvState: presente solo se ratio > 0.
        public let compressorKVState: Region?
        /// Compressor.scoreState: presente solo se ratio > 0.
        public let compressorScoreState: Region?

        /// Indexer.kvCache: presente solo se ratio == 4 (sparse attn).
        public let indexerKVCache: Region?
        /// Indexer.compressor.kvState: presente solo se ratio == 4.
        public let indexerCompressorKVState: Region?
        /// Indexer.compressor.scoreState: presente solo se ratio == 4.
        public let indexerCompressorScoreState: Region?
    }

    /// Una region nel payload del KVCacheFile.
    public struct Region: Sendable, Codable, Equatable {
        public let offset: Int     // byte offset dal payload start
        public let bytes: Int      // length in byte

        public init(offset: Int, bytes: Int) {
            self.offset = offset
            self.bytes = bytes
        }
    }

    public let layers: [LayerOffsets]
    public let mtpLayers: [LayerOffsets]
    public let totalBytes: Int

    /// Page size usato per allineare ogni region. 16384 = Apple
    /// Silicon page size (verificabile via `sysconf(_SC_PAGESIZE)`).
    public static let pageSize: Int = 16384

    public init(layers: [LayerOffsets],
                mtpLayers: [LayerOffsets],
                totalBytes: Int) {
        self.layers = layers
        self.mtpLayers = mtpLayers
        self.totalBytes = totalBytes
    }

    /// Calcola il layout dato un `ModelConfig`. La logica replica
    /// `Assembly.makeCompressor` / `makeIndexer` / `assemble` shape
    /// formulas per garantire consistenza con i Tensor effettivi
    /// che Assembly poi alloca.
    ///
    /// Future cleanup: estrarre le shape formulas in funzioni
    /// condivise con Assembly per single source of truth.
    public static func compute(config: ModelConfig) -> KVCacheLayout {
        var cursor = 0
        var layers: [LayerOffsets] = []
        layers.reserveCapacity(config.nLayers)

        for i in 0..<config.nLayers {
            let ratio = i < config.compressRatios.count
                ? config.compressRatios[i]
                : 0
            let (offsets, newCursor) = computeLayer(
                layerIndex: i, ratio: ratio,
                config: config, cursorStart: cursor)
            layers.append(offsets)
            cursor = newCursor
        }

        // MTP layers usano gli stessi compress ratios del main layer
        // corrispondente (mappatura `mtp[k] → layers[firstRatio4 + k]`
        // — vedi commento in Config.swift:388). Per il layout
        // riserviamo lo stesso spazio per ogni MTP.
        var mtpLayers: [LayerOffsets] = []
        for k in 0..<config.nMtpLayers {
            // Per default usiamo ratio=4 (caso tipico). Se serve
            // più preciso, il caller può specificare il ratio per
            // MTP. Per ora hard-code 4 = tipo del V4 reference.
            let ratio = 4
            let mtpIdx = 1000 + k  // convenzione MTP layer id
            let (offsets, newCursor) = computeLayer(
                layerIndex: mtpIdx, ratio: ratio,
                config: config, cursorStart: cursor)
            mtpLayers.append(offsets)
            cursor = newCursor
        }

        return KVCacheLayout(layers: layers,
                              mtpLayers: mtpLayers,
                              totalBytes: cursor)
    }

    private static func computeLayer(layerIndex: Int,
                                       ratio: Int,
                                       config: ModelConfig,
                                       cursorStart: Int)
        -> (LayerOffsets, Int)
    {
        var cursor = cursorStart
        let bs = config.maxBatchSize
        let headDim = config.headDim
        let f32Bytes = 4

        // 1. Attention.kvCache: [bs, kvCacheRows, headDim] f32
        let kvCacheRows = config.windowSize +
            (ratio > 0 ? config.maxSeqLen / ratio : 0)
        let attnRows = max(kvCacheRows, 1)
        let attnBytes = bs * attnRows * headDim * f32Bytes
        let attnRegion = Region(offset: cursor,
                                  bytes: attnBytes)
        cursor = roundUpToPage(cursor + attnBytes)

        // 2. Compressor.kvState/scoreState (solo se ratio > 0):
        // [bs, coff*ratio, coffHeadDim] f32 con coff=2 if ratio==4 else 1
        var compKV: Region? = nil
        var compScore: Region? = nil
        if ratio > 0 {
            let coff = (ratio == 4) ? 2 : 1
            let coffHeadDim = coff * headDim
            let compStateBytes = bs * coff * ratio * coffHeadDim * f32Bytes

            compKV = Region(offset: cursor, bytes: compStateBytes)
            cursor = roundUpToPage(cursor + compStateBytes)

            compScore = Region(offset: cursor, bytes: compStateBytes)
            cursor = roundUpToPage(cursor + compStateBytes)
        }

        // 3. Indexer (solo se ratio == 4): kvCache + compressor.
        var idxKV: Region? = nil
        var idxCompKV: Region? = nil
        var idxCompScore: Region? = nil
        if ratio == 4 {
            // Indexer kvCache: shape stessa formula di attn, ma con
            // indexHeadDim invece di headDim.
            let idxHeadDim = config.indexHeadDim
            let idxRows = max(kvCacheRows, 1)
            let idxBytes = bs * idxRows * idxHeadDim * f32Bytes
            idxKV = Region(offset: cursor, bytes: idxBytes)
            cursor = roundUpToPage(cursor + idxBytes)

            // Indexer compressor con ratio==4 → coff=2.
            let coff = 2
            let coffHeadDim = coff * idxHeadDim
            let cstateBytes = bs * coff * 4 * coffHeadDim * f32Bytes

            idxCompKV = Region(offset: cursor, bytes: cstateBytes)
            cursor = roundUpToPage(cursor + cstateBytes)

            idxCompScore = Region(offset: cursor, bytes: cstateBytes)
            cursor = roundUpToPage(cursor + cstateBytes)
        }

        return (LayerOffsets(layerIndex: layerIndex,
                              compressRatio: ratio,
                              attnKVCache: attnRegion,
                              compressorKVState: compKV,
                              compressorScoreState: compScore,
                              indexerKVCache: idxKV,
                              indexerCompressorKVState: idxCompKV,
                              indexerCompressorScoreState: idxCompScore),
                cursor)
    }

    @inline(__always)
    private static func roundUpToPage(_ x: Int) -> Int {
        let r = x % pageSize
        return r == 0 ? x : x + (pageSize - r)
    }

    // ---- Diagnostics ----

    /// Stampa human-readable del layout per debug. Mostra total +
    /// breakdown per layer.
    public func summary() -> String {
        var s = "KVCacheLayout: total \(totalBytes) bytes " +
                "(\(String(format: "%.2f", Double(totalBytes) / 1e9)) GB)\n"
        s += "  main layers (\(layers.count)):\n"
        for l in layers {
            s += "    L\(l.layerIndex) r=\(l.compressRatio) " +
                 "attn=\(l.attnKVCache.bytes)B"
            if let c = l.compressorKVState {
                s += " comp=\(c.bytes * 2)B"
            }
            if let i = l.indexerKVCache {
                s += " idx=\(i.bytes)B"
            }
            s += "\n"
        }
        if !mtpLayers.isEmpty {
            s += "  mtp layers (\(mtpLayers.count)):\n"
            for l in mtpLayers {
                s += "    M\(l.layerIndex - 1000) r=\(l.compressRatio) " +
                     "attn=\(l.attnKVCache.bytes)B\n"
            }
        }
        return s
    }
}
