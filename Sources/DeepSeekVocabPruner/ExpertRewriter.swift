import Foundation
import DeepSeekKit
import DeepSeekConverter   // `CancellationToken`

/// Phase 2 of the expert pruner: given an `ExpertKeepDecision`,
/// rewrite the checkpoint to a new directory with:
///
///  - expert tensors for dropped ids absent from the safetensors
///    (`layers.<L>.ffn.experts.<E>.{w1,w2,w3}.{weight,scale}` for
///    each `E ∈ decision.droppedIds[L]`),
///  - the gate weight row of every dropped expert in every layer
///    overwritten with a large-negative sentinel so the moe_gate
///    kernel's top-K never picks the missing slot,
///  - the gate bias of every dropped expert overwritten with the
///    same sentinel (when bias is present in the source),
///  - `tid2eid` table for the first `nHashLayers` layers remapped
///    so any entry pointing to a dropped expert is redirected to
///    the nearest kept expert (cosine similarity on the gate weight
///    row of the dropped expert vs. each kept candidate),
///  - `config.json` augmented with `pruned_experts: [[Int]]` so the
///    loader skips construction of the dropped Expert slots and
///    leaves `nil` in `MoEFFN.experts[]`,
///  - everything else (attention linears, norms, HC params, embed,
///    head, tokenizer, MTP) pass-through zero-copy.
///
/// Idempotent: refuses if `inputDir == outputDir`.
public enum ExpertRewriter {

    /// Sentinel value used to neutralise gate logits for dropped
    /// experts. Picked so `sqrt(softplus(value)) ≈ 0` even after
    /// any reasonable `x @ weight` perturbation in F32. -1e9 leaves
    /// ~10 orders of magnitude of headroom vs. the natural logit
    /// scale (typically O(1) to O(10)).
    public static let droppedGateLogit: Float = -1e9

    /// Run the rewrite. Emits `.shardWritten` per shard touched.
    @discardableResult
    public static func rewrite(
        inputDir: URL,
        outputDir: URL,
        decision: ExpertKeepDecision,
        alreadyCompletedShards: Set<String> = [],
        cancellation: CancellationToken? = nil,
        onShardDone: ((String) -> Void)? = nil,
        onEvent: (VocabPruneEvent) -> Void
    ) throws -> (bytesIn: UInt64, bytesOut: UInt64) {

        guard inputDir.standardizedFileURL
                != outputDir.standardizedFileURL else {
            throw NSError(domain: "ExpertRewriter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "Output directory must be different " +
                                      "from the input directory — refusing " +
                                      "to overwrite the source."])
        }

        let fm = FileManager.default
        try fm.createDirectory(at: outputDir,
                                withIntermediateDirectories: true)

        // 1. Load the source index.
        let indexURL = inputDir.appendingPathComponent("model.safetensors.index.json")
        let (sourceWeightMap, _) = try loadIndex(indexURL)

        // 2. Group by shard.
        var shardMap: [String: [String]] = [:]
        for (name, shard) in sourceWeightMap {
            shardMap[shard, default: []].append(name)
        }
        let shards = shardMap.keys.sorted()

        // 3. Index dropped sets per layer for O(1) lookup.
        let droppedByLayer: [Int: Set<Int>] = Dictionary(
            uniqueKeysWithValues: decision.droppedIds.enumerated().map {
                ($0.offset, Set($0.element))
            })

        // 4. Pre-collect the gate weight rows of dropped experts in
        //    each layer. Needed by the rewriter to remap `tid2eid`
        //    via cosine similarity, and to know what the dropped
        //    row's L2-norm was for the negative-sentinel overwrite.
        let gateRows = try collectGateRows(inputDir: inputDir,
                                            weightMap: sourceWeightMap,
                                            nLayers: decision.nLayers,
                                            nExperts: decision.nRoutedExperts)

        var bytesIn: UInt64 = 0
        var bytesOut: UInt64 = 0
        var newWeightMap: [String: String] = [:]

        for (i, shardName) in shards.enumerated() {
            try Task.checkCancellation()
            try cancellation?.throwIfCancelled()
            let inURL = inputDir.appendingPathComponent(shardName)
            let outURL = outputDir.appendingPathComponent(shardName)

            let inAttrs = try fm.attributesOfItem(atPath: inURL.path)
            if let sz = inAttrs[.size] as? UInt64 { bytesIn += sz }

            if alreadyCompletedShards.contains(shardName)
                && fm.fileExists(atPath: outURL.path)
            {
                let outAttrs = try? fm.attributesOfItem(atPath: outURL.path)
                if let sz = outAttrs?[FileAttributeKey.size] as? UInt64 {
                    bytesOut += sz
                }
            } else {
                try rewriteShard(inURL: inURL,
                                  outURL: outURL,
                                  decision: decision,
                                  droppedByLayer: droppedByLayer,
                                  gateRows: gateRows,
                                  cancellation: cancellation)
                let outAttrs = try fm.attributesOfItem(atPath: outURL.path)
                if let sz = outAttrs[.size] as? UInt64 { bytesOut += sz }
                onShardDone?(shardName)
            }

            // Update the weight_map, skipping dropped experts entirely.
            for name in shardMap[shardName]! {
                if isDroppedExpertTensor(name, droppedByLayer: droppedByLayer) {
                    continue
                }
                newWeightMap[name] = shardName
            }
            onEvent(.shardWritten(i: i + 1, total: shards.count))
        }

        // 5. Rewrite index.json.
        try writeIndex(at: outputDir.appendingPathComponent("model.safetensors.index.json"),
                       weightMap: newWeightMap,
                       totalSize: bytesOut)

        // 6. Rewrite config.json with the `pruned_experts` array.
        try rewriteConfigJSON(inputDir: inputDir,
                               outputDir: outputDir,
                               decision: decision)

        // 7. Copy tokenizer.json verbatim if present (the expert
        //    rewriter does NOT change the vocab).
        let tokInURL = inputDir.appendingPathComponent("tokenizer.json")
        let tokOutURL = outputDir.appendingPathComponent("tokenizer.json")
        if fm.fileExists(atPath: tokInURL.path),
           !fm.fileExists(atPath: tokOutURL.path)
        {
            try fm.copyItem(at: tokInURL, to: tokOutURL)
        }

        return (bytesIn, bytesOut)
    }

    // MARK: - Per-shard rewrite

    private static func rewriteShard(
        inURL: URL,
        outURL: URL,
        decision: ExpertKeepDecision,
        droppedByLayer: [Int: Set<Int>],
        gateRows: [Int: [Int: [Float]]],
        cancellation: CancellationToken? = nil
    ) throws {
        throw NSError(domain: "ExpertRewriter", code: 99,
                      userInfo: [NSLocalizedDescriptionKey: "Expert pruning is not supported with the new MLX backend yet."])
    }

    // MARK: - Tensor patches

    /// Rewrite `layers.<L>.ffn.gate.weight` (shape [nExperts, dim])
    /// row-wise: copies the source bytes verbatim, then overwrites
    /// each dropped row with the negative sentinel encoded in the
    /// tensor's native dtype.
    private static func patchGateWeight(
        inURL: URL,
        absOffset: Int,
        byteCount: Int,
        shape: [Int],
        dtype: String,
        dropped: Set<Int>
    ) throws -> Data {
        precondition(shape.count == 2,
                     "gate.weight expected rank-2, got \(shape)")
        let nExperts = shape[0]
        let dim = shape[1]
        let bytesPerElem = bytesPerElement(forDtype: dtype)
        let bytesPerRow = dim * bytesPerElem
        precondition(byteCount == nExperts * bytesPerRow,
                     "gate.weight byteCount mismatch")

        let fh = try FileHandle(forReadingFrom: inURL)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(absOffset))
        guard let bytes = try fh.read(upToCount: byteCount),
              bytes.count == byteCount else {
            throw NSError(domain: "ExpertRewriter", code: 41,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "short read on gate.weight"])
        }
        var data = Data(bytes)

        let pattern = encodeSentinelRow(value: droppedGateLogit,
                                         dtype: dtype,
                                         count: dim)
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            pattern.withUnsafeBytes { src in
                guard let srcBase = src.baseAddress else { return }
                for e in dropped {
                    memcpy(base.advanced(by: e * bytesPerRow),
                           srcBase,
                           bytesPerRow)
                }
            }
        }
        return data
    }

    /// Rewrite `layers.<L>.ffn.gate.bias` (shape [nExperts]) by
    /// overwriting the scalar at each dropped expert id with the
    /// negative sentinel.
    private static func patchGateBias(
        inURL: URL,
        absOffset: Int,
        byteCount: Int,
        shape: [Int],
        dtype: String,
        dropped: Set<Int>
    ) throws -> Data {
        precondition(shape.count == 1,
                     "gate.bias expected rank-1, got \(shape)")
        let nExperts = shape[0]
        let bytesPerElem = bytesPerElement(forDtype: dtype)
        precondition(byteCount == nExperts * bytesPerElem,
                     "gate.bias byteCount mismatch")

        let fh = try FileHandle(forReadingFrom: inURL)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(absOffset))
        guard let bytes = try fh.read(upToCount: byteCount),
              bytes.count == byteCount else {
            throw NSError(domain: "ExpertRewriter", code: 42,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "short read on gate.bias"])
        }
        var data = Data(bytes)

        let pattern = encodeSentinelRow(value: droppedGateLogit,
                                         dtype: dtype,
                                         count: 1)
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            pattern.withUnsafeBytes { src in
                guard let srcBase = src.baseAddress else { return }
                for e in dropped {
                    memcpy(base.advanced(by: e * bytesPerElem),
                           srcBase,
                           bytesPerElem)
                }
            }
        }
        return data
    }

    /// Rewrite `layers.<L>.ffn.gate.tid2eid` (shape [vocab, topK]).
    /// For each cell that maps to a dropped expert, replace with the
    /// cosine-nearest kept expert (in the same layer) based on the
    /// source gate weight rows.
    private static func patchTid2Eid(
        inURL: URL,
        absOffset: Int,
        byteCount: Int,
        shape: [Int],
        dtype: String,
        dropped: Set<Int>,
        keptIds: [Int],
        gateRows: [Int: [Float]]
    ) throws -> Data {
        precondition(shape.count == 2,
                     "tid2eid expected rank-2, got \(shape)")
        let cells = shape[0] * shape[1]

        // Build remap[droppedExpertId] -> nearestKeptExpertId.
        var remap: [Int: Int] = [:]
        for d in dropped {
            guard let droppedRow = gateRows[d] else {
                // No source row available (unexpected). Fall back to
                // the smallest kept id.
                remap[d] = keptIds.first ?? 0
                continue
            }
            var bestEid = keptIds.first ?? 0
            var bestSim = -Float.infinity
            for k in keptIds {
                guard let keptRow = gateRows[k] else { continue }
                let sim = cosineSimilarity(droppedRow, keptRow)
                if sim > bestSim { bestSim = sim; bestEid = k }
            }
            remap[d] = bestEid
        }

        let fh = try FileHandle(forReadingFrom: inURL)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(absOffset))
        guard let bytes = try fh.read(upToCount: byteCount),
              bytes.count == byteCount else {
            throw NSError(domain: "ExpertRewriter", code: 43,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "short read on tid2eid"])
        }
        var data = Data(bytes)

        // Apply remap in the tensor's native int dtype.
        switch dtype {
        case "I32":
            data.withUnsafeMutableBytes { raw in
                guard let p = raw.baseAddress?
                        .assumingMemoryBound(to: Int32.self) else { return }
                for i in 0..<cells {
                    let v = Int(p[i])
                    if let r = remap[v] { p[i] = Int32(r) }
                }
            }
        case "I64":
            data.withUnsafeMutableBytes { raw in
                guard let p = raw.baseAddress?
                        .assumingMemoryBound(to: Int64.self) else { return }
                for i in 0..<cells {
                    let v = Int(p[i])
                    if let r = remap[v] { p[i] = Int64(r) }
                }
            }
        default:
            throw NSError(domain: "ExpertRewriter", code: 44,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "tid2eid unexpected dtype \(dtype)"])
        }
        return data
    }

    // MARK: - Gate-row collection (needed for tid2eid cosine remap)

    /// Read the gate weight row vectors of each layer's experts
    /// (`[nExperts][dim] f32 view`) into host memory so the tid2eid
    /// remap can compute cosine similarity. Only the rows of dropped
    /// experts AND kept experts in layers with `tid2eid` (i.e. the
    /// first `nHashLayers`) actually need to be loaded — but we'd
    /// have to know `nHashLayers` upfront. Simpler: load everything,
    /// it's small (`nLayers · nExperts · dim · 4 bytes`).
    ///
    /// Returns `[layerId: [expertId: rowF32]]`. Missing entries
    /// (e.g. layer has no `gate.weight` because it's a non-MoE
    /// layer, or expert id out of range) are simply absent.
    private static func collectGateRows(
        inputDir: URL,
        weightMap: [String: String],
        nLayers: Int,
        nExperts: Int
    ) throws -> [Int: [Int: [Float]]] {
        return [:]
    }

    // MARK: - Name parsing

    /// Returns the layer id for a tensor name of shape
    /// `layers.<L>.ffn.experts.<E>.{w1|w2|w3}.{weight|scale}` IFF
    /// that expert id is in the dropped set; otherwise nil.
    static func isDroppedExpertTensor(_ name: String,
                                       droppedByLayer: [Int: Set<Int>]) -> Bool
    {
        guard let (layerId, expertId) = parseExpertTensor(name) else { return false }
        return droppedByLayer[layerId]?.contains(expertId) ?? false
    }

    /// `layers.<L>.ffn.experts.<E>.<rest>` → `(L, E)`. Returns nil for
    /// names that don't match this prefix exactly.
    static func parseExpertTensor(_ name: String) -> (Int, Int)? {
        let parts = name.split(separator: ".", maxSplits: 5,
                                omittingEmptySubsequences: false)
        // Expected: ["layers", "<L>", "ffn", "experts", "<E>", "<rest>"]
        guard parts.count >= 6,
              parts[0] == "layers",
              parts[2] == "ffn",
              parts[3] == "experts" else { return nil }
        guard let L = Int(parts[1]), let E = Int(parts[4]) else { return nil }
        return (L, E)
    }

    /// `layers.<L>.ffn.gate.weight` → L. Nil otherwise.
    static func parseGateWeightLayer(_ name: String) -> Int? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 5,
              parts[0] == "layers",
              parts[2] == "ffn",
              parts[3] == "gate",
              parts[4] == "weight" else { return nil }
        return Int(parts[1])
    }

    /// `layers.<L>.ffn.gate.bias` → L. Nil otherwise.
    static func parseGateBiasLayer(_ name: String) -> Int? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 5,
              parts[0] == "layers",
              parts[2] == "ffn",
              parts[3] == "gate",
              parts[4] == "bias" else { return nil }
        return Int(parts[1])
    }

    /// `layers.<L>.ffn.gate.tid2eid` → L. Nil otherwise.
    static func parseTid2EidLayer(_ name: String) -> Int? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 5,
              parts[0] == "layers",
              parts[2] == "ffn",
              parts[3] == "gate",
              parts[4] == "tid2eid" else { return nil }
        return Int(parts[1])
    }

    // MARK: - Dtype-aware helpers

    static func bytesPerElement(forDtype dtype: String) -> Int {
        switch dtype {
        case "BF16", "F16", "I16", "U16": return 2
        case "F32", "I32", "U32":         return 4
        case "F64", "I64", "U64":         return 8
        case "BOOL", "I8", "U8":          return 1
        case "F8_E4M3", "F8_E5M2":        return 1
        default:
            fatalError("ExpertRewriter: unknown safetensors dtype `\(dtype)`")
        }
    }

    /// Encode a single sentinel value repeated `count` times in the
    /// tensor's native dtype. Used to overwrite gate weight rows /
    /// bias scalars for dropped experts.
    static func encodeSentinelRow(value: Float, dtype: String, count: Int) -> Data {
        switch dtype {
        case "F32":
            var arr = [Float](repeating: value, count: count)
            return arr.withUnsafeMutableBufferPointer { Data(buffer: $0) }
        case "F16":
            var arr = [UInt16](repeating: f32ToF16Bits(value), count: count)
            return arr.withUnsafeMutableBufferPointer { Data(buffer: $0) }
        case "BF16":
            var arr = [UInt16](repeating: f32ToBF16Bits(value), count: count)
            return arr.withUnsafeMutableBufferPointer { Data(buffer: $0) }
        default:
            fatalError("ExpertRewriter: encodeSentinelRow unsupported dtype \(dtype)")
        }
    }

    /// Decode a row of `count` elements stored as bytes of `dtype`
    /// into host-side F32 for the cosine similarity remap.
    static func decodeRowAsF32(bytes: Data, dtype: String, count: Int) -> [Float] {
        switch dtype {
        case "F32":
            return bytes.withUnsafeBytes { raw -> [Float] in
                let p = raw.bindMemory(to: Float.self)
                return Array(p[0..<count])
            }
        case "F16":
            return bytes.withUnsafeBytes { raw -> [Float] in
                let p = raw.bindMemory(to: UInt16.self)
                var out = [Float](repeating: 0, count: count)
                for i in 0..<count { out[i] = f16BitsToF32(p[i]) }
                return out
            }
        case "BF16":
            return bytes.withUnsafeBytes { raw -> [Float] in
                let p = raw.bindMemory(to: UInt16.self)
                var out = [Float](repeating: 0, count: count)
                for i in 0..<count { out[i] = bf16BitsToF32(p[i]) }
                return out
            }
        default:
            fatalError("ExpertRewriter: decodeRowAsF32 unsupported dtype \(dtype)")
        }
    }

    // F32 ↔ BF16 conversion. BF16 = sign + 8-bit exp + 7-bit mantissa
    // (truncate the trailing 16 bits of an F32 with round-to-nearest-even).
    @inline(__always)
    static func f32ToBF16Bits(_ x: Float) -> UInt16 {
        var bits = x.bitPattern
        // Round-to-nearest-even on the bottom 16 bits.
        let lsb = (bits >> 16) & 1
        let roundBias = UInt32(0x7FFF) + lsb
        bits = bits &+ roundBias
        return UInt16(bits >> 16)
    }
    @inline(__always)
    static func bf16BitsToF32(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    // F32 ↔ F16 (IEEE half). Simple unsaturated path; -1e9 in F16 is
    // -inf which is fine for the sentinel.
    @inline(__always)
    static func f32ToF16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF) - 127 + 15
        let mant = UInt16((bits >> 13) & 0x3FF)
        if exp <= 0 { return sign | 0x0000 }
        if exp >= 31 { return sign | 0x7C00 }   // ±inf
        return sign | UInt16(exp << 10) | mant
    }
    @inline(__always)
    static func f16BitsToF32(_ b: UInt16) -> Float {
        let sign = UInt32(b & 0x8000) << 16
        let exp = Int((b >> 10) & 0x1F)
        let mant = UInt32(b & 0x3FF)
        if exp == 0 && mant == 0 { return Float(bitPattern: sign) }
        if exp == 31 { return Float(bitPattern: sign | 0x7F800000 | (mant << 13)) }
        let normExp = exp == 0 ? 1 : exp
        let bits = sign | UInt32(normExp + 127 - 15) << 23 | (mant << 13)
        return Float(bitPattern: bits)
    }

    @inline(__always)
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = (na * nb).squareRoot()
        guard denom > 1e-12 else { return 0 }
        return dot / denom
    }

    // MARK: - index.json / config.json

    private static func loadIndex(_ url: URL) throws
        -> (weightMap: [String: String], totalSize: UInt64?)
    {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ExpertRewriter", code: 10,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "index.json not a JSON object"])
        }
        guard let wm = root["weight_map"] as? [String: String] else {
            throw NSError(domain: "ExpertRewriter", code: 11,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "index.json missing weight_map"])
        }
        let total = (root["metadata"] as? [String: Any])?["total_size"] as? UInt64
        return (wm, total)
    }

    private static func writeIndex(at url: URL,
                                     weightMap: [String: String],
                                     totalSize: UInt64) throws {
        let obj: [String: Any] = [
            "metadata": ["total_size": totalSize] as [String: Any],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: obj,
            options: [.sortedKeys, .prettyPrinted])
        try data.write(to: url)
    }

    /// Set `config.json["pruned_experts"] = decision.droppedIds`.
    /// All other fields preserved. If the source config doesn't
    /// exist, writes a minimal one with just this field — same
    /// permissive behavior as `VocabRewriter.rewriteConfigJSON`.
    static func rewriteConfigJSON(inputDir: URL,
                                    outputDir: URL,
                                    decision: ExpertKeepDecision) throws {
        let inURL = inputDir.appendingPathComponent("config.json")
        let outURL = outputDir.appendingPathComponent("config.json")
        let prunedExpertsField: [[Int]] = decision.droppedIds
        guard let data = try? Data(contentsOf: inURL),
              var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else
        {
            let minimal: [String: Any] = ["pruned_experts": prunedExpertsField]
            let bytes = try JSONSerialization.data(withJSONObject: minimal,
                                                   options: [.sortedKeys, .prettyPrinted])
            try bytes.write(to: outURL)
            return
        }
        obj["pruned_experts"] = prunedExpertsField
        let bytes = try JSONSerialization.data(withJSONObject: obj,
                                               options: [.sortedKeys, .prettyPrinted])
        try bytes.write(to: outURL)
    }

    /// Replicated from `VocabRewriter` because the field isn't
    /// exposed publicly by `SafeTensorsFile`.
    private static func readDataStart(_ url: URL) throws -> Int {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let head = try fh.read(upToCount: 8), head.count == 8 else {
            throw NSError(domain: "ExpertRewriter", code: 30,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "safetensors file too small (no header len)"])
        }
        let headerLen = head.withUnsafeBytes { raw -> UInt64 in
            return raw.load(as: UInt64.self).littleEndian
        }
        return 8 + Int(headerLen)
    }
}
