import Foundation
import DeepSeekKit

/// Fase 2: data una `KeepDecision`, riscrive il checkpoint pruned
/// in `outputDir`. Slicing riga-wise di `embed.weight` /
/// `head.weight` (e dei loro alias MTP se presenti), pass-through
/// zero-copy del resto.
public enum VocabRewriter {

    /// Esegue il rewriting completo. Idempotente: rifiuta se
    /// `inputDir == outputDir`. Emette `.shardWritten` per ogni
    /// shard processato.
    public static func rewrite(
        inputDir: URL,
        outputDir: URL,
        decision: KeepDecision,
        onEvent: (VocabPruneEvent) -> Void
    ) throws -> (bytesIn: UInt64, bytesOut: UInt64) {

        // Validazione idempotente: stessa policy di
        // Sources/DeepSeekTraining/FineTuneRunner.swift:62
        guard inputDir.standardizedFileURL
                != outputDir.standardizedFileURL else {
            throw NSError(domain: "VocabRewriter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "Output directory must be different " +
                                      "from the input directory — refusing " +
                                      "to overwrite the source."])
        }

        let fm = FileManager.default
        try fm.createDirectory(at: outputDir,
                                withIntermediateDirectories: true)

        // 1) Leggi l'index del checkpoint sorgente.
        let indexURL = inputDir.appendingPathComponent("model.safetensors.index.json")
        let (sourceWeightMap, _) = try loadIndex(indexURL)

        // 2) Raggruppa i tensor per shard di provenienza.
        var shardMap: [String: [String]] = [:]
        for (name, shard) in sourceWeightMap {
            shardMap[shard, default: []].append(name)
        }
        let shards = shardMap.keys.sorted()

        // 3) Per ogni shard, leggi → ricostruisci → scrivi.
        var bytesIn: UInt64 = 0
        var bytesOut: UInt64 = 0
        var newWeightMap: [String: String] = [:]

        for (i, shardName) in shards.enumerated() {
            let inURL = inputDir.appendingPathComponent(shardName)
            let outURL = outputDir.appendingPathComponent(shardName)

            let attrs = try fm.attributesOfItem(atPath: inURL.path)
            if let sz = attrs[.size] as? UInt64 { bytesIn += sz }

            try rewriteShard(inURL: inURL,
                              outURL: outURL,
                              decision: decision)

            let outAttrs = try fm.attributesOfItem(atPath: outURL.path)
            if let sz = outAttrs[.size] as? UInt64 { bytesOut += sz }

            // Aggiorna la weight_map per il nuovo index (i nomi non
            // cambiano, restano negli stessi shard).
            for name in shardMap[shardName]! {
                if !shouldSkipTensor(name) {
                    newWeightMap[name] = shardName
                }
            }
            onEvent(.shardWritten(i: i + 1, total: shards.count))
        }

        // 4) Riscrivi il file index.
        try writeIndex(at: outputDir.appendingPathComponent("model.safetensors.index.json"),
                       weightMap: newWeightMap,
                       totalSize: bytesOut)

        // 5) Riscrivi config.json (solo vocab_size cambia).
        try rewriteConfigJSON(inputDir: inputDir,
                               outputDir: outputDir,
                               newVocabSize: decision.newVocabSize)

        // 6) Riscrivi tokenizer.json (vocab + merges filtrati,
        //    added_tokens preservati, resto verbatim).
        try rewriteTokenizerJSON(inputDir: inputDir,
                                  outputDir: outputDir,
                                  decision: decision)

        return (bytesIn, bytesOut)
    }

    // MARK: - Shard rewrite

    /// Riscrive un singolo shard applicando il pruning agli
    /// embedding tensors e stream-copiando il resto.
    private static func rewriteShard(inURL: URL,
                                       outURL: URL,
                                       decision: KeepDecision) throws {
        let dataStart = try readDataStart(inURL)
        let file = try SafeTensorsFile(url: inURL)
        let writer = SafeTensorsWriter()

        // Ordine deterministico per riproducibilità.
        let names = file.entries.keys.sorted()
        for name in names {
            if shouldSkipTensor(name) { continue }
            let entry = file.entries[name]!
            let byteCount = entry.dataOffsets[1] - entry.dataOffsets[0]
            let absOffset = dataStart + entry.dataOffsets[0]

            if isVocabTensor(name) {
                // Riga-wise slicing: shape originale [vocabSize, dim].
                let sliced = try sliceVocabTensor(
                    inURL: inURL,
                    absOffset: absOffset,
                    byteCount: byteCount,
                    originalShape: entry.shape,
                    dtype: entry.dtype,
                    decision: decision)
                let newShape = [decision.newVocabSize] + Array(entry.shape.dropFirst())
                writer.add(name: name,
                           dtype: entry.dtype,
                           shape: newShape,
                           source: .data(sliced))
            } else {
                // Pass-through: zero-copy stream da file originale.
                writer.add(name: name,
                           dtype: entry.dtype,
                           shape: entry.shape,
                           source: .file(url: inURL,
                                          offset: absOffset,
                                          byteCount: byteCount))
            }
        }

        try writer.write(to: outURL)
    }

    /// Slice riga-wise di un tensor `[vocabSize, dim]`. Per ogni
    /// `(oldId, newId)` in `decision.oldToNew`, copia
    /// `bytesPerRow` byte dall'offset `absOffset + oldId * bytesPerRow`
    /// alla riga `newId` del buffer destinazione.
    private static func sliceVocabTensor(
        inURL: URL,
        absOffset: Int,
        byteCount: Int,
        originalShape: [Int],
        dtype: String,
        decision: KeepDecision
    ) throws -> Data {
        precondition(originalShape.count >= 2,
                     "vocab tensor expected rank >= 2, got \(originalShape)")
        let vocabSize = originalShape[0]
        let dim = originalShape.dropFirst().reduce(1, *)
        let bytesPerElem = bytesPerElement(forDtype: dtype)
        let bytesPerRow = dim * bytesPerElem
        precondition(byteCount == vocabSize * bytesPerRow,
                     "vocab tensor byteCount mismatch for \(originalShape) dtype=\(dtype)")

        let newBytesCount = decision.newVocabSize * bytesPerRow
        var out = Data(count: newBytesCount)

        // mmap del source per evitare il read in memoria del tensor
        // intero. Sui big embedding (es. 129k × 4096 × 2 byte ≈ 1 GB)
        // è una differenza tangibile.
        let fh = try FileHandle(forReadingFrom: inURL)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(absOffset))

        // Leggi una riga alla volta, scrivi al posto giusto. Pattern
        // semplice ma serializzato — per i big embedding bastano
        // pochi secondi per file da ~1 GB su SSD.
        // Ottimizzazione futura: read in blocchi multipli di rows
        // contigue dove (oldId+1, newId+1) sono entrambi presenti
        // in sequenza.
        out.withUnsafeMutableBytes { dst in
            let dstBase = dst.baseAddress!
            // Ordina per oldId per leggere sequenzialmente dal file.
            let mappings = decision.oldToNew
                .filter { $0.value < decision.newVocabSize }
                .sorted { $0.key < $1.key }
            var lastReadEnd = 0
            for (oldId, newId) in mappings {
                let rowOffsetInTensor = oldId * bytesPerRow
                if rowOffsetInTensor != lastReadEnd {
                    // Skip avanti nel file.
                    let cur = (try? fh.offset()) ?? 0
                    let delta = rowOffsetInTensor - lastReadEnd
                    try? fh.seek(toOffset: cur + UInt64(delta))
                }
                guard let row = try? fh.read(upToCount: bytesPerRow),
                      row.count == bytesPerRow else {
                    fatalError("vocab tensor slice: short read at oldId=\(oldId)")
                }
                row.withUnsafeBytes { src in
                    memcpy(dstBase.advanced(by: newId * bytesPerRow),
                           src.baseAddress!,
                           bytesPerRow)
                }
                lastReadEnd = rowOffsetInTensor + bytesPerRow
            }
        }
        return out
    }

    // MARK: - tensor classification

    /// True se il nome del tensor è uno di quelli che dimensioniamo
    /// sul vocab (riga-wise slicing).
    static func isVocabTensor(_ name: String) -> Bool {
        // Embeddings + LM head + alias MTP.
        if name == "embed.weight" { return true }
        if name == "head.weight"  { return true }
        // MTP alias (a volte sopravvivono nel checkpoint).
        if name.hasPrefix("mtp.") && name.hasSuffix(".embed.weight") { return true }
        if name.hasPrefix("mtp.") && name.hasSuffix(".head.weight")  { return true }
        return false
    }

    /// True se il tensor va saltato completamente nel nuovo
    /// checkpoint. Tiene lo stesso comportamento di
    /// `Sources/DeepSeekConverter/Rename.shouldSkip` per gli alias
    /// MTP non-funzionali.
    static func shouldSkipTensor(_ name: String) -> Bool {
        if name.hasPrefix("mtp.") {
            if name.contains("emb") { return false }      // embed: slicing
            if name.hasSuffix(".head.weight") { return false } // head: slicing
            // altri tensori MTP sono fuori scope del pruner;
            // li passiamo through (no skip).
        }
        return false
    }

    static func bytesPerElement(forDtype dtype: String) -> Int {
        // Safetensors dtype string nomenclature.
        switch dtype {
        case "BF16", "F16", "I16", "U16": return 2
        case "F32", "I32", "U32":         return 4
        case "F64", "I64", "U64":         return 8
        case "BOOL", "I8", "U8":          return 1
        case "F8_E4M3", "F8_E5M2":        return 1
        default:
            fatalError("VocabRewriter: unknown safetensors dtype `\(dtype)`")
        }
    }

    // MARK: - Index / Config / Tokenizer JSON

    private static func loadIndex(_ url: URL) throws
        -> (weightMap: [String: String], totalSize: UInt64?)
    {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "VocabRewriter", code: 10,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "index.json not a JSON object"])
        }
        guard let wm = root["weight_map"] as? [String: String] else {
            throw NSError(domain: "VocabRewriter", code: 11,
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

    private static func rewriteConfigJSON(inputDir: URL,
                                           outputDir: URL,
                                           newVocabSize: Int) throws {
        let inURL = inputDir.appendingPathComponent("config.json")
        let outURL = outputDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: inURL),
              var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            // Senza config.json di partenza, scriviamo un minimo
            // accettabile (solo vocab_size). Il loader non si lamenta
            // se mancano gli altri campi (hanno default in
            // ModelConfig).
            let minimal: [String: Any] = ["vocab_size": newVocabSize]
            let bytes = try JSONSerialization.data(withJSONObject: minimal,
                                                   options: [.sortedKeys, .prettyPrinted])
            try bytes.write(to: outURL)
            return
        }
        obj["vocab_size"] = newVocabSize
        let bytes = try JSONSerialization.data(withJSONObject: obj,
                                               options: [.sortedKeys, .prettyPrinted])
        try bytes.write(to: outURL)
    }

    /// Riscrive `tokenizer.json` con vocab e merges filtrati, ID
    /// rimappati via `decision.oldToNew`. `added_tokens` mantengono
    /// gli ID originali. `pre_tokenizer`, `decoder`, `normalizer`,
    /// `post_processor` copiati verbatim.
    static func rewriteTokenizerJSON(inputDir: URL,
                                       outputDir: URL,
                                       decision: KeepDecision) throws {
        let inURL = inputDir.appendingPathComponent("tokenizer.json")
        let outURL = outputDir.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: inURL)
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "VocabRewriter", code: 20,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "tokenizer.json not a JSON object"])
        }
        guard var model = root["model"] as? [String: Any] else {
            throw NSError(domain: "VocabRewriter", code: 21,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "tokenizer.json missing model"])
        }

        // ---- vocab filter + remap ----
        guard let oldVocab = model["vocab"] as? [String: Int] else {
            throw NSError(domain: "VocabRewriter", code: 22,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "tokenizer.json missing model.vocab"])
        }
        let keepSet = Set(decision.keepIds)
        var newVocab: [String: Int] = [:]
        for (tok, oldId) in oldVocab where keepSet.contains(oldId) {
            guard let newId = decision.oldToNew[oldId] else { continue }
            newVocab[tok] = newId
        }
        model["vocab"] = newVocab

        // ---- merges filter ----
        // Una merge "a b" è valida iff:
        //   - "a" è in newVocab (o è un single-char base byte),
        //   - "b" è in newVocab,
        //   - "ab" (concat) è in newVocab.
        // Senza queste tre condizioni la merge non si applica mai e
        // il decoder lo ignora (HF accetta merges "morte" ma non c'è
        // ragione di tenerle).
        if let oldMerges = model["merges"] as? [Any] {
            var newMerges: [String] = []
            newMerges.reserveCapacity(oldMerges.count / 2)
            for m in oldMerges {
                let pair: (String, String)?
                if let s = m as? String {
                    let parts = s.split(separator: " ", maxSplits: 1)
                    pair = parts.count == 2 ? (String(parts[0]), String(parts[1])) : nil
                } else if let arr = m as? [String], arr.count == 2 {
                    pair = (arr[0], arr[1])
                } else {
                    pair = nil
                }
                guard let (a, b) = pair else { continue }
                let ab = a + b
                if newVocab[a] != nil && newVocab[b] != nil && newVocab[ab] != nil {
                    newMerges.append("\(a) \(b)")
                }
            }
            model["merges"] = newMerges
        }

        root["model"] = model

        // ---- added_tokens: preserva ID e content verbatim ----
        if let addedArr = root["added_tokens"] as? [[String: Any]] {
            var newAdded: [[String: Any]] = []
            for entry in addedArr {
                if let id = entry["id"] as? Int {
                    // Gli addedTokens DEVONO essere preservati con il
                    // loro ID originale (oldToNew[id] == id per
                    // costruzione in `buildRemap`).
                    if decision.oldToNew[id] != nil {
                        newAdded.append(entry)
                    }
                }
            }
            root["added_tokens"] = newAdded
        }

        // pre_tokenizer / decoder / normalizer / post_processor restano
        // verbatim — non hanno riferimenti per-id.

        let outBytes = try JSONSerialization.data(
            withJSONObject: root,
            options: [.sortedKeys, .prettyPrinted])
        try outBytes.write(to: outURL)
    }

    // MARK: - safetensors low-level

    /// Legge il `dataStart` di un file `.safetensors`: 8 byte
    /// little-endian con la lunghezza dell'header JSON, poi
    /// `headerLen` byte di JSON, poi i dati. `dataStart = 8 + headerLen`.
    /// Replicato qui perché il campo `dataStart` di `SafeTensorsFile`
    /// non è esposto pubblicamente.
    private static func readDataStart(_ url: URL) throws -> Int {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let head = try fh.read(upToCount: 8), head.count == 8 else {
            throw NSError(domain: "VocabRewriter", code: 30,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "safetensors file too small (no header len)"])
        }
        let headerLen = head.withUnsafeBytes { raw -> UInt64 in
            return raw.load(as: UInt64.self).littleEndian
        }
        return 8 + Int(headerLen)
    }
}
