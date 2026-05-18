import Foundation
import DeepSeekKit

/// Preview di un singolo token droppato dall'analyzer. Il content
/// è la stringa byte-encoded come appare nel vocab originale (es.
/// `ĠBuongiorno` per "Buongiorno" preceduto da spazio).
public struct DroppedTokenPreview: Sendable, Codable, Equatable {
    public let id: Int
    public let content: String
    public let count: Int
    public init(id: Int, content: String, count: Int) {
        self.id = id
        self.content = content
        self.count = count
    }
}

/// Decisione di pruning prodotta da `VocabAnalyzer`: chi tiene,
/// chi va via, e la mappatura old→new ID.
public struct KeepDecision: Sendable, Codable {
    /// Set dei `tokenId` (vocab originale) che sopravvivono.
    public let keepIds: [Int]               // serializzato come array per Codable

    /// Mappa `oldId → newId`. Gli ID degli `addedTokens` sono
    /// preservati (nuovo == vecchio). Gli altri sono ricompattati a
    /// partire da 0, saltando i buchi degli addedTokens.
    public let oldToNew: [Int: Int]

    /// Dimensione del nuovo vocab = max(newId) + 1.
    public let newVocabSize: Int

    /// Dimensione del vocab originale (totalVocab).
    public let oldVocabSize: Int

    /// Copertura cumulativa raggiunta (0..1).
    public let coveragePct: Double

    /// Top-N (default 50) dei token che sono stati droppati ma che
    /// AVEVANO una frequenza nel corpus. Utile come "anteprima
    /// dell'impatto" nella UI / dry-run. Non include i token che
    /// sono stati droppati dal force-exclude (script foreign) e che
    /// avevano count 0 nel corpus — quelli sono virtualmente
    /// infiniti e poco informativi.
    public let previewDropped: [DroppedTokenPreview]

    public init(keepIds: [Int],
                oldToNew: [Int: Int],
                newVocabSize: Int,
                oldVocabSize: Int,
                coveragePct: Double,
                previewDropped: [DroppedTokenPreview] = []) {
        self.keepIds = keepIds
        self.oldToNew = oldToNew
        self.newVocabSize = newVocabSize
        self.oldVocabSize = oldVocabSize
        self.coveragePct = coveragePct
        self.previewDropped = previewDropped
    }

    // Codable custom per `[Int: Int]` (Codable di default usa
    // String keys; lo serializziamo come due array paralleli).
    private enum CodingKeys: String, CodingKey {
        case keepIds, oldToNewKeys, oldToNewValues, newVocabSize,
             oldVocabSize, coveragePct, previewDropped
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.keepIds = try c.decode([Int].self, forKey: .keepIds)
        let ks = try c.decode([Int].self, forKey: .oldToNewKeys)
        let vs = try c.decode([Int].self, forKey: .oldToNewValues)
        var map: [Int: Int] = [:]
        for (k, v) in zip(ks, vs) { map[k] = v }
        self.oldToNew = map
        self.newVocabSize = try c.decode(Int.self, forKey: .newVocabSize)
        self.oldVocabSize = try c.decode(Int.self, forKey: .oldVocabSize)
        self.coveragePct = try c.decode(Double.self, forKey: .coveragePct)
        self.previewDropped = (try? c.decode([DroppedTokenPreview].self,
                                              forKey: .previewDropped)) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(keepIds, forKey: .keepIds)
        let sorted = oldToNew.sorted { $0.key < $1.key }
        try c.encode(sorted.map { $0.key }, forKey: .oldToNewKeys)
        try c.encode(sorted.map { $0.value }, forKey: .oldToNewValues)
        try c.encode(newVocabSize, forKey: .newVocabSize)
        try c.encode(oldVocabSize, forKey: .oldVocabSize)
        try c.encode(coveragePct, forKey: .coveragePct)
        try c.encode(previewDropped, forKey: .previewDropped)
    }
}

/// Fase 1: scansiona il corpus, tokenizza, conta le frequenze e
/// decide chi tiene.
public enum VocabAnalyzer {

    // MARK: - Entry point

    /// Esegue l'analisi: tokenizza il corpus, costruisce la curva
    /// cumulativa di copertura, applica i force-include e i
    /// force-exclude, restituisce un `KeepDecision`.
    public static func analyze(
        tokenizerJSON: URL,
        corpus: URL,
        coverage: Double,
        onEvent: (VocabPruneEvent) -> Void
    ) throws -> KeepDecision {
        precondition(coverage > 0 && coverage <= 1.0,
                     "coverage must be in (0, 1]")

        // 1) Carica il tokenizer originale.
        let tokenizer = try Self.loadTokenizer(at: tokenizerJSON)
        let totalVocab = max(tokenizer.invVocab.keys.max() ?? 0,
                              tokenizer.invAddedTokens.keys.max() ?? 0) + 1

        // 2) Scan + count: conta le occorrenze di ogni token id.
        var counts = [Int: Int]()
        counts.reserveCapacity(totalVocab / 4)
        var lines = 0
        var tokens = 0

        try Self.walkCorpus(corpus) { line in
            let ids = tokenizer.encode(line)
            for id in ids {
                counts[id, default: 0] += 1
                tokens += 1
            }
            lines += 1
            if lines % 5_000 == 0 {
                onEvent(.scanned(lines: lines, tokens: tokens))
            }
        }
        onEvent(.scanned(lines: lines, tokens: tokens))

        // 3) Force-include: addedTokens, byte-level base 256, ASCII,
        //    Latin-Extended (utili anche se non visti nel corpus).
        var forcedKeep = Set<Int>()
        for id in tokenizer.addedTokens.values { forcedKeep.insert(id) }
        for id in tokenizer.invAddedTokens.keys { forcedKeep.insert(id) }
        for id in Self.byteLevelBaseIds(in: tokenizer) { forcedKeep.insert(id) }
        for id in Self.latinAndAsciiIds(in: tokenizer) { forcedKeep.insert(id) }

        // 4) Force-exclude: token che decodificano a script non latini
        //    (CJK, Hangul, Hiragana/Katakana, arabo, ebraico, devanagari,
        //    thai). Anche se vengono visti raramente nel corpus, è
        //    deliberato tagliarli per ridurre la matrice.
        let forceDrop = Self.scriptForeignIds(in: tokenizer)

        // 5) Top-K per copertura cumulativa.
        let sortedByFreq = counts.sorted { $0.value > $1.value }
        let totalCount = sortedByFreq.reduce(0) { $0 + $1.value }
        var cumulative = 0
        var topK = Set<Int>()
        for (id, c) in sortedByFreq {
            if forceDrop.contains(id) { continue }
            topK.insert(id)
            cumulative += c
            if Double(cumulative) >= Double(totalCount) * coverage {
                break
            }
        }

        // 6) Unione finale: forcedKeep ∪ topK, meno forceDrop.
        var finalKeep = forcedKeep.union(topK).subtracting(forceDrop)
        // Edge case: gli addedTokens NON vanno tolti anche se in
        // forceDrop — sono special tokens DeepSeek e non sono mai
        // "script foreign". `forcedKeep.union` li ha già aggiunti,
        // ma `.subtracting(forceDrop)` potrebbe rimuoverne uno per
        // errore se il content per qualche ragione matcha. Lo
        // ripristiniamo esplicitamente.
        for id in tokenizer.invAddedTokens.keys { finalKeep.insert(id) }

        let pct = totalCount > 0
            ? Double(cumulative) / Double(totalCount) : 0
        onEvent(.coverage(pct: pct,
                          kept: finalKeep.count,
                          total: totalVocab))

        // 7) Costruisci la mappa oldToNew. addedTokens preservano il
        //    loro ID (necessario perché il chat template / EncodingDSV4
        //    riferisce gli special token per stringa ma il
        //    SafeTensorsRewriter usa lo stesso ID-slot dell'embedding).
        let oldToNew = Self.buildRemap(
            keep: finalKeep,
            preserveIds: Set(tokenizer.invAddedTokens.keys),
            totalVocab: totalVocab)

        let newVocabSize = (oldToNew.values.max() ?? -1) + 1

        // Top-N dropped per UI / dry-run preview.
        let droppedIds = Set(counts.keys).subtracting(finalKeep)
        let preview: [DroppedTokenPreview] = droppedIds
            .compactMap { id -> DroppedTokenPreview? in
                guard let count = counts[id],
                      let token = tokenizer.invVocab[id] else { return nil }
                return DroppedTokenPreview(id: id, content: token, count: count)
            }
            .sorted { $0.count > $1.count }
            .prefix(50)
            .map { $0 }

        return KeepDecision(
            keepIds: finalKeep.sorted(),
            oldToNew: oldToNew,
            newVocabSize: newVocabSize,
            oldVocabSize: totalVocab,
            coveragePct: pct,
            previewDropped: preview)
    }

    /// Costruisce la mappa oldId→newId. Gli ID in `preserveIds`
    /// mantengono il loro valore originale. Tutti gli altri ID in
    /// `keep` vengono ricompattati a partire da 0, saltando gli slot
    /// occupati dai preserved (così non c'è collisione).
    static func buildRemap(keep: Set<Int>,
                            preserveIds: Set<Int>,
                            totalVocab: Int) -> [Int: Int] {
        var out: [Int: Int] = [:]
        // Step 1: gli ID preservati mantengono il valore.
        for id in preserveIds where keep.contains(id) {
            out[id] = id
        }
        // Step 2: gli ID non preservati vengono assegnati ai primi
        // slot liberi, partendo da 0 e saltando i preservedIds.
        let remaining = keep.subtracting(preserveIds).sorted()
        var nextSlot = 0
        for id in remaining {
            while preserveIds.contains(nextSlot) {
                nextSlot += 1
            }
            out[id] = nextSlot
            nextSlot += 1
        }
        return out
    }

    // MARK: - Loaders

    static func loadTokenizer(at url: URL) throws -> BPETokenizer {
        let data = try Data(contentsOf: url)
        return try BPETokenizer(jsonData: data)
    }

    /// Cammina il corpus emettendo le linee al consumer. Supporta:
    /// - file `.txt` (una linea per record),
    /// - file `.jsonl` (`{"text": "..."}` per record),
    /// - directory (walk ricorsivo per `.txt`/`.jsonl`).
    static func walkCorpus(_ root: URL,
                            line consumer: (String) throws -> Void) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root.path, isDirectory: &isDir) else {
            throw NSError(domain: "VocabAnalyzer", code: 10,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "Corpus path does not exist: \(root.path)"])
        }
        if isDir.boolValue {
            guard let it = fm.enumerator(at: root,
                                          includingPropertiesForKeys: nil,
                                          options: [.skipsHiddenFiles]) else {
                return
            }
            for case let url as URL in it {
                let ext = url.pathExtension.lowercased()
                if ext == "txt" || ext == "jsonl" {
                    try Self.readLines(url, consumer: consumer, isJsonl: ext == "jsonl")
                }
            }
        } else {
            let ext = root.pathExtension.lowercased()
            try Self.readLines(root, consumer: consumer, isJsonl: ext == "jsonl")
        }
    }

    private static func readLines(_ url: URL,
                                    consumer: (String) throws -> Void,
                                    isJsonl: Bool) throws {
        let data = try Data(contentsOf: url)
        guard let s = String(data: data, encoding: .utf8) else {
            return  // file binario / encoding non utf-8: skip silently
        }
        for raw in s.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(raw)
            if isJsonl {
                if let bytes = line.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                   let text = obj["text"] as? String {
                    try consumer(text)
                }
            } else {
                try consumer(line)
            }
        }
    }

    // MARK: - Force-include / force-exclude predicates

    /// Restituisce gli ID dei 256 byte-level base token (mapping
    /// GPT-2 byte→unicode). Necessari per il fallback UTF-8 su
    /// qualunque input.
    static func byteLevelBaseIds(in tok: BPETokenizer) -> Set<Int> {
        let bytes = Self.byteLevelBaseStrings()
        var ids = Set<Int>()
        for s in bytes {
            if let id = tok.vocab[s] {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Token che decodificano (via byteLevelUnicodeToByte) in stringhe
    /// con solo caratteri ASCII / Latin-1 / Latin Extended A/B. Forced
    /// keep — utili a coprire input italiano/europeo anche se rari.
    static func latinAndAsciiIds(in tok: BPETokenizer) -> Set<Int> {
        var ids = Set<Int>()
        for (id, token) in tok.invVocab {
            if tok.invAddedTokens[id] != nil { continue }   // skip special
            guard let decoded = decodeByteLevelToken(token) else { continue }
            if isAllLatinOrAscii(decoded) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Token che decodificano in stringhe contenenti SCRIPT NON
    /// LATINI (CJK, hangul, kana, arabic, hebrew, devanagari, thai).
    /// Forced drop — taglio aggressivo per ridurre la matrice.
    static func scriptForeignIds(in tok: BPETokenizer) -> Set<Int> {
        var ids = Set<Int>()
        for (id, token) in tok.invVocab {
            if tok.invAddedTokens[id] != nil { continue }   // mai droppare special
            guard let decoded = decodeByteLevelToken(token) else { continue }
            if containsForeignScript(decoded) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Decode di un token byte-level GPT-2 → stringa UTF-8 reale.
    /// Restituisce nil se il token contiene un character non in
    /// `unicodeToByte` (succede solo per special tokens, già filtrati
    /// dal caller).
    static func decodeByteLevelToken(_ token: String) -> String? {
        let u2b = unicodeToByteMap()
        var bytes: [UInt8] = []
        for ch in token {
            let s = String(ch)
            if let b = u2b[s] {
                bytes.append(b)
            } else {
                // Carattere fuori dalla byteToUnicode map (es. emoji
                // letterale in un added_token). Skip: il caller non
                // dovrebbe arrivarci, ma defenseive.
                return nil
            }
        }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// True se tutti i character sono in ASCII (U+0000..U+007F) o
    /// Latin Extended (U+0080..U+024F) o punteggiatura latina di base
    /// (U+2010..U+206F). Whitespace incluso.
    static func isAllLatinOrAscii(_ s: String) -> Bool {
        for scalar in s.unicodeScalars {
            let v = scalar.value
            if v <= 0x024F { continue }                       // ASCII + Latin
            if (0x2010...0x206F).contains(v) { continue }     // Latin punctuation
            if (0x2070...0x209F).contains(v) { continue }     // sub/superscripts
            return false
        }
        return true
    }

    /// True se la stringa contiene almeno un character in uno dei
    /// range "foreign" che vogliamo eliminare in modalità italiano-only.
    /// Conservativo: se contiene anche UN solo char foreign, il token
    /// va droppato (perché incapsula bytes di quel character).
    static func containsForeignScript(_ s: String) -> Bool {
        for scalar in s.unicodeScalars {
            let v = scalar.value
            // CJK
            if (0x3400...0x4DBF).contains(v) { return true }   // Ext A
            if (0x4E00...0x9FFF).contains(v) { return true }   // Unified Ideographs
            if (0x20000...0x2FA1F).contains(v) { return true } // Ext B-G + supplement
            // Kana
            if (0x3040...0x30FF).contains(v) { return true }   // Hiragana + Katakana
            if (0x31F0...0x31FF).contains(v) { return true }   // Katakana phonetic
            // Hangul
            if (0x1100...0x11FF).contains(v) { return true }   // Hangul Jamo
            if (0x3130...0x318F).contains(v) { return true }   // Compat Jamo
            if (0xAC00...0xD7AF).contains(v) { return true }   // Hangul Syllables
            // Arabic
            if (0x0600...0x06FF).contains(v) { return true }
            if (0x0750...0x077F).contains(v) { return true }
            if (0x08A0...0x08FF).contains(v) { return true }
            if (0xFB50...0xFDFF).contains(v) { return true }
            if (0xFE70...0xFEFF).contains(v) { return true }
            // Hebrew
            if (0x0590...0x05FF).contains(v) { return true }
            // Devanagari
            if (0x0900...0x097F).contains(v) { return true }
            // Thai
            if (0x0E00...0x0E7F).contains(v) { return true }
        }
        return false
    }

    // MARK: - GPT-2 byteToUnicode (replicato da BPETokenizer perché private)

    /// Restituisce le 256 stringhe di base che GPT-2 byte-level usa
    /// per rappresentare ogni byte 0..255. Sono i "single-character"
    /// token che servono come fallback UTF-8.
    static func byteLevelBaseStrings() -> [String] {
        let (b2u, _) = byteUnicodeMaps()
        return (UInt8(0)...UInt8(0xFF)).map { b2u[$0]! }
    }

    static func unicodeToByteMap() -> [String: UInt8] {
        return byteUnicodeMaps().1
    }

    /// Replica deterministica della `makeByteToUnicode` di
    /// `BPETokenizer.swift` (privata in quel file). Cache-able su
    /// chiamate ripetute (per ora ricalcolata — 256 entry, costo
    /// trascurabile).
    static func byteUnicodeMaps() -> ([UInt8: String], [String: UInt8]) {
        var bs: [UInt8] = []
        for b in UInt8(0x21)...UInt8(0x7E) { bs.append(b) }    // ASCII printable
        for b in UInt8(0xA1)...UInt8(0xAC) { bs.append(b) }    // Latin-1 printable
        for b in UInt8(0xAE)...UInt8(0xFF) { bs.append(b) }
        var cs = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...0xFF {
            let bb = UInt8(b)
            if !bs.contains(bb) {
                bs.append(bb)
                cs.append(256 + n)
                n += 1
            }
        }
        var b2u: [UInt8: String] = [:]
        var u2b: [String: UInt8] = [:]
        for (b, c) in zip(bs, cs) {
            let s = String(UnicodeScalar(c)!)
            b2u[b] = s
            u2b[s] = b
        }
        return (b2u, u2b)
    }
}
