import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
public struct BenchPoint: Sendable {
        public let contextTokens: Int
        public let prefillTps: Double
        /// Media semplice (token generati / tempo totale) — sporca dei costi
        /// una-tantum (primo token freddo, stalli).
        public let genTps: Double
        /// 99° percentile della VELOCITÀ per-token (1/durata di ogni token,
        /// ordinati): la velocità di regime raggiunta, robusta agli outlier
        /// lenti. È la metrica riportata dal Bench.
        public let genTpsP99: Double
        public let kvBytes: UInt64
        /// Velocità per-token IN ORDINE DI GENERAZIONE (1/durata di ciascuno):
        /// distingue la partenza fredda (primi token lenti, poi regime — normale
        /// dopo un reload) dal degrado progressivo da pressione di memoria
        /// (coda più lenta della testa) — media e p99 da soli non li separano.
        public let genSpeeds: [Double]
        public init(contextTokens: Int, prefillTps: Double, genTps: Double, kvBytes: UInt64,
                    genTpsP99: Double = 0, genSpeeds: [Double] = []) {
            self.contextTokens = contextTokens; self.prefillTps = prefillTps
            self.genTps = genTps; self.kvBytes = kvBytes
            self.genTpsP99 = genTpsP99 > 0 ? genTpsP99 : genTps
            self.genSpeeds = genSpeeds
        }
    }

    /// Native benchmark (replaces the removed `ds4-bench` binary): prefill a
    /// synthetic prompt of `contextTokens` tokens and decode `genTokens` from it,
    /// returning prefill/generation throughput at that context frontier. Resets
    /// the conversation; `contextTokens` is clamped to fit the loaded context.
    public func benchmark(contextTokens: Int, genTokens: Int) throws -> BenchPoint {
        resetConversation(systemPrompt: nil)
        let ctx = max(8, min(contextTokens, contextSize - genTokens - 4))
        // Synthetic prompt: BOS + a tiled filler tokenization (output quality is
        // irrelevant for timing; the work — attention, MoE gather — is the same).
        var ids: [Int] = [Int(tok.bosId)]
        let filler = tok.tokenizeRenderedChat("The quick brown fox jumps over the lazy dog. ").map { Int($0) }
        let pad = filler.isEmpty ? [Int(tok.eosId)] : filler
        var i = 0
        while ids.count < ctx { ids.append(pad[i % pad.count]); i += 1 }
        ids = Array(ids.prefix(ctx))

        let t0 = Date()
        var lastLogits = try decoder.prefill(tokens: ids, startPos: 0)
        let prefillDt = Date().timeIntervalSince(t0)

        var pos = ids.count
        var rng: UInt64 = 0xD54
        var produced = 0
        var tokenSpeeds: [Double] = []          // 1/durata di OGNI token generato
        tokenSpeeds.reserveCapacity(genTokens)
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: 0.6, topK: 0, topP: 0.95, minP: 0.05, rng: &rng)
            let t0 = Date()
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            let dt = Date().timeIntervalSince(t0)
            if dt > 0 { tokenSpeeds.append(1.0 / dt) }
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        kvDirty = true   // synthetic KV state — force a rebuild on the next real turn
        let kv = UInt64(DSV4Shape.nLayer) * UInt64(ctx) * UInt64(dims.headDim) * 4
        // p99 della velocità per-token: ordina le velocità e prendi il valore
        // al 99° percentile — il regime raggiunto, insensibile al primo token
        // freddo e agli stalli che schiacciano la media.
        var p99 = 0.0
        if !tokenSpeeds.isEmpty {
            let sorted = tokenSpeeds.sorted()
            p99 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.99))]
        }
        return BenchPoint(contextTokens: ctx,
                          prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
                          genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
                          kvBytes: kv,
                          genTpsP99: p99,
                          genSpeeds: tokenSpeeds)
    }

    /// Riscalda il motore subito dopo il load. La slot-cache degli esperti crea
    /// i pool SOLO alla prima richiesta (allocazione Metal + fill dei top-usage:
    /// ~slot × ~7 MB × layer instradati ≈ GB letti da SSD) e i percorsi
    /// Metal/lookahead partono freddi: senza warmup è il PRIMO messaggio
    /// dell'utente a pagare tutto — primi chunk di prefill lenti e 4-7 s sul
    /// primo token. Un mini giro sintetico (12 token di prefill + 3 di decode)
    /// sposta quel costo al load. Idempotente; preserva il system prompt attivo
    /// e lascia lo stato pulito (il benchmark lo sporca di proposito).
    public func warmup() {
        guard !warmedUp else { return }
        warmedUp = true
        let saved = systemPrompt
        let t0 = Date()
        do {
            _ = try benchmark(contextTokens: 12, genTokens: 3)
            FileHandle.standardError.write(Data(String(
                format: "DS4 engine: warmup completato in %.1fs (pool esperti + kernel caldi)\n",
                Date().timeIntervalSince(t0)).utf8))
        } catch {
            // Best-effort: un warmup fallito non deve bloccare nulla — il primo
            // messaggio reale pagherà la partenza fredda come prima.
            FileHandle.standardError.write(Data("DS4 engine: warmup fallito: \(error)\n".utf8))
        }
        resetConversation(systemPrompt: saved)
    }
}

