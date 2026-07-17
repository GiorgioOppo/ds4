import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
func run(suffix: String, think: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        run(suffixIds: tok.tokenizeRenderedChat(suffix).map { Int($0) },
            think: think, sampling: sampling, maxTokens: maxTokens)
    }

    func run(suffixIds: [Int], think: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int, checkpointAfter: Int = 0,
                     resumablePrefill: Bool = false) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            // .userInitiated: this is the task that actually runs the (synchronous,
            // GPU-blocking) decode loop on the actor's executor. The chat path already
            // drives it from a .userInitiated task, but other callers (HTTP server,
            // sub-agents) may not — pin it so the decode never runs at a throttled QoS.
            let task = Task(priority: .userInitiated) {
                do {
                    try self.generate(suffixIds: suffixIds, think: think, sampling: sampling,
                                      maxTokens: maxTokens, checkpointAfter: checkpointAfter,
                                      resumablePrefill: resumablePrefill, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// `checkpointAfter` > 0: posizione ASSOLUTA (in token) del confine del
    /// prefisso condiviso (system prompt + tool) — appena il prefill la supera,
    /// lo stato KV viene checkpointato su disco così un transcript divergente
    /// (chat nuova, stream annullato) lo restaura invece di riprefillarlo.
    /// `resumablePrefill`: il chiamante (server stateless) ricava il prompt dal
    /// transcript completo e verifica il prefisso via `starts(with:)` — quindi
    /// i token prefillati si possono COMMITTARE chunk per chunk e un annullamento
    /// a metà prefill lascia stato pulito e riprendibile. I percorsi chat NON
    /// lo usano: lì il chiamante assume suffissi tutto-o-niente.
    func generate(suffixIds initialSuffixIds: [Int], think: DS4ThinkMode, sampling: SamplingParams, maxTokens: Int,
                          checkpointAfter: Int = 0, resumablePrefill: Bool = false,
                          continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        var suffixIds = initialSuffixIds

        // Disk KV (ds4_kvstore model): on a COLD start, restore the longest stored
        // checkpoint that is an exact token prefix of this prompt and prefill only
        // the remainder. Covers both a fresh chat (system/agent prefix) and the
        // stateless HTTP server (each request re-sends the whole transcript).
        if committedIds.isEmpty, !kvDirty, let store = diskKV,
           let hit = store.findLongestPrefix(of: suffixIds, modelName: modelName) {
            continuation.yield(.progress("ripristino KV da disco (\(hit.tokens.count) token)…"))
            // Streaming restore: one layer at a time from file to KV buffers,
            // each batch freed after import (peak RAM = one layer, F_NOCACHE).
            let t0 = Date()
            if store.restore(hit, into: decoder) {
                committedIds = hit.tokens
                suffixIds.removeFirst(hit.tokens.count)
                lastDiskStoreCount = hit.tokens.count
                FileHandle.standardError.write(Data(String(
                    format: "DS4 diskkv: ripristino streaming %d token in %.2fs (F_NOCACHE, per-layer)\n",
                    hit.tokens.count, Date().timeIntervalSince(t0)).utf8))
            } else {
                committedIds = []          // fall back to a cold prefill
                kvDirty = true             // a partial import may have touched the KV
            }
        }

        let startPos = committedIds.count
        guard startPos + suffixIds.count < contextSize else {
            throw InferenceError.contextExceeded(prompt: startPos + suffixIds.count, context: contextSize)
        }
        guard !suffixIds.isEmpty else { continuation.yield(.progress("")); return }

        // Dirty-until-clean: any throw below (user cancel, error) leaves the GPU
        // KV/compressor state possibly out of sync with committedIds; the flag makes
        // the NEXT generation rebuild before continuing.
        // Clean slate for the prefill profile: everything timed from here to the
        // prefill→decode boundary (including a KV rebuild) is prefill work.
        decoder.resetProfile()
        let needsRebuild = kvDirty
        kvDirty = true
        if needsRebuild && !committedIds.isEmpty {
            // Recover from an interrupted generation: replay the exact committed ids
            // from position 0 (resets the recurrent compressor) — slow once, correct.
            continuation.yield(.progress("ripristino KV (\(committedIds.count) token)…"))
            _ = try decoder.prefill(tokens: committedIds, startPos: 0)
        }

        // Prefill ONLY the new suffix; positions 0..startPos-1 are reused from the KV.
        continuation.yield(.progress(startPos == 0 ? "prefill \(suffixIds.count) token…"
                                                   : "prefill +\(suffixIds.count) token (riuso KV)…"))
        // A BLOCCHI con progresso: un system prompt di migliaia di token
        // (Xcode & co.) costa minuti a ~0.3 s/token — senza avanzamento il
        // primo turno è indistinguibile da un hang. La granularità è la
        // STESSA del chunking interno del decoder (DS4_PREFILL_CHUNK): il
        // batching dell'unione esperti per blocco non cambia, zero costi.
        let pfChunk = max(64, ProcessInfo.processInfo.environment["DS4_PREFILL_CHUNK"].flatMap(Int.init) ?? 512)
        // Confine del checkpoint relativo a QUESTO suffisso (0 = nessuno): se un
        // restore da disco ha già superato il confine assoluto, non c'è nulla da
        // checkpointare.
        let checkpointRel = (checkpointAfter > startPos && checkpointAfter <= startPos + suffixIds.count)
            ? checkpointAfter - startPos : 0
        var lastLogits: [Float] = []
        var pfDone = 0
        let pfT0 = Date()
        while pfDone < suffixIds.count {
            if Task.isCancelled {
                // A inizio giro lo stato è coerente per costruzione (l'ultimo
                // chunk è stato forwardato E committato): in modalità riprendibile
                // il prossimo transcript identico ESTENDE da qui invece di
                // ripartire da zero. Fuori da quella modalità i token del chunk
                // non sono in committedIds → il flag dirty resta e si fa replay.
                if resumablePrefill { kvDirty = false }
                throw CancellationError()
            }
            var end = min(pfDone + pfChunk, suffixIds.count)
            if checkpointRel > pfDone && checkpointRel < end { end = checkpointRel }
            lastLogits = try decoder.prefill(tokens: Array(suffixIds[pfDone..<end]),
                                             startPos: startPos + pfDone)
            if resumablePrefill { committedIds.append(contentsOf: suffixIds[pfDone..<end]) }
            pfDone = end
            if checkpointRel > 0 && pfDone == checkpointRel, let store = diskKV {
                // Il prefisso condiviso è appena entrato nel KV: checkpoint SUBITO,
                // non a fine generazione — uno stream annullato a metà decode o
                // una conversazione nuova con lo stesso system prompt ripartono
                // dal restore (secondi) invece che dal prefill (minuti).
                continuation.yield(.progress("checkpoint del prefisso su disco (\(startPos + pfDone) token)…"))
                let t0 = Date()
                let box = DiskKVStore.SnapshotBox(decoder.exportKV(nKeys: startPos + pfDone))
                FileHandle.standardError.write(Data(String(
                    format: "DS4 diskkv: snapshot prefisso %d token esportato in %.2fs\n",
                    startPos + pfDone, Date().timeIntervalSince(t0)).utf8))
                let idsSnap = resumablePrefill ? committedIds
                                               : committedIds + Array(suffixIds[0..<pfDone])
                let name = modelName
                scheduleDiskKVWrite(store: store, tokens: idsSnap, modelName: name,
                                    box: box, reason: .cold)
                lastDiskStoreCount = startPos + pfDone
            }
            if pfDone < suffixIds.count {
                let dt = Date().timeIntervalSince(pfT0)
                let tps = dt > 0 ? Double(pfDone) / dt : 0
                continuation.yield(.progress(String(format: "prefill %d/%d token · %.1f tok/s…",
                                                    pfDone, suffixIds.count, tps)))
            }
        }
        if !resumablePrefill { committedIds.append(contentsOf: suffixIds) }
        // Profile the DECODE only (not the prefill), matching DS4Demo: reset the
        // per-phase counters at the prefill→decode boundary so decodeProfileReport()
        // reflects steady-state generation. The decode loop is opaque to the UI (it
        // runs inside the stream's task), so this is the only place to reset cleanly.
        // The prefill's own per-phase numbers are captured HERE, just before the
        // reset would discard them (surfaced in Log motore / demo DIAG).
        lastPrefillProfile = decoder.profile.report(title: "Profilo prefill")
        decoder.resetProfile()
        // The committed KV now ends with an open assistant turn; mark it immediately
        // so a mid-decode interruption still closes the turn on the next suffix.
        needsClose = true
        var pos = committedIds.count

        var rng = sampling.seed
        var inReasoning = think == .high       // suffix ends with <think> when enabled
        var inTool = false
        var pending: [UInt8] = []
        var visible = ""
        var toolBytes: [UInt8] = []
        var toolEmitted = 0
        let dsmlId = tok.dsmlId
        let lt = UInt8(ascii: "<")

        func flush(_ asReasoning: Bool) {
            guard !pending.isEmpty, let s = String(bytes: pending, encoding: .utf8) else { return }
            pending.removeAll(keepingCapacity: true)
            if asReasoning { continuation.yield(.reasoning(s)) }
            else { visible += s; continuation.yield(.text(s)) }
        }

        // Stream the not-yet-emitted suffix of the tool block as raw markup, so the
        // user watches the tool call being generated. Holds back partial UTF-8.
        func streamTool() {
            guard toolEmitted < toolBytes.count,
                  let s = String(bytes: toolBytes[toolEmitted...], encoding: .utf8) else { return }
            toolEmitted = toolBytes.count
            continuation.yield(.toolStream(s))
        }

        // Flush pending text but keep a trailing '<' buffered: it may begin the
        // tool-call opener "<｜DSML｜…" (the '<' and ｜DSML｜ are separate tokens) and
        // must not be streamed as a stray bubble before we know what follows.
        func flushHoldingOpener(_ asReasoning: Bool) {
            guard pending.last == lt else { flush(asReasoning); return }
            pending.removeLast()
            flush(asReasoning)        // emit everything before the '<'
            pending.append(lt)        // re-buffer the '<' for the next round
        }

        var produced = 0
        let genStart = Date()
        var sampleS = 0.0                          // CPU sampler (full-vocab sort at temp>0)
        var lastProgress = Date(timeIntervalSince1970: 0)
        var regimeStart: Date?                     // timestamp after token 4 (demo's REGIME cut)
        while produced < maxTokens && pos < contextSize {
            if Task.isCancelled {
                // A inizio giro il KV corrisponde ESATTAMENTE a committedIds
                // (l'ultimo token generato è stato forwardato E committato):
                // uno stop qui non sporca nulla. Senza questo, ogni stop utente
                // (o disconnessione del client) costava il replay INTEGRALE
                // della conversazione al turno successivo. Il turno assistant
                // resta aperto: needsClose lo chiude nel prossimo suffisso.
                kvDirty = false
                throw CancellationError()
            }
            // Penalize the recently produced tokens to break repeat-loop collapse.
            let lo = max(0, committedIds.count - sampling.repeatLastN)
            let tSample = Date()
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP,
                                      repetitionPenalty: sampling.repetitionPenalty,
                                      recent: committedIds[lo...], rng: &rng)
            sampleS += Date().timeIntervalSince(tSample)
            if Int32(next) == tok.eosId { break }   // eos closes the turn; not forwarded (next suffix re-adds it)
            if !inTool, Int32(next) == dsmlId {
                // A held opener '<' belongs to the tool block, not the visible text:
                // move it into toolBytes (so the parser sees "<｜DSML｜…") without ever
                // streaming it as a stray bubble.
                if pending.last == lt { pending.removeLast(); toolBytes.append(lt) }
                flush(inReasoning)                               // flush any remaining real text
                inTool = true
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
                streamTool()                                     // begin streaming the raw markup
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
                streamTool()
            } else if Int32(next) == tok.thinkStartId {
                // The model opened a reasoning block on its own (even with think
                // off): route it to the reasoning stream, don't show the tag.
                flush(inReasoning)
                inReasoning = true
            } else if Int32(next) == tok.thinkEndId {
                // Close reasoning (also when we weren't in it: suppress a stray
                // literal "</think>" instead of showing it as text).
                flush(inReasoning)
                inReasoning = false
            } else {
                pending.append(contentsOf: tok.tokenText(Int32(next)))
                flushHoldingOpener(inReasoning)
            }
            produced += 1
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            committedIds.append(next)           // the generated token is now in the KV
            pos += 1
            // THROTTLED progress (max ~4/s): every yield is a MainActor hop + a
            // SwiftUI invalidation in the GUI — per-token it costs main-thread
            // time that competes with the decode's own CPU work.
            let now = Date()
            if produced == 4 && regimeStart == nil { regimeStart = now }   // demo's REGIME cut
            if now.timeIntervalSince(lastProgress) >= 0.25 {
                lastProgress = now
                let elapsed = now.timeIntervalSince(genStart)
                let ms = elapsed / Double(produced) * 1000
                continuation.yield(.progress(String(format: "%d token · %.2f tok/s · %.0f ms/token",
                                                     produced,
                                                     elapsed > 0 ? Double(produced) / elapsed : 0, ms)))
            }
        }
        flush(inReasoning)
        // Attribution of the turn's wall clock: engine (per-phase profile),
        // sampler (CPU, full-vocab sort when temp>0 — the demo's greedy path
        // costs ~0), rest = streaming/tokenizer/actor/UI backpressure. This is
        // the number that answers "why is the GUI slower than the demo".
        if produced > 0 {
            let end = Date()
            let wall = end.timeIntervalSince(genStart)
            let engine = decoder.profile.totalS
            let per = 1000.0 / Double(produced)
            let other = max(0, wall - engine - sampleS)
            // Regime = dal token 5 (stesso taglio della demo): scarta il warm-up
            // (cache fredda, wiring) che sporca la media cumulativa sui turni corti.
            var regime = ""
            if let r = regimeStart, produced > 4 {
                let rWall = end.timeIntervalSince(r)
                let rTok = Double(produced - 4)
                regime = String(format: " — regime %.2f tok/s (%.0f ms/token)",
                                rTok / max(rWall, 0.001), rWall / rTok * 1000)
            }
            FileHandle.standardError.write(Data(String(
                format: "DS4 gui: %d token in %.1f s — %.0f ms/token = motore %.0f + sampler %.0f + stream/UI %.0f%@\n",
                produced, wall, wall * per, engine * per, sampleS * per, other * per, regime).utf8))
            // La STESSA attribuzione anche nello stream (→ log del pannello
            // Server): "il server è lento" si diagnostica solo sapendo se il
            // tempo è nel motore (gather/GPU: pressione di memoria, es. Xcode
            // aperto sui 16GB) o fuori (percorso server). Il regime scarta il
            // warm-up del primo token, che sporca la media cumulativa.
            continuation.yield(.progress(String(
                format: "resa: %.0f ms/token = motore %.0f + sampler %.0f + resto %.0f%@",
                wall * per, engine * per, sampleS * per, other * per, regime)))
        }

        // Extract any tool calls from the generated output.
        var calls: [ToolCall] = []
        if inTool {
            let block = String(bytes: toolBytes, encoding: .utf8) ?? ""
            do {
                calls = try ToolCallParser.parseStrict(visible + block, markup: markup).calls
            } catch {
                // The model opened a DSML block we could not parse: surface it
                // instead of dropping it silently. Strict parsing is deliberately
                // all-or-nothing: a truncated write/edit call is never executed.
                continuation.yield(.text("\n[chiamata tool incompleta o non valida: non eseguita]\n" + block))
            }
        } else {
            // Some quantized models spell DSML with ordinary BPE pieces instead
            // of emitting the dedicated token. A complete block remains valid,
            // but recovery/partial parsing is never an execution boundary.
            if let parsed = try? ToolCallParser.parseStrict(visible, markup: markup),
               !parsed.calls.isEmpty {
                calls = parsed.calls
            }
        }
        if !calls.isEmpty { continuation.yield(.toolCall(calls)) }
        kvDirty = false                         // clean completion: KV matches committedIds
        saveExpertUsage()                       // persist the usage imatrix (cheap)
        // Disk KV checkpoint (interval-gated: each entry is tens of MB).
        if let store = diskKV,
           committedIds.count - lastDiskStoreCount >= store.options.storeIntervalTokens {
            continuation.yield(.progress("salvataggio KV su disco…"))
            // First checkpoint of a conversation = "cold" (anchor: the shared
            // system/agent prefix, 2× protected in eviction); later = "continued"
            // (superseded under pressure by longer checkpoints of the same chat).
            let reason: KVCFile.Reason = lastDiskStoreCount == 0 ? .cold : .continued
            let t0 = Date()
            // Il Box è l'UNICO proprietario dello snapshot: il writer svuota
            // ogni layer appena scritto, così la RAM scende DURANTE la
            // scrittura invece di tenere l'intero checkpoint fino alla fine.
            let box = DiskKVStore.SnapshotBox(decoder.exportKV(nKeys: committedIds.count))
            let snapS = Date().timeIntervalSince(t0)
            FileHandle.standardError.write(Data(String(
                format: "DS4 diskkv: snapshot %d token esportato in %.2fs\n",
                committedIds.count, snapS).utf8))
            // La SCRITTURA va fuori dal percorso critico del turno: l'ultimo
            // token è già sullo schermo, ma prima il turno restava "aperto"
            // finché il checkpoint non era su disco (secondi percepiti come
            // tempo di generazione). DiskKVStore è Sendable: la scrittura
            // F_NOCACHE prosegue in background.
            let ids = committedIds, name = modelName
            scheduleDiskKVWrite(store: store, tokens: ids, modelName: name,
                                box: box, reason: reason)
            lastDiskStoreCount = committedIds.count   // gate even on dedup/failure
        }
        continuation.yield(.progress(""))
    }

    /// Launch a non-blocking checkpoint while retaining an actor-visible task
    /// handle. `quiesceForTeardown()` waits these handles before a model swap.
    private func scheduleDiskKVWrite(
        store: DiskKVStore,
        tokens: [Int],
        modelName: String,
        box: DiskKVStore.SnapshotBox,
        reason: KVCFile.Reason
    ) {
        let id = UUID()
        let writer = Task.detached(priority: .utility) { [weak self] in
            store.store(tokens: tokens, modelName: modelName, box: box, reason: reason)
            await self?.diskKVWriterFinished(id)
        }
        diskKVWriterTasks[id] = writer
    }

    private func diskKVWriterFinished(_ id: UUID) {
        diskKVWriterTasks.removeValue(forKey: id)
    }
}
