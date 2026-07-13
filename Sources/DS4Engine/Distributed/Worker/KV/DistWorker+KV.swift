import Foundation

extension DistWorker {
    /// Serve one KV control frame: query stored prefixes, restore an exact
    /// checkpoint, or checkpoint the current shard state.
    func handleKV(_ type: Dist.MsgType, _ payload: Data, on conn: DistConnection) async throws {
        guard let (engine, _) = activeEngine() else {
            let msg = "worker not ready: no assignment loaded"
            switch type {
            case .kvQuery: try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
            default: try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: msg))
            }
            return
        }
        switch type {
        case .kvQuery:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
                return
            }
            try await conn.sendFrame(.kvLengths,
                                     DistKV.encodeLengths(engine.storedPrefixLengths(of: ids)))
        case .kvRestore:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvRestore"))
                return
            }
            // Through the gate: the restore writes the shard's KV buffers and
            // must not interleave with compute.
            let ok = await gate.run { engine.restoreKV(tokens: ids) }
            if ok { onLog("KV shard ripristinato da disco (\(ids.count) token)\n") }
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(
                ok: ok, message: ok ? "" : "no checkpoint for \(ids.count) tokens"))
        case .kvSave:
            guard let (ids, cold) = DistKV.decodeSave(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvSave"))
                return
            }
            // Export under the gate (state must hold still); the disk write
            // itself streams in the background (SnapshotBox).
            await gate.run { engine.saveKV(tokens: ids, cold: cold) }
            onLog("KV shard: checkpoint \(ids.count) token avviato\n")
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: true))
        default:
            break
        }
    }
}
