import Foundation
import DS4Core

extension DistCoordinator {
    // MARK: Cluster KV continuity

    /// Read a KV-control reply on a relay connection, skipping stale RESULT
    /// frames an abandoned turn may have left in the socket buffer.
    func readControl(_ conn: DistConnection) async throws -> (Dist.MsgType, Data) {
        while true {
            let (type, payload) = try await conn.readFrame()
            if type == .result { continue }                       // stale turn reply: drop
            if type == .progress { continue }                     // informational only
            if type == .error { throw DistError.remote(String(decoding: payload, as: UTF8.self)) }
            return (type, payload)
        }
    }

    /// Negotiate a disk restore for `ids`: intersect every worker's stored
    /// prefix lengths, pick the longest ALL shards can restore, and restore it
    /// everywhere. Returns the restored length, or nil (cold prefill — any
    /// partially restored shard is overwritten by the pos-0 prefill).
    func negotiateRestore(ids: [Int], onLog: @Sendable (String) -> Void) async -> Int? {
        do {
            var common: Set<Int>?
            for conn in conns {
                try await conn.sendFrame(.kvQuery, DistKV.encodeTokens(ids))
                let (type, payload) = try await readControl(conn)
                guard type == .kvLengths, let lengths = DistKV.decodeLengths(payload) else { return nil }
                common = common.map { $0.intersection(lengths) } ?? Set(lengths)
                if common?.isEmpty == true { return nil }
            }
            guard let best = common?.max(), best > 0 else { return nil }
            let prefix = Array(ids.prefix(best))
            for (i, conn) in conns.enumerated() {
                try await conn.sendFrame(.kvRestore, DistKV.encodeTokens(prefix))
                let (type, payload) = try await readControl(conn)
                guard type == .kvAck, let ack = DistKV.decodeAck(payload), ack.ok else {
                    onLog("restore KV fallito su \(entries[i].host):\(entries[i].port) — prefill da zero\n")
                    return nil
                }
            }
            onLog("KV ripristinato da disco su \(conns.count) worker (\(best) token)\n")
            return best
        } catch {
            onLog("negoziazione restore KV fallita: \(error) — prefill da zero\n")
            return nil
        }
    }

    /// Checkpoint the committed prefix on every worker (interval-gated by the
    /// caller). The workers export synchronously and write in the background.
    func broadcastSave(ids: [Int], cold: Bool, onLog: @Sendable (String) -> Void) async {
        do {
            for conn in conns {
                try await conn.sendFrame(.kvSave, DistKV.encodeSave(tokens: ids, cold: cold))
                let (type, payload) = try await readControl(conn)
                guard type == .kvAck, let ack = DistKV.decodeAck(payload), ack.ok else {
                    onLog("checkpoint KV rifiutato da un worker" +
                          ((DistKV.decodeAck(payload)?.message).map { ": \($0)" } ?? "") + "\n")
                    return
                }
            }
            lastDiskStoreCount = ids.count
            onLog("checkpoint KV cluster: \(ids.count) token su \(conns.count) worker\n")
        } catch {
            onLog("checkpoint KV fallito: \(error)\n")
        }
    }
}
