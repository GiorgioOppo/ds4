import Foundation
import DS4Core
import DS4Metal

extension DiskKVStore {
    // MARK: File scanning / parsing

    func entryURLs() -> [URL] {
        let all = (try? FileManager.default.contentsOfDirectory(at: directory,
                                                                includingPropertiesForKeys: nil)) ?? []
        return all.filter { KVCFile.shaHexName($0.lastPathComponent) != nil }
    }

    func entryName(tokens: [Int], modelName: String) -> String {
        var bytes = Array(modelName.utf8)
        for t in tokens {
            let v = UInt32(truncatingIfNeeded: t)
            bytes.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                                      UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
        }
        return KVCFile.sha1Hex(bytes) + ".kv"
    }

    /// Cheap scan: header + model name + token list (no tensor body).
    func scanEntry(_ url: URL) -> (model: String, tokens: [Int])? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }
        func read(_ n: Int) -> [UInt8]? {
            guard n >= 0, let d = try? fh.read(upToCount: n), d.count == n else { return nil }
            return [UInt8](d)
        }
        guard let head = read(KVCFile.fixedHeader), KVCFile.parseHeader(head) != nil,
              let nameLenB = read(4) else { return nil }
        let nameLen = Int(KVCFile.leGet32(nameLenB, 0))
        guard nameLen < 4096, let nameB = read(nameLen),
              let countB = read(4) else { return nil }
        let count = Int(KVCFile.leGet32(countB, 0))
        guard count > 0, count < 1_000_000, let tokB = read(count * 4) else { return nil }
        var tokens = [Int](); tokens.reserveCapacity(count)
        for i in 0..<count { tokens.append(Int(KVCFile.leGet32(tokB, i * 4))) }
        return (String(decoding: nameB, as: UTF8.self), tokens)
    }

    func readHeader(_ url: URL) -> KVCFile.Header? {
        guard let fh = try? FileHandle(forReadingFrom: url),
              let d = try? fh.read(upToCount: KVCFile.fixedHeader) else { return nil }
        try? fh.close()
        return KVCFile.parseHeader([UInt8](d))
    }

    /// Bump hits + lastUsed in the 48-byte header, in place.
    func bumpHit(_ url: URL) {
        guard var h = readHeader(url) else { return }
        h.hits &+= 1
        h.lastUsed = UInt64(Date().timeIntervalSince1970)
        guard let fh = try? FileHandle(forWritingTo: url) else { return }
        try? fh.write(contentsOf: Data(KVCFile.fillHeader(h)))
        try? fh.close()
    }

}
