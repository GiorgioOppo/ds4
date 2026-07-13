import Foundation
import DS4Core
import DS4Metal

extension DiskKVStore {
    /// Parse error of the streaming reader (truncated file, bogus lengths).
    enum StreamError: Error { case corrupt }

    /// Stream-parse one entry: header/meta first (via `onMeta`, which may throw
    /// to abort BEFORE any tensor body is read — the shape gate), then ONE layer
    /// at a time to `onLayer`. Each layer's slabs are freed as soon as the
    /// callback returns; nothing accumulates. The fd reads with F_NOCACHE so the
    /// checkpoint's pages never displace the hot page cache (dense weights /
    /// expert bundle) — same rule as the store path. Throws StreamError.corrupt
    /// on a malformed file; rethrows whatever the callbacks throw.
    func streamSnapshot(_ url: URL,
                        onMeta: (_ nKeys: Int, _ headDim: Int, _ layerCount: Int) throws -> Void,
                        onLayer: (_ index: Int, _ layer: KVLayerSnapshot) throws -> Void) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw StreamError.corrupt }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw StreamError.corrupt }
        var remaining = Int(st.st_size)

        func readAll(_ p: UnsafeMutableRawPointer, _ n: Int) throws {
            guard n <= remaining else { throw StreamError.corrupt }
            var off = 0
            while off < n {
                let r = read(fd, p + off, n - off)
                guard r > 0 else { throw StreamError.corrupt }
                off += r
            }
            remaining -= n
        }
        func readBytes(_ n: Int) throws -> [UInt8] {
            guard n > 0 else { throw StreamError.corrupt }   // no zero-length fields in this format
            var b = [UInt8](repeating: 0, count: n)
            try b.withUnsafeMutableBytes { try readAll($0.baseAddress!, n) }
            return b
        }
        func u32() throws -> Int {
            Int(KVCFile.leGet32(try readBytes(4), 0))
        }
        func skip(_ n: Int) throws {
            guard n >= 0, n <= remaining, lseek(fd, off_t(n), SEEK_CUR) >= 0 else {
                throw StreamError.corrupt
            }
            remaining -= n
        }
        /// One tensor slab, read straight into its final allocation (no Data
        /// intermediate). The `remaining` guard rejects bogus lengths before
        /// they can turn into a giant allocation.
        func floats(_ n: Int) throws -> [Float] {
            guard n >= 0, n * 4 <= remaining else { throw StreamError.corrupt }
            return try [Float](unsafeUninitializedCapacity: n) { buf, count in
                if n > 0 { try readAll(UnsafeMutableRawPointer(buf.baseAddress!), n * 4) }
                count = n
            }
        }
        func readComp() throws -> CompSnapshot? {
            let has = try readBytes(1)[0]
            if has != 1 { return nil }
            let cCount = try u32(), stateLen = try u32()
            let kv = try floats(stateLen), score = try floats(stateLen)
            let cache = try floats(try u32())
            return CompSnapshot(count: cCount, stateKv: kv, stateScore: score, cacheRows: cache)
        }

        guard KVCFile.parseHeader(try readBytes(KVCFile.fixedHeader)) != nil else {
            throw StreamError.corrupt
        }
        try skip(try u32())                    // model name (matched by the caller's scan)
        let count = try u32()                  // nKeys
        guard count > 0, count < 1_000_000 else { throw StreamError.corrupt }
        try skip(count * 4)                    // token ids (the Hit already carries them)
        guard let ph = DSV4PayloadHeader(try readBytes(DSV4PayloadHeader.u32Fields * 4)) else {
            throw StreamError.corrupt
        }
        try onMeta(count, Int(ph.rawHeadKVDim), Int(ph.layerCount))
        for i in 0..<Int(ph.layerCount) {
            let rawStart = try u32()
            let raw = try floats(try u32())
            let comp = try readComp()
            let idx = try readComp()           // NSA indexer compressor (ratio-4 layers)
            try onLayer(i, KVLayerSnapshot(rawStart: rawStart, raw: raw, comp: comp, idx: idx))
        }
    }

    /// Full in-RAM parse of one entry (tests / tools): the streaming reader with
    /// an accumulator. The engine restore path never materializes this.
    func loadSnapshot(_ url: URL) -> KVSnapshot? {
        var nKeys = 0, headDim = 0
        var layers: [KVLayerSnapshot] = []
        do {
            try streamSnapshot(url,
                onMeta: { n, h, _ in nKeys = n; headDim = h },
                onLayer: { _, layer in layers.append(layer) })
        } catch { return nil }
        return KVSnapshot(nKeys: nKeys, headDim: headDim, layers: layers)
    }
}
