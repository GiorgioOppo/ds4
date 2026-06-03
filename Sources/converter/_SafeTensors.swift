import Foundation

// MARK: - SafeTensors format
//
// Minimal Swift port of the safetensors binary layout, sized to what
// `converter/main.swift` actually uses. The format is:
//
//   [ 8 bytes ]  header length, little-endian uint64
//   [ N bytes ]  JSON header:
//                   { "<tensor name>": { "dtype": "<DT>",
//                                         "shape": [int, …],
//                                         "data_offsets": [start, end] },
//                     … ,
//                     "__metadata__": { … }   (optional, skipped) }
//   [ ... ]      raw tensor bytes, offsets RELATIVE to the start of
//                this section
//
// `SafeTensorsFile` parses the header and exposes the entries; the
// caller computes absolute file offsets via `absoluteOffset(of:in:)`
// in `main.swift`. `SafeTensorsWriter` mirrors the inverse path: the
// caller stages every tensor (`add(...)`) then flushes one shard
// with `write(to:)`.

/// Read-only descriptor of a `.safetensors` shard. The full
/// `entries` table is parsed at init; tensor bytes are read lazily
/// by the caller through plain `FileHandle` / `mmap`.
struct SafeTensorsFile {

    /// One tensor record inside the shard's JSON header.
    struct Entry {
        /// Lowercase or uppercase short-form dtype string the
        /// safetensors spec uses ("F32", "F16", "BF16", "I8", "U8",
        /// "F8_E4M3", "F4_E2M1", "E8M0", …). The converter compares
        /// case-insensitively so we keep the wire value verbatim.
        let dtype: String

        /// Logical shape, exactly as serialised in the header.
        /// `data_offsets[1] - data_offsets[0]` is the *byte* size;
        /// element count derives from `shape.reduce(1, *)`.
        let shape: [Int]

        /// Two-element `[start, end]` byte range relative to the
        /// data section start (8 + headerLen). The converter
        /// indexes `[0]` to compute absolute file positions.
        let dataOffsets: [Int]
    }

    /// All entries in shard-physical order — sorted by
    /// `data_offsets[0]` so iteration matches the on-disk layout.
    /// The `__metadata__` key (if present) is dropped.
    let entries: [(name: String, entry: Entry)]

    /// Absolute byte position where tensor data begins: 8-byte
    /// header length + the JSON header bytes. Exposed for
    /// consumers that want to mmap the whole shard and slice it
    /// per tensor.
    let dataStart: Int

    /// URL the file was opened from. Kept so callers can re-open
    /// it without juggling a parallel URL alongside the file
    /// descriptor.
    let url: URL

    init(url: URL) throws {
        self.url = url
        guard let fh = FileHandle(forReadingAtPath: url.path) else {
            throw NSError(
                domain: "SafeTensorsFile", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Couldn't open \(url.path) for reading."])
        }
        defer { try? fh.close() }

        guard let lenData = try fh.read(upToCount: 8),
              lenData.count == 8
        else {
            throw NSError(
                domain: "SafeTensorsFile", code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(url.lastPathComponent) too small to be a "
                    + "safetensors file (missing 8-byte header length)."])
        }
        let headerLen = lenData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        // Sanity bound: a 256 MB JSON header would indicate a
        // corrupt file. Real shards stay under a few MB.
        guard headerLen < 256 * 1024 * 1024 else {
            throw NSError(
                domain: "SafeTensorsFile", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(url.lastPathComponent) declares an "
                    + "unreasonably large header (\(headerLen) bytes)."])
        }
        guard let jsonData = try fh.read(upToCount: Int(headerLen)),
              jsonData.count == Int(headerLen)
        else {
            throw NSError(
                domain: "SafeTensorsFile", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(url.lastPathComponent) header truncated."])
        }
        let raw = try JSONSerialization.jsonObject(with: jsonData)
        guard let obj = raw as? [String: Any] else {
            throw NSError(
                domain: "SafeTensorsFile", code: 5,
                userInfo: [NSLocalizedDescriptionKey:
                    "\(url.lastPathComponent) header isn't a JSON object."])
        }

        var parsed: [(name: String, entry: Entry)] = []
        parsed.reserveCapacity(obj.count)
        for (name, value) in obj {
            // Spec-reserved key holding free-form provenance —
            // never a tensor record.
            if name == "__metadata__" { continue }
            guard let dict = value as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shapeRaw = dict["shape"] as? [Any],
                  let offsetsRaw = dict["data_offsets"] as? [Any]
            else { continue }
            let shape: [Int] = shapeRaw.compactMap { v in
                if let i = v as? Int { return i }
                if let n = v as? NSNumber { return n.intValue }
                return nil
            }
            let offsets: [Int] = offsetsRaw.compactMap { v in
                if let i = v as? Int { return i }
                if let n = v as? NSNumber { return n.intValue }
                return nil
            }
            guard shape.count == shapeRaw.count,
                  offsets.count == 2
            else { continue }
            parsed.append((name, Entry(dtype: dtype,
                                         shape: shape,
                                         dataOffsets: offsets)))
        }
        // Stable on-disk order: tensor blob layout sorted by start
        // offset. Within the same start (shouldn't happen in real
        // shards) fall back to name to keep init deterministic.
        parsed.sort { a, b in
            if a.entry.dataOffsets[0] != b.entry.dataOffsets[0] {
                return a.entry.dataOffsets[0] < b.entry.dataOffsets[0]
            }
            return a.name < b.name
        }
        self.entries = parsed
        self.dataStart = 8 + Int(headerLen)
    }
}

/// Write-side counterpart of `SafeTensorsFile`. Accumulates a
/// shard's worth of tensors via `add(...)` (each carrying its bytes
/// either as an already-prepared `Data` blob, a closure that
/// produces one, or a slice of an existing file) and emits a single
/// `.safetensors` file on `write(to:)`.
///
/// Output layout matches what `SafeTensorsFile` parses: 8-byte LE
/// header length, JSON header, raw tensor bytes in add-order.
struct SafeTensorsWriter {

    /// Where a tensor's bytes come from. The two cases let the
    /// converter avoid materialising big buffers when a passthrough
    /// tensor can be copied straight from the input shard.
    enum Source {
        /// Copy `byteCount` bytes from `url` starting at `offset`.
        /// Used for native-dtype tensors that don't need any
        /// transformation.
        case file(url: URL, offset: Int, byteCount: Int)

        /// Run the closure on flush to produce exactly `byteCount`
        /// bytes. Used for INT8 quantised weights and group scales
        /// — the closure typically owns a lazily-released MTLBuffer
        /// or similar resource.
        case compute(byteCount: Int, () throws -> Data)
    }

    private struct StagedTensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let source: Source
    }

    private var staged: [StagedTensor] = []

    init() {}

    /// Stage one tensor for the current shard. Order is preserved
    /// in the output — first added → lowest data offset.
    mutating func add(name: String,
                       dtype: String,
                       shape: [Int],
                       source: Source) {
        staged.append(StagedTensor(name: name, dtype: dtype,
                                    shape: shape, source: source))
    }

    /// Flush everything staged so far to `url`. Writes to a temp
    /// file in the destination directory first and renames on
    /// completion so a crash mid-write doesn't leave a corrupt
    /// shard the next run would try to mmap.
    func write(to url: URL) throws {
        let fm = FileManager.default
        let parent = url.deletingLastPathComponent()
        try fm.createDirectory(at: parent,
                                withIntermediateDirectories: true)

        // First pass: build the JSON header with computed offsets.
        var headerEntries: [(String, [String: Any])] = []
        headerEntries.reserveCapacity(staged.count)
        var cursor = 0
        for t in staged {
            let byteCount = byteCount(of: t.source)
            let start = cursor
            let end = cursor + byteCount
            headerEntries.append((t.name, [
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [start, end],
            ]))
            cursor = end
        }
        var headerObject: [String: Any] = [:]
        for (name, entry) in headerEntries {
            headerObject[name] = entry
        }
        // `.sortedKeys` so the same input produces the same shard
        // bytes across runs — useful for diffing converter output
        // between commits.
        let headerData = try JSONSerialization.data(
            withJSONObject: headerObject, options: [.sortedKeys])

        let tmpURL = parent.appendingPathComponent(
            ".safetensors-tmp-\(UUID().uuidString)")
        fm.createFile(atPath: tmpURL.path, contents: nil)
        guard let out = FileHandle(forWritingAtPath: tmpURL.path) else {
            throw NSError(
                domain: "SafeTensorsWriter", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Couldn't open \(tmpURL.path) for writing."])
        }
        do {
            var lenLE = UInt64(headerData.count).littleEndian
            let lenData = Data(bytes: &lenLE,
                                count: MemoryLayout<UInt64>.size)
            try out.write(contentsOf: lenData)
            try out.write(contentsOf: headerData)
            for t in staged {
                try streamBytes(of: t.source, into: out)
            }
            try out.close()
        } catch {
            try? out.close()
            try? fm.removeItem(at: tmpURL)
            throw error
        }

        // Atomic-ish rename. If `url` already exists (resume / re-
        // write), replace it so the result is whichever shard
        // finished writing last.
        if fm.fileExists(atPath: url.path) {
            _ = try fm.replaceItemAt(url, withItemAt: tmpURL)
        } else {
            try fm.moveItem(at: tmpURL, to: url)
        }
    }

    private func byteCount(of source: Source) -> Int {
        switch source {
        case .file(_, _, let n):    return n
        case .compute(let n, _):    return n
        }
    }

    /// Append exactly `byteCount(of:source)` bytes to `out`.
    /// `.file` streams in 4 MB chunks so an mmap-style copy of a
    /// multi-GB tensor doesn't materialise the whole slice in RAM.
    /// `.compute` calls the closure once and writes the returned
    /// `Data` verbatim.
    private func streamBytes(of source: Source,
                              into out: FileHandle) throws {
        switch source {
        case .file(let url, let offset, let byteCount):
            guard let input = FileHandle(forReadingAtPath: url.path)
            else {
                throw NSError(
                    domain: "SafeTensorsWriter", code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                        "Couldn't open \(url.path) for streaming."])
            }
            defer { try? input.close() }
            try input.seek(toOffset: UInt64(offset))
            var remaining = byteCount
            let chunk = 4 * 1024 * 1024
            while remaining > 0 {
                let want = min(chunk, remaining)
                guard let data = try input.read(upToCount: want),
                      data.count == want
                else {
                    throw NSError(
                        domain: "SafeTensorsWriter", code: 3,
                        userInfo: [NSLocalizedDescriptionKey:
                            "Short read at offset "
                            + "\(offset + (byteCount - remaining)) of "
                            + "\(url.lastPathComponent)."])
                }
                try out.write(contentsOf: data)
                remaining -= want
            }
        case .compute(let byteCount, let producer):
            let data = try producer()
            guard data.count == byteCount else {
                throw NSError(
                    domain: "SafeTensorsWriter", code: 4,
                    userInfo: [NSLocalizedDescriptionKey:
                        "Compute closure returned \(data.count) bytes, "
                        + "expected \(byteCount)."])
            }
            try out.write(contentsOf: data)
        }
    }
}
