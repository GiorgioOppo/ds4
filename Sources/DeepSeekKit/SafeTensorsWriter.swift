import Foundation

/// Streaming writer for the safetensors format. Mirrors the read side in
/// `SafeTensorsFile`. File layout:
///
///   [u64 little-endian header length][JSON header][tensor data...]
///
/// Usage:
///   var w = SafeTensorsWriter()
///   w.add(name: "weight", dtype: "F32", shape: [4, 8], source: .data(bytes))
///   w.add(name: "weight2", dtype: "BF16", shape: [4, 8],
///         source: .file(url: input, offset: 1024))
///   try w.write(to: outputURL)
///
/// `.file` sources are streamed from the input file without loading the
/// whole tensor in RAM (mmap read + chunked write). `.data` sources are
/// already-in-memory buffers (small post-processed tensors like the fused
/// wo_a). `.compute` sources are user-provided closures that produce bytes
/// lazily — used for big computed buffers we don't want to keep in RAM.
public final class SafeTensorsWriter {

    public enum Source {
        case data(Data)
        case file(url: URL, offset: Int, byteCount: Int)
        case compute(byteCount: Int, () throws -> Data)
    }

    private struct Entry {
        let name: String
        let dtype: String
        let shape: [Int]
        let byteCount: Int
        let source: Source
    }

    private var entries: [Entry] = []

    public init() {}

    public func add(name: String, dtype: String, shape: [Int], source: Source) {
        let bc: Int
        switch source {
        case .data(let d): bc = d.count
        case .file(_, _, let n): bc = n
        case .compute(let n, _): bc = n
        }
        entries.append(Entry(name: name, dtype: dtype, shape: shape,
                             byteCount: bc, source: source))
    }

    public func write(to url: URL) throws {
        // 1. Compute byte layout in the header.
        var offset = 0
        var headerObj: [String: Any] = [:]
        for e in entries {
            let start = offset
            let end = offset + e.byteCount
            headerObj[e.name] = [
                "dtype": e.dtype,
                "shape": e.shape,
                "data_offsets": [start, end],
            ]
            offset = end
        }
        // Sort keys for deterministic output (helps reproducibility).
        let opts: JSONSerialization.WritingOptions = [.sortedKeys]
        let headerBytes = try JSONSerialization.data(withJSONObject: headerObj, options: opts)

        // 2. Open output file. Truncate to zero, allocate, then stream.
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let out = FileHandle(forWritingAtPath: url.path) else {
            throw NSError(domain: "SafeTensorsWriter", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "could not open \(url.path) for writing"
            ])
        }
        defer { try? out.close() }

        // 3. Header length (u64 little-endian).
        var hl = UInt64(headerBytes.count).littleEndian
        try out.write(contentsOf: Swift.withUnsafeBytes(of: &hl) { Data($0) })

        // 4. Header bytes.
        try out.write(contentsOf: headerBytes)

        // 5. Tensor data, in declaration order.
        for e in entries {
            try writeSource(e.source, byteCount: e.byteCount, to: out)
        }
    }

    private func writeSource(_ s: Source, byteCount: Int,
                              to out: FileHandle) throws {
        switch s {
        case .data(let d):
            try out.write(contentsOf: d)
        case .compute(_, let producer):
            let d = try producer()
            precondition(d.count == byteCount, "compute produced wrong byte count")
            try out.write(contentsOf: d)
        case .file(let url, let off, let n):
            // Stream-copy from input file in 64 MB chunks to keep memory
            // usage bounded regardless of tensor size.
            let chunk = 64 * 1024 * 1024
            guard let input = FileHandle(forReadingAtPath: url.path) else {
                throw NSError(domain: "SafeTensorsWriter", code: 2)
            }
            defer { try? input.close() }
            try input.seek(toOffset: UInt64(off))
            var remaining = n
            while remaining > 0 {
                let want = min(chunk, remaining)
                guard let data = try input.read(upToCount: want), data.count == want else {
                    throw NSError(domain: "SafeTensorsWriter", code: 3, userInfo: [
                        NSLocalizedDescriptionKey: "short read at \(url.path)"
                    ])
                }
                try out.write(contentsOf: data)
                remaining -= want
            }
        }
    }
}
