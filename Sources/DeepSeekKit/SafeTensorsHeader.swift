import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Parsed safetensors header. Only the JSON metadata — no mmap, no
/// MTLBuffer. Used by the streaming pool (`.streaming` strategy)
/// which builds its own buffer slots and pread-loads tensor data
/// on demand, instead of holding a shard's data in a long-lived
/// `MTLBuffer(bytesNoCopy:)` (whose mmap pages the kernel won't
/// evict on Apple Silicon, since the driver considers the buffer
/// "in use").
public struct SafeTensorsHeader {
    public let url: URL
    /// First byte AFTER the 8-byte length prefix + JSON header.
    /// pread + this offset reaches the data section directly.
    public let dataStart: Int
    /// Total bytes of the data section
    /// (`fileSize - dataStart`).
    public let dataByteCount: Int
    /// Same `Entry` struct used by `SafeTensorsFile`.
    public let entries: [String: SafeTensorsFile.Entry]

    /// Parse a single shard's header using FileHandle reads. No
    /// mmap allocation — releases the descriptor before returning.
    public static func parse(url: URL) throws -> SafeTensorsHeader {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "SafeTensorsHeader", code: 20, userInfo: [
                NSLocalizedDescriptionKey: "open failed: \(url.path)"
            ])
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw NSError(domain: "SafeTensorsHeader", code: 21)
        }
        let fileSize = Int(st.st_size)

        // 1. Read 8-byte little-endian header length prefix.
        var lenBytes = [UInt8](repeating: 0, count: 8)
        let n0 = lenBytes.withUnsafeMutableBufferPointer { read(fd, $0.baseAddress, 8) }
        guard n0 == 8 else {
            throw NSError(domain: "SafeTensorsHeader", code: 22)
        }
        let headerLen = lenBytes.withUnsafeBufferPointer {
            $0.baseAddress!.withMemoryRebound(to: UInt64.self, capacity: 1) { $0.pointee }
        }
        guard headerLen > 0, Int(headerLen) < fileSize else {
            throw NSError(domain: "SafeTensorsHeader", code: 23)
        }

        // 2. Read the JSON header.
        var headerBytes = [UInt8](repeating: 0, count: Int(headerLen))
        var read_total = 0
        while read_total < Int(headerLen) {
            let n = headerBytes.withUnsafeMutableBufferPointer {
                read(fd, $0.baseAddress!.advanced(by: read_total),
                     Int(headerLen) - read_total)
            }
            guard n > 0 else {
                throw NSError(domain: "SafeTensorsHeader", code: 24)
            }
            read_total += n
        }

        let headerData = Data(headerBytes)
        let rawJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        var parsed: [String: SafeTensorsFile.Entry] = [:]
        for (k, v) in rawJSON {
            if k == "__metadata__" { continue }
            let entryData = try JSONSerialization.data(withJSONObject: v)
            parsed[k] = try JSONDecoder().decode(SafeTensorsFile.Entry.self, from: entryData)
        }

        let dataStart = 8 + Int(headerLen)
        return SafeTensorsHeader(url: url,
                                  dataStart: dataStart,
                                  dataByteCount: fileSize - dataStart,
                                  entries: parsed)
    }
}
