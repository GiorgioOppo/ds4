import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Memory-mapped safetensors reader.
///
/// File layout:
///   [u64 little-endian header length][JSON header][tensor data...]
///
/// The whole file is `mmap`-ed once and exposed as a single `MTLBuffer`
/// (via `makeBuffer(bytesNoCopy:length:options:deallocator:)`). On Apple
/// Silicon the GPU and CPU share the same address space, so the mmapped
/// pages are directly readable by Metal kernels without copying. Pages are
/// faulted in on first access and can be evicted by the OS under memory
/// pressure — this is the only way to handle a 900 GB V4-Pro checkpoint
/// on a Mac with 192-512 GB of unified memory.
///
/// `Tensor` returned from `load(_:)` carries the shared buffer + the
/// absolute byte offset into it; reads at the GPU do not allocate.
public final class SafeTensorsFile {
    public struct Entry: Decodable {
        public let dtype: String
        public let shape: [Int]
        public let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }

    public let url: URL
    public let entries: [String: Entry]
    private let dataStart: Int

    /// Shared MTLBuffer covering the entire mmapped file. All Tensors
    /// returned by `load(_:)` reference this buffer with their `offset`
    /// field set to the absolute byte position in the file.
    private let sharedBuffer: MTLBuffer

    public init(url: URL) throws {
        self.url = url

        // 1. Open the file for read.
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "SafeTensors", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "open failed: \(url.path)"
            ])
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw NSError(domain: "SafeTensors", code: 11)
        }
        let fileSize = Int(st.st_size)

        // 2. mmap the whole file. The pointer is page-aligned by mmap
        //    design; we round the mapping size up to a page boundary so
        //    `makeBuffer(bytesNoCopy:length:)` accepts it.
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let alignedSize = ((fileSize + pageSize - 1) / pageSize) * pageSize
        guard let raw = mmap(nil, alignedSize, PROT_READ, MAP_PRIVATE, fd, 0),
              raw != MAP_FAILED else {
            throw NSError(domain: "SafeTensors", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "mmap failed for \(url.path)"
            ])
        }

        // 3. Wrap as MTLBuffer. Deallocator unmaps when the buffer is
        //    released, which happens when the last Tensor referencing it
        //    is freed.
        let dealloc: (UnsafeMutableRawPointer, Int) -> Void = { p, n in
            munmap(p, n)
        }
        let device = Device.shared.mtl
        guard let buf = device.makeBuffer(
                bytesNoCopy: raw, length: alignedSize,
                options: .storageModeShared,
                deallocator: dealloc) else {
            munmap(raw, alignedSize)
            let gib = 1024.0 * 1024.0 * 1024.0
            let detail = """
            MTLBuffer creation failed for mmap of \(url.path)
              file size     : \(String(format: "%.2f", Double(fileSize) / gib)) GiB
              aligned size  : \(String(format: "%.2f", Double(alignedSize) / gib)) GiB (page=\(pageSize))
              maxBufferLen  : \(String(format: "%.2f", Double(device.maxBufferLength) / gib)) GiB
            Likely causes:
              - shard exceeds device.maxBufferLength (lower --shard-size-gb in converter)
              - filesystem is not APFS (exFAT/NTFS mmap may be rejected by Metal)
            """
            throw NSError(domain: "SafeTensors", code: 13, userInfo: [
                NSLocalizedDescriptionKey: detail
            ])
        }
        self.sharedBuffer = buf

        // 4. Parse the JSON header.
        let lenPtr = raw.bindMemory(to: UInt64.self, capacity: 1)
        let headerLen = Int(lenPtr.pointee)
        let headerData = Data(bytesNoCopy: raw.advanced(by: 8),
                              count: headerLen,
                              deallocator: .none)
        let rawJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        var parsed: [String: Entry] = [:]
        for (k, v) in rawJSON {
            if k == "__metadata__" { continue }
            let entryData = try JSONSerialization.data(withJSONObject: v)
            parsed[k] = try JSONDecoder().decode(Entry.self, from: entryData)
        }
        self.entries = parsed
        self.dataStart = 8 + headerLen
    }

    /// Returns a Tensor referencing the mmapped region. No bytes are
    /// copied; the GPU reads directly from the mmapped pages.
    public func load(_ name: String, on device: Device = .shared) throws -> Tensor {
        guard let e = entries[name] else {
            throw NSError(domain: "SafeTensors", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "missing tensor \(name)"])
        }
        let absOffset = dataStart + e.dataOffsets[0]
        let dt = Self.parseDType(e.dtype)
        return Tensor(shape: e.shape, dtype: dt,
                      buffer: sharedBuffer, offset: absOffset)
    }

    private static func parseDType(_ s: String) -> DType {
        switch s.uppercased() {
        case "F32": return .f32
        case "F16": return .f16
        case "BF16": return .bf16
        case "I32", "U32": return .i32
        case "I64", "U64": return .i64
        case "I8", "U8": return .i8
        case "F8_E4M3", "F8E4M3", "FLOAT8_E4M3FN": return .fp8E4M3
        case "F4_E2M1", "F4E2M1", "FLOAT4_E2M1FN_X2": return .fp4E2M1
        case "F8_E8M0", "F8E8M0", "FLOAT8_E8M0FNU": return .e8m0
        default:
            fatalError("unsupported safetensors dtype: \(s)")
        }
    }
}
