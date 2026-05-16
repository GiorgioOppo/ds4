import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// GGUF reader. Mirrors the `SafeTensorsFile` pattern: `mmap` the file
/// once, expose it as one shared `MTLBuffer`, then return `Tensor`
/// views into that buffer.
///
/// Scope of this MVP: pass-through dtypes only (F32 / F16 / BF16 / I32
/// / I8). Quantised tensors (Q4_0, Q4_K, Q8_0, …) are surfaced by
/// `info(name:)` so a caller can read the raw bytes + `GGUFType`, but
/// `load(name:)` will throw on them until the matching dequant
/// kernels land. Architecturally this leaves a clean seam for the
/// follow-up:
///
///   - add `q*_dequant.metal` kernels
///   - call them from `loadDequantized(name:targetDtype:)`
///   - the file mmap, name index, and tensor table parsing are
///     already in place
///
/// V4 / DeepSeek does NOT have a GGUF release yet (as of May 2026) —
/// the value of this loader is to enable testing the GGUF surface
/// with the many Llama / Mistral / Qwen quants available, with the
/// understanding that *running* those models also requires their
/// architecture to be implemented in the Transformer forward pass
/// (out of scope).
public final class GGUFFile {
    public let url: URL
    public let header: GGUFHeader
    private let infoByName: [String: GGUFTensorInfo]
    private let sharedBuffer: MTLBuffer

    /// mmap-backed reader.
    public init(url: URL) throws {
        self.url = url

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw GGUFError.io("open failed: \(url.path)")
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw GGUFError.io("fstat failed: \(url.path)")
        }
        let fileSize = Int(st.st_size)
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let alignedSize = ((fileSize + pageSize - 1) / pageSize) * pageSize

        guard let raw = mmap(nil, alignedSize, PROT_READ, MAP_PRIVATE, fd, 0),
              raw != MAP_FAILED else {
            throw GGUFError.io("mmap failed: \(url.path)")
        }
        let dealloc: (UnsafeMutableRawPointer, Int) -> Void = { p, n in
            munmap(p, n)
        }
        let device = Device.shared.mtl
        guard let buf = device.makeBuffer(
            bytesNoCopy: raw, length: alignedSize,
            options: .storageModeShared,
            deallocator: dealloc) else {
            munmap(raw, alignedSize)
            throw GGUFError.io("MTLBuffer makeBuffer(bytesNoCopy:) failed")
        }
        self.sharedBuffer = buf

        // Parse only the leading metadata region into the header.
        // Cap the parse window at a generous 256 MB — even the largest
        // GGUF metadata blobs (tens of thousands of tensors) come in
        // well under that.
        let parseWindow = min(fileSize, 256 * 1024 * 1024)
        let raw_buf = UnsafeRawBufferPointer(start: raw, count: parseWindow)
        let parsed = try GGUFHeader.parse(buffer: raw_buf)
        self.header = parsed

        var byName: [String: GGUFTensorInfo] = [:]
        byName.reserveCapacity(parsed.tensors.count)
        for t in parsed.tensors { byName[t.name] = t }
        self.infoByName = byName
    }

    /// Tensor names present in this file, in declaration order.
    public var tensorNames: [String] {
        return header.tensors.map(\.name)
    }

    public func info(name: String) -> GGUFTensorInfo? {
        return infoByName[name]
    }

    /// Pass-through load: returns a `Tensor` view into the mmap for
    /// F32/F16/BF16/I32/I8 tensors. Throws for quantised types until
    /// per-type dequant kernels land.
    public func load(_ name: String) throws -> Tensor {
        guard let info = infoByName[name] else {
            throw GGUFError.malformed("tensor not found: \(name)")
        }
        let dtype: DType
        switch info.type {
        case .f32:  dtype = .f32
        case .f16:  dtype = .f16
        case .bf16: dtype = .bf16
        case .i32:  dtype = .i32
        case .i8:   dtype = .i8
        default:
            throw GGUFError.unsupportedType(info.type.rawValue)
        }
        return Tensor(shape: info.shape, dtype: dtype,
                      buffer: sharedBuffer, offset: info.absoluteOffset)
    }
}
