import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// GGUF reader. Mirrors the `SafeTensorsFile` pattern: `mmap` the file
/// once, expose it as one shared `MTLBuffer`, then return `Tensor`
/// views into that buffer.
///
/// Supported dtypes:
///   - Pass-through (zero-copy view into the mmap): F32 / F16 / BF16
///     / I32 / I8.
///   - Dequant-on-load (TODO §10.2 / T2): Q8_0 / Q4_0 / Q4_K (Q4_K_M
///     uses the same `Q4_K` block format). Each call to `load(name:)`
///     for one of these types allocates a new F32 `MTLBuffer` and
///     dispatches the matching kernel from
///     `Sources/DeepSeekKit/Kernels/dequant_gguf.metal`. The result
///     is a dense F32 tensor with `info.shape`.
///
/// Other quantized types (Q5_K, Q6_K, the IQ family) still throw
/// `unsupportedType` until their kernels land.
///
/// V4 / DeepSeek does NOT have a GGUF release yet (as of May 2026) —
/// the value of this loader is to enable running the many Llama /
/// Mistral / Qwen quants available, paired with `LlamaModel` (TODO
/// §10.2 / T2).
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

    /// Load a tensor by name. Pass-through types (F32 / F16 / BF16 /
    /// I32 / I8) return a zero-copy view into the mmap; quantized
    /// types (Q8_0 / Q4_0 / Q4_K) allocate a fresh F32 `MTLBuffer`
    /// and dispatch the matching dequant kernel. Other quantized
    /// types still throw `unsupportedType`.
    public func load(_ name: String) throws -> Tensor {
        guard let info = infoByName[name] else {
            throw GGUFError.malformed("tensor not found: \(name)")
        }
        switch info.type {
        case .f32:
            return Tensor(shape: info.shape, dtype: .f32,
                           buffer: sharedBuffer, offset: info.absoluteOffset)
        case .f16:
            return Tensor(shape: info.shape, dtype: .f16,
                           buffer: sharedBuffer, offset: info.absoluteOffset)
        case .bf16:
            return Tensor(shape: info.shape, dtype: .bf16,
                           buffer: sharedBuffer, offset: info.absoluteOffset)
        case .i32:
            return Tensor(shape: info.shape, dtype: .i32,
                           buffer: sharedBuffer, offset: info.absoluteOffset)
        case .i8:
            return Tensor(shape: info.shape, dtype: .i8,
                           buffer: sharedBuffer, offset: info.absoluteOffset)
        case .q8_0:
            return dispatchDequant(info: info,
                                    kernel: "dequant_q8_0_to_f32")
        case .q4_0:
            return dispatchDequant(info: info,
                                    kernel: "dequant_q4_0_to_f32")
        case .q4_K:
            return dispatchDequant(info: info,
                                    kernel: "dequant_q4_k_m_to_f32")
        default:
            throw GGUFError.unsupportedType(info.type.rawValue)
        }
    }

    /// Allocate an F32 output tensor and run one of the dequant
    /// kernels declared in
    /// `Sources/DeepSeekKit/Kernels/dequant_gguf.metal`. The kernel
    /// is a 1-D dispatch with one thread per output element; we
    /// `waitUntilCompleted` before returning so the caller sees a
    /// fully materialized tensor and doesn't have to thread a
    /// `MTLCommandBuffer` through the load path.
    ///
    /// The synchronous wait is fine here because `load` runs at
    /// model-load time, not in the hot generation loop — saving a
    /// few ms per weight by deferring the wait would complicate the
    /// API for no user-visible win.
    private func dispatchDequant(info: GGUFTensorInfo,
                                  kernel name: String) -> Tensor
    {
        let nElem = info.shape.reduce(1, *)
        let out = Tensor.empty(shape: info.shape, dtype: .f32)
        let pipeline = Device.shared.makePipeline(name)
        guard let cmd = Device.shared.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder()
        else {
            fatalError("GGUF dequant: failed to allocate Metal command buffer")
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(sharedBuffer, offset: info.absoluteOffset, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        var nElemU32 = UInt32(nElem)
        enc.setBytes(&nElemU32, length: MemoryLayout<UInt32>.size, index: 2)

        let tgWidth = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: nElem, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return out
    }
}
