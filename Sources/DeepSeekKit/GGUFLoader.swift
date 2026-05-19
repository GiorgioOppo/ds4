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
    /// types (Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K) allocate a fresh
    /// `MTLBuffer` of `outputDtype` and dispatch the matching
    /// dequant kernel. Other quantized types still throw
    /// `unsupportedType`.
    ///
    /// `outputDtype` is `.f32` by default for full precision; pass
    /// `.bf16` to halve the resident memory of the dequantized
    /// weight (the loaded model keeps these tensors live for its
    /// entire lifetime, so the saving compounds). Only `.f32` and
    /// `.bf16` are valid; other targets throw `unsupportedType`.
    /// Pass-through dtypes ignore the parameter — there's nothing
    /// to dequant.
    public func load(_ name: String,
                      outputDtype: DType = .f32) throws -> Tensor {
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
            return try dispatchDequant(info: info, base: "dequant_q8_0",
                                        outputDtype: outputDtype)
        case .q4_0:
            return try dispatchDequant(info: info, base: "dequant_q4_0",
                                        outputDtype: outputDtype)
        case .q4_K:
            return try dispatchDequant(info: info, base: "dequant_q4_k_m",
                                        outputDtype: outputDtype)
        case .q5_K:
            // Q5_K BF16 variant not shipped yet; fall back to F32
            // and ignore outputDtype with a soft warning. F32 still
            // unblocks the model; BF16 follow-up is a 1-line kernel
            // add when needed.
            return try dispatchDequant(info: info, base: "dequant_q5_k",
                                        outputDtype: .f32,
                                        bf16Available: false)
        case .q6_K:
            return try dispatchDequant(info: info, base: "dequant_q6_k",
                                        outputDtype: .f32,
                                        bf16Available: false)
        default:
            throw GGUFError.unsupportedType(info.type.rawValue)
        }
    }

    /// Allocate the output tensor (in `outputDtype`) and run one of
    /// the dequant kernels declared in
    /// `Sources/DeepSeekKit/Kernels/dequant_gguf.metal`. `base` is
    /// the kernel name prefix without the `_to_<dtype>` suffix; we
    /// append `_to_f32` or `_to_bf16` based on the requested
    /// `outputDtype`.
    ///
    /// `bf16Available` is a per-format gate: not every quantized
    /// type has its BF16 variant landed yet. When false (Q5_K /
    /// Q6_K today), we silently fall back to F32 — the caller still
    /// gets correct values, just at 2× the memory footprint.
    ///
    /// Synchronous `waitUntilCompleted` matches the design: `load`
    /// runs at model-load time, not in the hot loop.
    private func dispatchDequant(info: GGUFTensorInfo,
                                  base: String,
                                  outputDtype: DType,
                                  bf16Available: Bool = true) throws -> Tensor
    {
        let effectiveDtype: DType
        let suffix: String
        switch outputDtype {
        case .f32:
            effectiveDtype = .f32
            suffix = "_to_f32"
        case .bf16 where bf16Available:
            effectiveDtype = .bf16
            suffix = "_to_bf16"
        case .bf16:
            // BF16 not available for this format — fall back.
            effectiveDtype = .f32
            suffix = "_to_f32"
        default:
            throw GGUFError.unsupportedType(info.type.rawValue)
        }
        let nElem = info.shape.reduce(1, *)
        let out = Tensor.empty(shape: info.shape, dtype: effectiveDtype)
        let kernelName = base + suffix
        let pipeline = Device.shared.makePipeline(kernelName)
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
