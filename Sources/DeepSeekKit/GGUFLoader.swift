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
    /// MTLBuffer backing every tensor view returned by `load(...)`.
    /// For `.mmap` / `.streaming` this wraps the mmap region; for
    /// `.preload` it's a freshly allocated buffer we read the file
    /// into. Tensor views use offsets within this buffer.
    private let sharedBuffer: MTLBuffer
    /// Strategy the loader picked. Affects `load(...)` behavior only
    /// at the margin (currently a logging concern); the per-layer
    /// streaming hints live on `LlamaStreamingModel`.
    public let strategy: LoadStrategy
    /// Raw mmap pointer + length, kept for `madvise` calls. Nil
    /// when `strategy == .preload` (the buffer is owned by Metal,
    /// not mmap).
    private let mmapPointer: UnsafeMutableRawPointer?
    private let mmapLength: Int

    /// Open a GGUF file.
    ///
    /// `strategy` mirrors the safetensors `LoadStrategy` semantics:
    ///   - `.mmap` (default): `mmap(PROT_READ, MAP_PRIVATE)`, OS
    ///     pages on demand. Lowest steady-state RSS.
    ///   - `.preload`: open + `read(2)` the whole file into a
    ///     freshly allocated MTLBuffer. No per-page faults during
    ///     inference, full size resident immediately.
    ///   - `.streaming`: mmap (same byte layout as `.mmap`) plus a
    ///     contract with `LlamaStreamingModel` to advise
    ///     `MADV_DONTNEED` per layer between forward passes. Same
    ///     init path as `.mmap`; the per-layer hints land at use
    ///     time, not here.
    ///
    /// `useMapShared` swaps `MAP_PRIVATE` for `MAP_SHARED` on the
    /// mmap variants. On Apple Silicon's unified-memory APFS this
    /// is slightly faster (direct page reuse, no copy-on-write
    /// shadow); on exotic filesystems / network mounts MAP_SHARED
    /// can fail at `mmap()` time and we fall back to `MAP_PRIVATE`
    /// silently.
    ///
    /// `warmup` triggers `posix_madvise(POSIX_MADV_WILLNEED)` on
    /// the whole mmap range right after creation. The kernel
    /// prefetches the file into the page cache asynchronously,
    /// removing the first-forward "cold mmap" latency at the cost
    /// of upfront I/O. No-op for `.preload`.
    public init(url: URL,
                strategy: LoadStrategy = .mmap,
                useMapShared: Bool = false,
                warmup: Bool = false) throws {
        self.url = url
        self.strategy = strategy

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

        let device = Device.shared.mtl
        let backingBuffer: MTLBuffer
        var mmapPtr: UnsafeMutableRawPointer? = nil
        var mmapLen: Int = 0

        switch strategy {
        case .preload:
            // Read the whole file into a fresh MTLBuffer. No mmap
            // involved — Metal owns the pages. RSS bumps by the
            // full file size at init time; trade-off for predictable
            // forward latency.
            guard let buf = device.makeBuffer(length: fileSize,
                                               options: .storageModeShared)
            else {
                throw GGUFError.io("preload MTLBuffer alloc failed (\(fileSize) bytes)")
            }
            var off = 0
            let dst = buf.contents()
            while off < fileSize {
                let n = read(fd, dst.advanced(by: off), fileSize - off)
                if n <= 0 {
                    throw GGUFError.io("preload read failed at offset \(off)")
                }
                off += n
            }
            backingBuffer = buf

        case .mmap, .streaming:
            let mapFlags: Int32 = useMapShared
                ? MAP_SHARED : MAP_PRIVATE
            var raw = mmap(nil, alignedSize, PROT_READ, mapFlags, fd, 0)
            if raw == MAP_FAILED && useMapShared {
                // Fall back to MAP_PRIVATE on filesystems that
                // refuse MAP_SHARED on read-only mmaps (rare, but
                // happens on some FUSE / network mounts). Matches
                // SafeTensorsFile's recovery path.
                raw = mmap(nil, alignedSize, PROT_READ, MAP_PRIVATE, fd, 0)
            }
            guard let raw, raw != MAP_FAILED else {
                throw GGUFError.io("mmap failed: \(url.path)")
            }
            mmapPtr = raw
            mmapLen = alignedSize
            let dealloc: (UnsafeMutableRawPointer, Int) -> Void = { p, n in
                munmap(p, n)
            }
            guard let buf = device.makeBuffer(
                bytesNoCopy: raw, length: alignedSize,
                options: .storageModeShared,
                deallocator: dealloc)
            else {
                munmap(raw, alignedSize)
                throw GGUFError.io("MTLBuffer makeBuffer(bytesNoCopy:) failed")
            }
            backingBuffer = buf
        }
        self.sharedBuffer = backingBuffer
        self.mmapPointer = mmapPtr
        self.mmapLength = mmapLen

        // Parse only the leading metadata region into the header.
        // Cap the parse window at a generous 256 MB — even the largest
        // GGUF metadata blobs (tens of thousands of tensors) come in
        // well under that.
        let parseWindow = min(fileSize, 256 * 1024 * 1024)
        let raw_buf = UnsafeRawBufferPointer(
            start: backingBuffer.contents(), count: parseWindow)
        let parsed = try GGUFHeader.parse(buffer: raw_buf)
        self.header = parsed

        var byName: [String: GGUFTensorInfo] = [:]
        byName.reserveCapacity(parsed.tensors.count)
        for t in parsed.tensors { byName[t.name] = t }
        self.infoByName = byName

        if warmup, let p = mmapPtr {
            // POSIX_MADV_WILLNEED asks the kernel to prefetch
            // asynchronously. We don't wait for completion — just
            // give the hint so the next read() / mmap touch is more
            // likely to be in the page cache.
            _ = posix_madvise(p, mmapLen, POSIX_MADV_WILLNEED)
        }
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

    // MARK: - Streaming hints

    /// Apply a `posix_madvise` to a sub-range of the mmap.
    /// Used by `LlamaStreamingModel` to evict / prefetch a layer's
    /// bytes between forward passes. No-op for `.preload` (no mmap
    /// to advise). The kernel may ignore the advice — it's a hint,
    /// not a guarantee.
    public func madviseRange(offset: Int, length: Int, advice: Int32) {
        guard let base = mmapPointer, length > 0 else { return }
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        // Round inward so a layer's madvise doesn't evict pages
        // that straddle the boundary with an adjacent layer's
        // tensors we still want resident. The edge pages (up to
        // ~one page on each side, typically 16 KB on macOS) stay
        // touched — negligible vs. the multi-MB layer interior.
        let end = offset + length
        let alignedStart = ((offset + pageSize - 1) / pageSize) * pageSize
        let alignedEnd   = (end / pageSize) * pageSize
        guard alignedEnd > alignedStart,
              alignedStart >= 0,
              alignedEnd <= mmapLength
        else { return }
        let ptr = base.advanced(by: alignedStart)
        _ = posix_madvise(ptr, alignedEnd - alignedStart, advice)
    }

    /// Total byte range a layer's tensors occupy in the mmap.
    /// Spans from the lowest `absoluteOffset` of any tensor whose
    /// name starts with `prefix` to the highest `absoluteOffset +
    /// byteCount`. Used by `LlamaStreamingModel` to emit one
    /// madvise per layer rather than one per tensor.
    public func byteRange(forNamePrefix prefix: String) -> (offset: Int, length: Int)? {
        var minOff = Int.max
        var maxEnd = 0
        for (name, info) in infoByName where name.hasPrefix(prefix) {
            minOff = min(minOff, info.absoluteOffset)
            maxEnd = max(maxEnd, info.absoluteOffset + info.byteCount)
        }
        guard maxEnd > 0 else { return nil }
        return (minOff, maxEnd - minOff)
    }
}
