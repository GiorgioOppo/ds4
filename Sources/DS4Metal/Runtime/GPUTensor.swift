import Foundation
import Metal

// Stage B: a resident GPU buffer that persists across many dispatches without
// CPU round-trips, the unit the graph passes between tensor-ops. Wraps an
// MTLBuffer (shared storage, hazard-tracked so chained dispatches on one encoder
// serialize correctly) plus its byte length and a logical float/element count.

public final class GPUTensor {
    public let buffer: MTLBuffer
    public let byteLength: Int
    public let count: Int   // logical element count (floats unless byte tensor)
    /// Offset (bytes) of the logical data within `buffer`. Non-zero only for
    /// no-copy mmap views, where the buffer starts at a page boundary <= the data.
    /// Encode binds that read this tensor must use `setBuffer(offset: byteOffset)`.
    public let byteOffset: Int

    public init(buffer: MTLBuffer, byteLength: Int, count: Int, byteOffset: Int = 0) {
        self.buffer = buffer
        self.byteLength = byteLength
        self.count = count
        self.byteOffset = byteOffset
    }

    /// No-copy GPU buffer over an mmap'd model region [ptr, ptr+byteLength). The
    /// Metal buffer must begin page-aligned, so it spans from the page boundary
    /// <= ptr; `byteOffset` carries the intra-page offset of the real data. No RAM
    /// copy — pages are served by the OS page cache (true SSD streaming, like the
    /// C g_model_views). Requires the GGUF mmap to be MAP_SHARED (metalMapping:true).
    public static func mappedNoCopy(_ rt: MetalRuntime, ptr: UnsafeRawPointer,
                                    byteLength: Int, elementCount: Int) throws -> GPUTensor {
        let page = Int(getpagesize())
        let addr = UInt(bitPattern: ptr)
        let alignedAddr = addr & ~UInt(page - 1)
        let off = Int(addr - alignedAddr)
        let mapLen = ((off + byteLength + page - 1) / page) * page
        guard let base = UnsafeMutableRawPointer(bitPattern: alignedAddr),
              let b = rt.device.makeBuffer(bytesNoCopy: base, length: mapLen,
                                           options: .storageModeShared, deallocator: nil) else {
            throw MetalError.bufferAlloc
        }
        return GPUTensor(buffer: b, byteLength: byteLength, count: elementCount, byteOffset: off)
    }

    /// Allocate `floatCount` zeroed F32 elements.
    public static func zeros(_ rt: MetalRuntime, floatCount: Int) throws -> GPUTensor {
        let len = max(1, floatCount) * 4
        guard let b = rt.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        memset(b.contents(), 0, len)
        return GPUTensor(buffer: b, byteLength: floatCount * 4, count: floatCount)
    }

    /// Allocate a zeroed raw byte buffer (e.g. F16 scratch, mask, tmp).
    public static func zerosBytes(_ rt: MetalRuntime, byteLength: Int) throws -> GPUTensor {
        let len = max(1, byteLength)
        guard let b = rt.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        memset(b.contents(), 0, len)
        return GPUTensor(buffer: b, byteLength: byteLength, count: byteLength)
    }

    /// Upload an F32 array.
    public static func floats(_ rt: MetalRuntime, _ a: [Float]) throws -> GPUTensor {
        let len = max(1, a.count) * 4
        guard let b = rt.device.makeBuffer(bytes: a.isEmpty ? [0] : a, length: len, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return GPUTensor(buffer: b, byteLength: a.count * 4, count: a.count)
    }

    /// Upload a raw byte blob (e.g. Q8_0/Q4_K quantized weights).
    public static func bytes(_ rt: MetalRuntime, _ a: [UInt8], elementCount: Int) throws -> GPUTensor {
        let len = max(1, a.count)
        guard let b = rt.device.makeBuffer(bytes: a.isEmpty ? [0] : a, length: len, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return GPUTensor(buffer: b, byteLength: a.count, count: elementCount)
    }

    /// Copy `byteLength` bytes from a raw pointer (e.g. an mmap'd GGUF tensor).
    public static func raw(_ rt: MetalRuntime, ptr: UnsafeRawPointer, byteLength: Int, elementCount: Int) throws -> GPUTensor {
        let len = max(1, byteLength)
        guard let b = rt.device.makeBuffer(bytes: ptr, length: len, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return GPUTensor(buffer: b, byteLength: byteLength, count: elementCount)
    }

    /// Read back the first `count` F32 elements (default: whole tensor).
    public func floatArray(_ n: Int? = nil) -> [Float] {
        let c = n ?? (byteLength / 4)
        let p = buffer.contents().bindMemory(to: Float.self, capacity: c)
        return Array(UnsafeBufferPointer(start: p, count: c))
    }

    public func zero() { memset(buffer.contents(), 0, byteLength) }
}
