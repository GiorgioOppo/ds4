import Foundation
import Metal

// Phase 8 of the C->Swift port: the Metal runtime foundation. This brings up a
// Swift Metal runtime that compiles the SAME vendored metal/*.metal kernels the
// C engine uses (ds4_metal.m concatenates a small prelude + the 19 kernel files
// and calls newLibraryWithSource at runtime), and dispatches them.
//
// The kernels themselves are unchanged GPU code, so this phase de-risks the
// Metal-in-Swift bet: if the library compiles and a kernel runs correctly from
// Swift, the much larger kernel-orchestration / graph phases (9, 10) can build
// on this foundation. Tensor-API (#ifdef DS4_METAL_HAS_TENSOR) blocks stay off,
// matching the engine's behavior on pre-M5/pre-A19 devices.

public enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case missingKernel(String)
    case kernelRead(String)
    case bufferAlloc

    public var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .noQueue: return "could not create Metal command queue"
        case .missingKernel(let n): return "kernel function not found: \(n)"
        case .kernelRead(let p): return "could not read kernel source: \(p)"
        case .bufferAlloc: return "Metal buffer allocation failed"
        }
    }
}

public final class MetalRuntime {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    /// Cache for pipelines specialized by function constants (see MetalDense).
    var mulMVPipelineCache: [String: MTLComputePipelineState] = [:]

    /// The 19 kernel files in the exact concatenation order used by
    /// ds4_gpu_full_source in ds4_metal.m (order can affect compilation).
    public static let kernelFiles = [
        "flash_attn", "dense", "moe", "dsv4_hc", "unary", "dsv4_kv", "dsv4_rope",
        "dsv4_misc", "argsort", "cpy", "concat", "get_rows", "sum_rows",
        "softmax", "repeat", "glu", "norm", "bin", "set_rows",
    ]

    /// Default runtime: kernel sources are embedded in the binary (KernelSources.swift),
    /// so no on-disk metal/ folder is needed — works in SwiftPM, the .xcodeproj, and
    /// a shipped .app.
    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let q = dev.makeCommandQueue() else { throw MetalError.noQueue }
        self.library = try dev.makeLibrary(source: MetalRuntime.buildSourceEmbedded(), options: MTLCompileOptions())
        self.device = dev
        self.queue = q
    }

    /// Explicit override: compile kernels from an on-disk metal/ folder (used by
    /// the validation tests). Production code uses the no-arg embedded init.
    public init(metalDir: String) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let q = dev.makeCommandQueue() else { throw MetalError.noQueue }
        let source = try MetalRuntime.buildSource(metalDir: metalDir)
        let opts = MTLCompileOptions()
        // Throws MTLLibraryError with the compiler diagnostics on failure.
        self.library = try dev.makeLibrary(source: source, options: opts)
        self.device = dev
        self.queue = q
    }

    /// Concatenate the prelude + embedded kernel sources (KernelSources.swift),
    /// in the canonical kernelFiles order.
    static func buildSourceEmbedded() throws -> String {
        var s = prelude
        for name in kernelFiles {
            guard let body = embeddedKernels[name] else { throw MetalError.kernelRead("embedded:" + name) }
            s += body
            s += "\n"
        }
        return s
    }

    public var functionNames: [String] { library.functionNames }
    public var deviceName: String { device.name }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw MetalError.missingKernel(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    /// Build the full kernel source: the ds4_metal.m prelude plus every vendored
    /// kernel file, concatenated in order.
    static func buildSource(metalDir: String) throws -> String {
        var s = prelude
        for name in kernelFiles {
            let path = (metalDir as NSString).appendingPathComponent(name + ".metal")
            guard let body = try? String(contentsOfFile: path, encoding: .utf8) else {
                throw MetalError.kernelRead(path)
            }
            s += body
            s += "\n"
        }
        return s
    }

    /// Functional bring-up check: drive kernel_touch_u8_stride (from the prelude),
    /// which copies src[i*stride] -> dst[i]. Returns true iff the GPU output
    /// matches the expected gather. Validates library + pipeline + dispatch +
    /// buffer readback end-to-end from Swift.
    public func runTouchSelfTest(count: Int = 512, stride: Int = 7) throws -> Bool {
        let totalSrc = count * stride
        var src = [UInt8](repeating: 0, count: totalSrc)
        for i in 0..<totalSrc { src[i] = UInt8(truncatingIfNeeded: i &* 131 &+ 17) }

        guard let srcBuf = device.makeBuffer(bytes: &src, length: totalSrc, options: .storageModeShared),
              let dstBuf = device.makeBuffer(length: count, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var strideU = UInt64(stride)
        var bytesU = UInt64(totalSrc)
        var dstOffset = UInt64(0)

        let pso = try pipeline("kernel_touch_u8_stride")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(srcBuf, offset: 0, index: 0)
        enc.setBuffer(dstBuf, offset: 0, index: 1)
        enc.setBytes(&strideU, length: 8, index: 2)
        enc.setBytes(&bytesU, length: 8, index: 3)
        enc.setBytes(&dstOffset, length: 8, index: 4)
        let tg = MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let out = dstBuf.contents().bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<count where out[i] != src[i * stride] { return false }
        return true
    }

    // Verbatim port of the ds4_gpu_source prelude in ds4_metal.m. Keep in sync.
    static let prelude = """
    #include <metal_stdlib>
    #ifdef DS4_METAL_HAS_TENSOR
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    #endif
    using namespace metal;
    #ifdef DS4_METAL_HAS_TENSOR
    using namespace mpp::tensor_ops;
    #endif

    #define MAX(x, y) ((x) > (y) ? (x) : (y))
    #define MIN(x, y) ((x) < (y) ? (x) : (y))
    #define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
    #define QK8_0 32
    #define N_SIMDWIDTH 32
    #define N_R0_Q8_0 2
    #define N_SG_Q8_0 4
    #define FC_MUL_MV 600
    #define FC_MUL_MM 700
    #define FC_BIN 1300
    #define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
    #define M_PI_F 3.14159265358979323846f

    kernel void kernel_touch_u8_stride(
            device const uchar    *src        [[buffer(0)]],
            device uchar          *dst        [[buffer(1)]],
            constant ulong        &stride     [[buffer(2)]],
            constant ulong        &bytes      [[buffer(3)]],
            constant ulong        &dst_offset [[buffer(4)]],
            uint gid [[thread_position_in_grid]]) {
        ulong off = (ulong)gid * stride;
        if (off >= bytes) return;
        dst[dst_offset + (ulong)gid] = src[off];
    }

    enum ds4_sort_order {
        DS4_SORT_ORDER_ASC,
        DS4_SORT_ORDER_DESC,
    };

    struct block_q8_0 {
        half d;
        int8_t qs[QK8_0];
    };

    """
}
