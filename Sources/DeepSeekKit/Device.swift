import Foundation
import Metal

public final class Device {
    public static let shared = Device()

    public let mtl: MTLDevice
    public let queue: MTLCommandQueue

    // Loaded on first use. The converter target only needs `mtl` to wrap
    // mmapped safetensors as MTLBuffers and never touches the kernel
    // library; eagerly loading the metallib here would make the converter
    // fail on builds that don't ship one (e.g. plain `swift build`, which
    // doesn't always compile the .metal resources into default.metallib).
    private lazy var _library: MTLLibrary = Self.loadLibrary(device: mtl)
    public var library: MTLLibrary { _library }

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device. DeepSeekKit requires Apple Silicon.")
        }
        guard let q = dev.makeCommandQueue() else {
            fatalError("Could not create command queue.")
        }
        self.mtl = dev
        self.queue = q
    }

    private static func loadLibrary(device: MTLDevice) -> MTLLibrary {
        let bundle = Bundle.module
        do {
            return try device.makeDefaultLibrary(bundle: bundle)
        } catch {
            fatalError("Failed to load Metal library from bundle: \(error)")
        }
    }

    public func makePipeline(_ name: String) -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name) else {
            fatalError("Metal function not found: \(name)")
        }
        do {
            return try mtl.makeComputePipelineState(function: fn)
        } catch {
            fatalError("Failed to create pipeline for \(name): \(error)")
        }
    }
}
