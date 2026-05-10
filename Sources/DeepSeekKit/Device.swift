import Foundation
import Metal

public final class Device {
    public static let shared = Device()

    public let mtl: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device. DeepSeekKit requires Apple Silicon.")
        }
        guard let q = dev.makeCommandQueue() else {
            fatalError("Could not create command queue.")
        }
        self.mtl = dev
        self.queue = q
        self.library = Self.loadLibrary(device: dev)
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
