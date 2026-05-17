import XCTest
import Metal
@testable import DeepSeekKit

/// Test della fondazione OOP introdotta in
/// `Sources/DeepSeekKit/Abstractions/` + `Sources/DeepSeekKit/Demo/`.
/// Copre `MetalKernel` (template method) e `DemoSoftmaxKernel`.
final class AbstractionsTests: XCTestCase {

    /// Subclass minimale che cattura le chiamate ai metodi
    /// astratti senza toccare il Device per validare la
    /// composizione del template method.
    private final class CountingKernel: MetalKernel {
        private let lock = NSLock()
        private var _threadgroupCalls = 0
        private var _gridCalls = 0
        private var _bindCalls = 0

        var threadgroupCalls: Int { lock.lock(); defer { lock.unlock() }; return _threadgroupCalls }
        var gridCalls: Int { lock.lock(); defer { lock.unlock() }; return _gridCalls }
        var bindCalls: Int { lock.lock(); defer { lock.unlock() }; return _bindCalls }

        override func threadgroupSize(for problem: KernelProblem) -> MTLSize {
            lock.lock(); _threadgroupCalls += 1; lock.unlock()
            return MTLSize(width: 32, height: 1, depth: 1)
        }
        override func gridSize(for problem: KernelProblem) -> MTLSize {
            lock.lock(); _gridCalls += 1; lock.unlock()
            return MTLSize(width: problem.dims[0], height: 1, depth: 1)
        }
        override func bind(problem: KernelProblem,
                            buffers: [MTLBuffer?],
                            offsets: [Int],
                            to encoder: MTLComputeCommandEncoder) {
            lock.lock(); _bindCalls += 1; lock.unlock()
            // No real binding — the test only checks invocation order.
        }
    }

    func testDemoSoftmaxKernelEncodesAndProducesUniformDistribution() throws {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")

        let rows = 2
        let cols = 4
        var data = [Float](repeating: 0, count: rows * cols) // softmax(0,…,0) = 1/cols
        let device = Device.shared
        let bytes = data.count * MemoryLayout<Float>.stride
        guard let buf = device.mtl.makeBuffer(
            bytes: &data, length: bytes, options: []) else {
            XCTFail("Could not allocate Metal buffer")
            return
        }

        guard let cmd = device.queue.makeCommandBuffer() else {
            XCTFail("Could not create command buffer")
            return
        }
        let kernel = DemoSoftmaxKernel()
        let problem = KernelProblem(dims: [rows, cols])
        kernel.dispatch(problem: problem,
                        buffers: [buf],
                        offsets: [0],
                        in: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let p = buf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let result = Array(UnsafeBufferPointer(start: p, count: rows * cols))
        let expected: Float = 1.0 / Float(cols)
        for v in result {
            XCTAssertEqual(v, expected, accuracy: 1e-5)
        }
    }

    func testMetalKernelTemplateMethodCallsAbstractOverrides() throws {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")
        // Usa softmax come pipeline reale (per evitare "function not
        // found"), ma con la subclass `CountingKernel` che conta gli
        // override.
        let kernel = CountingKernel(functionName: "softmax_f32")
        let device = Device.shared
        var data: [Float] = [0, 0, 0, 0]
        let buf = device.mtl.makeBuffer(
            bytes: &data,
            length: data.count * MemoryLayout<Float>.stride,
            options: [])!
        guard let cmd = device.queue.makeCommandBuffer() else {
            XCTFail("Could not create command buffer")
            return
        }
        let problem = KernelProblem(dims: [1, 4])
        kernel.dispatch(problem: problem,
                        buffers: [buf],
                        offsets: [0],
                        in: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        XCTAssertEqual(kernel.threadgroupCalls, 1)
        XCTAssertEqual(kernel.gridCalls, 1)
        XCTAssertEqual(kernel.bindCalls, 1)
    }
}
