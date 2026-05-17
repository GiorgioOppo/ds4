import XCTest
import Metal
@testable import DeepSeekKit

/// Test che `MTLComputePipelineState.tunedThreadgroup(forGrid:)`
/// rispetti gli invarianti dichiarati (vedi
/// `Sources/DeepSeekKit/PipelineTuning.swift`).
final class TunedThreadgroupTests: XCTestCase {

    private func pipeline() throws -> MTLComputePipelineState {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")
        // `silu_mul_f32` è un 1-D elementwise senza function constants
        // — pipeline disponibile su qualsiasi build del kit.
        return Device.shared.makePipeline("silu_mul_f32")
    }

    private func assertInvariants(_ tg: MTLSize,
                                   grid: MTLSize,
                                   pipeline: MTLComputePipelineState,
                                   file: StaticString = #file,
                                   line: UInt = #line) {
        let simd = pipeline.threadExecutionWidth
        let maxT = pipeline.maxTotalThreadsPerThreadgroup

        // tg.width invariant: multiplo di simd, oppure == grid.width
        // quando grid.width < simd.
        if grid.width < simd {
            XCTAssertEqual(tg.width, grid.width,
                           "Per grid.width < simd, tg.width deve essere == grid.width",
                           file: file, line: line)
        } else {
            XCTAssertEqual(tg.width % simd, 0,
                           "tg.width \(tg.width) non multiplo di simd \(simd)",
                           file: file, line: line)
        }

        // Cap su ogni asse.
        XCTAssertLessThanOrEqual(tg.width,  grid.width,  file: file, line: line)
        XCTAssertLessThanOrEqual(tg.height, grid.height, file: file, line: line)
        XCTAssertLessThanOrEqual(tg.depth,  grid.depth,  file: file, line: line)

        // Total <= maxT.
        let total = tg.width * tg.height * tg.depth
        XCTAssertLessThanOrEqual(total, maxT,
                                  "tg total \(total) > maxT \(maxT)",
                                  file: file, line: line)

        // Tutte le dimensioni positive.
        XCTAssertGreaterThanOrEqual(tg.width,  1, file: file, line: line)
        XCTAssertGreaterThanOrEqual(tg.height, 1, file: file, line: line)
        XCTAssertGreaterThanOrEqual(tg.depth,  1, file: file, line: line)
    }

    func test1DSmallGrid() throws {
        let p = try pipeline()
        let grid = MTLSize(width: 100, height: 1, depth: 1)
        let tg = p.tunedThreadgroup(forGrid: grid)
        assertInvariants(tg, grid: grid, pipeline: p)
    }

    func test1DLargeGrid() throws {
        let p = try pipeline()
        let grid = MTLSize(width: 1 << 20, height: 1, depth: 1)
        let tg = p.tunedThreadgroup(forGrid: grid)
        assertInvariants(tg, grid: grid, pipeline: p)
    }

    func test2DGrid() throws {
        let p = try pipeline()
        let grid = MTLSize(width: 17, height: 2048, depth: 1)
        let tg = p.tunedThreadgroup(forGrid: grid)
        assertInvariants(tg, grid: grid, pipeline: p)
    }

    func test3DGrid() throws {
        let p = try pipeline()
        let grid = MTLSize(width: 7, height: 5, depth: 128)
        let tg = p.tunedThreadgroup(forGrid: grid)
        assertInvariants(tg, grid: grid, pipeline: p)
    }

    func testGridSmallerThanSimd() throws {
        let p = try pipeline()
        let grid = MTLSize(width: 8, height: 1, depth: 1)
        let tg = p.tunedThreadgroup(forGrid: grid)
        // Caso limite: grid.width < simd (32 su Apple Silicon).
        // tg.width deve essere 8, NON un multiplo di simd.
        XCTAssertEqual(tg.width, 8)
        assertInvariants(tg, grid: grid, pipeline: p)
    }
}
