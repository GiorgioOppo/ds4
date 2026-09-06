import XCTest
import Foundation
import CryptoKit
import CoreGraphics
import ImageIO
import Metal
import MetalPerformanceShaders
import DS4Core
@testable import DS4Metal

final class DeepSeekV4VisionTests: XCTestCase {
    func testLayoutMatchesUpstreamGoldenVectors() throws {
        let cases: [(Int, Int, Int, [Int], [Int])] = [
            (2, 3, 0, [1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4], [0, 3, 1, 4, 2, 5]),
            (3, 2, 5, [1, 1, 0, 2, 2, 2, 2, 3, 3, 2, 1, 2, 1, 3, 1, 4], [0, 2, 1, 3, 4, 5]),
            (1, 1, 3, [0, 2, 1, 3, 1, 4], [0]),
        ]
        for (height, width, position, types, permutation) in cases {
            let layout = try DeepSeekV4VisionLayout(gridHeight: height, gridWidth: width, startPosition: position)
            XCTAssertEqual(layout.types.map(\.rawValue), types)
            XCTAssertEqual(layout.imagePermutation, permutation)
            XCTAssertEqual((position + (try XCTUnwrap(layout.types.firstIndex(of: .start)))) % 4, 3)
        }
        XCTAssertThrowsError(try DeepSeekV4VisionLayout(gridHeight: 384, gridWidth: 384, startPosition: 0))
        XCTAssertThrowsError(try DeepSeekV4VisionLayout(gridHeight: 0, gridWidth: 3, startPosition: 0))
    }

    func testPreprocessingMatchesUpstreamGeometryAndPatchChannels() throws {
        let target = try DeepSeekV4ImagePreprocessor.targetSize(width: 17, height: 9)
        XCTAssertEqual(target.width, 532)
        XCTAssertEqual(target.height, 280)
        var pixels = [UInt8](repeating: 255, count: 17 * 9 * 4)
        for pixel in 0..<(17 * 9) {
            pixels[pixel * 4] = 255
            pixels[pixel * 4 + 1] = 0
            pixels[pixel * 4 + 2] = 127
        }
        let patches = try DeepSeekV4ImagePreprocessor.preprocess(rgba: pixels, width: 17, height: 9)
        XCTAssertEqual(patches.gridWidth, 38)
        XCTAssertEqual(patches.gridHeight, 20)
        XCTAssertEqual(patches.values.count, 760 * 588)
        XCTAssertTrue(patches.values.allSatisfy { $0.isFinite && (-1...1).contains($0) })
        // The second patch is wholly inside the content, past its 2px letterbox.
        XCTAssertEqual(Array(patches.values[588..<(588 + 196)]), Array(repeating: 1, count: 196))
        XCTAssertEqual(Array(patches.values[(588 + 196)..<(588 + 392)]), Array(repeating: -1, count: 196))
        XCTAssertEqual(patches.values[588 + 392], Float(127) / 127.5 - 1)
    }

    func testResizeStaysInsideImageTokenBudgetAcrossAspectRatios() throws {
        for (width, height) in [(1, 1), (16_384, 1), (1, 16_384), (1920, 1080), (3024, 4032), (4096, 4096)] {
            let target = try DeepSeekV4ImagePreprocessor.targetSize(width: width, height: height)
            XCTAssertEqual(target.width % 14, 0)
            XCTAssertEqual(target.height % 14, 0)
            for position in 0..<4 {
                let layout = try DeepSeekV4VisionLayout(gridHeight: (target.height / 14 + 2) / 3,
                    gridWidth: (target.width / 14 + 2) / 3, startPosition: position)
                XCTAssertLessThanOrEqual(layout.types.count, 384)
            }
        }
    }

    func testPreprocessingPixelsMatchUpstreamCReference() throws {
        // The official ds4_image.c fixture at upstream 9ab7053: digest covers
        // all 446,880 little-endian Float32 values from its C preprocessor.
        var pixels = [UInt8](repeating: 255, count: 17 * 9 * 4)
        for y in 0..<9 {
            for x in 0..<17 {
                let index = (y * 17 + x) * 4
                pixels[index] = UInt8(x * 13 + y * 3)
                pixels[index + 1] = UInt8(x * 5 + y * 17)
                pixels[index + 2] = UInt8(x * 7 + y * 11)
            }
        }
        let patches = try DeepSeekV4ImagePreprocessor.preprocess(rgba: pixels, width: 17, height: 9)
        let digest = patches.values.withUnsafeBytes { SHA256.hash(data: Data($0)) }
            .map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(digest, "8c549814b1ea046bed62eec2a1d1d1e98c19c2c2fbd0039dd0fd2537be0947df")
    }

    func testUnreadableImageFailsBeforeGPUWork() {
        XCTAssertThrowsError(try DeepSeekV4ImagePreprocessor.preprocess(data: Data("not an image".utf8)))
    }

    func testPNGDecodingPreservesRasterOrientationAndRGBChannels() throws {
        var pixels = [UInt8](repeating: 255, count: 17 * 9 * 4)
        for y in 0..<9 {
            for x in 0..<17 {
                let index = (y * 17 + x) * 4
                pixels[index] = UInt8(x * 13 + y * 3)
                pixels[index + 1] = UInt8(x * 5 + y * 17)
                pixels[index + 2] = UInt8(x * 7 + y * 11)
            }
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        let image = try XCTUnwrap(CGImage(width: 17, height: 9, bitsPerComponent: 8,
            bitsPerPixel: 32, bytesPerRow: 17 * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let encoded = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let actual = try DeepSeekV4ImagePreprocessor.preprocess(data: encoded as Data)
        let expected = try DeepSeekV4ImagePreprocessor.preprocess(rgba: pixels, width: 17, height: 9)
        XCTAssertEqual(actual.gridHeight, expected.gridHeight)
        XCTAssertEqual(actual.gridWidth, expected.gridWidth)
        XCTAssertEqual(actual.values, expected.values)
    }

    private func gpu() throws -> (MTLDevice, MTLCommandQueue, MTLLibrary) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) { options.mathMode = .safe }
        else { options.fastMathEnabled = false }
        return (device, queue, try device.makeLibrary(source: deepSeekV4VisionMetalSource, options: options))
    }

    private func buffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            try XCTUnwrap(device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
        }
    }

    private func run(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, name: String,
                     args: [UInt32], buffers: [MTLBuffer], width: Int, height: Int = 1,
                     groups: Bool = false, threads: Int = 32) throws {
        let cb = try XCTUnwrap(queue.makeCommandBuffer()), enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        let fn = try XCTUnwrap(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: fn)
        enc.setComputePipelineState(pipeline)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (index, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: index + 1) }
        let grid = MTLSize(width: width, height: height, depth: 1)
        let group = MTLSize(width: threads, height: 1, depth: 1)
        if groups { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group) }
        else { enc.dispatchThreads(grid, threadsPerThreadgroup: group) }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let error = cb.error { throw error }
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    func testBF16RoundingPreservesTiesAndInfinity() throws {
        let (device, queue, library) = try gpu()
        let input = [Float(bitPattern: 0x3f808000), Float(bitPattern: 0x3f818000), -.infinity, .infinity]
        let values = try buffer(input, device: device)
        try run(device: device, queue: queue, library: library, name: "kernel_deepseek4_vision_round_bf16",
            args: [4, 1], buffers: [values], width: 4)
        XCTAssertEqual(floats(values, count: 4).map(\.bitPattern), [0x3f800000, 0x3f820000, 0xff800000, 0x7f800000])
    }

    func testBF16MPSProjectionMatchesCPUAndWeightOrientation() throws {
        let (device, queue, library) = try gpu()
        let x: [Float] = [1, 2, 3, 4, 5, 6]
        let w: [Float] = [1, -1, 2, 0, 3, -2]
        let encoded = try buffer(w.map { UInt16($0.bitPattern >> 16) }, device: device)
        let weights = try buffer(Array(repeating: Float.zero, count: 6), device: device)
        try run(device: device, queue: queue, library: library, name: "kernel_vision_convert_bf16",
            args: [6], buffers: [encoded, weights], width: 6)
        let input = try buffer(x, device: device), output = try buffer(Array(repeating: Float.zero, count: 4), device: device)
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let multiply = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
            resultRows: 2, resultColumns: 2, interiorColumns: 3, alpha: 1, beta: 0)
        multiply.encode(commandBuffer: cb,
            leftMatrix: MPSMatrix(buffer: input, descriptor: .init(rows: 2, columns: 3, rowBytes: 12, dataType: .float32)),
            rightMatrix: MPSMatrix(buffer: weights, descriptor: .init(rows: 2, columns: 3, rowBytes: 12, dataType: .float32)),
            resultMatrix: MPSMatrix(buffer: output, descriptor: .init(rows: 2, columns: 2, rowBytes: 8, dataType: .float32)))
        cb.commit(); cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        XCTAssertEqual(floats(output, count: 4), [5, 0, 11, 3])
    }

    func testVisionAttentionSeesEveryPatch() throws {
        let (device, queue, library) = try gpu()
        let q = try buffer(Array(repeating: Float.zero, count: 2048), device: device)
        let k = try buffer(Array(repeating: Float.zero, count: 2048), device: device)
        let v = try buffer(Array(repeating: Float(2), count: 1024) + Array(repeating: Float(6), count: 1024), device: device)
        let out = try buffer(Array(repeating: Float.zero, count: 2048), device: device)
        try run(device: device, queue: queue, library: library, name: "kernel_glm53_vision_attention",
            args: [2, Float(0.125).bitPattern], buffers: [q, k, v, out], width: 2, height: 16, groups: true)
        XCTAssertEqual(floats(out, count: 2048), Array(repeating: 4, count: 2048))
    }

    func testAlignerUnfoldUsesChannelMajorOrderAndZeroPadding() throws {
        let (device, queue, library) = try gpu()
        let values: [Float] = (0..<4).flatMap { row in (0..<1024).map { Float(row * 10_000 + $0) } }
        let input = try buffer(values, device: device)
        let output = try buffer(Array(repeating: Float(-1), count: 9216), device: device)
        try run(device: device, queue: queue, library: library, name: "kernel_deepseek4_vision_aligner_reorder",
            args: [2, 2, 1], buffers: [input, output], width: 9216)
        let actual = floats(output, count: 9216)
        XCTAssertEqual(Array(actual[0..<9]), [0, 10_000, 0, 20_000, 30_000, 0, 0, 0, 0])
        XCTAssertEqual(Array(actual[9..<18]), [1, 10_001, 0, 20_001, 30_001, 0, 0, 0, 0])
    }
}
