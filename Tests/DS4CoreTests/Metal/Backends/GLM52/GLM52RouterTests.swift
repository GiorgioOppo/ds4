import XCTest
@testable import DS4Metal

final class GLM52RouterTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func fixture() -> (logits: [Float], bias: [Float]) {
        var seed: UInt64 = 0x475F4C4D_35325F52
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }
        var logits = [Float](repeating: 0, count: 256)
        var bias = [Float](repeating: 0, count: 256)
        for index in 0..<256 {
            logits[index] = next() * 7
            bias[index] = next() * 0.25
        }
        // Exercise the stable sigmoid branches and a selection tie. The lower
        // expert id must win when the complete selection scores are equal.
        logits[0] = -100
        logits[1] = 100
        logits[17] = 0.75
        logits[23] = 0.75
        bias[17] = 0.5
        bias[23] = 0.5
        return (logits, bias)
    }

    func testReferenceUsesBiasOnlyForSelection() throws {
        var logits = [Float](repeating: -20, count: 256)
        var bias = [Float](repeating: 0, count: 256)
        logits[10] = 8
        logits[20] = 7
        bias[30] = 2

        let output = try GLM52RouterReference.route(logits: logits, bias: bias)
        XCTAssertTrue(output.selected.contains(30), "selection bias must affect top-8")

        let index = try XCTUnwrap(output.selected.firstIndex(of: 30))
        let denominator = output.selected.reduce(Float.zero) {
            $0 + output.probabilities[Int($1)]
        }
        let expected = output.probabilities[30] / denominator * 2.5
        XCTAssertEqual(output.weights[index], expected, accuracy: 1e-7)
    }

    func testMetalRouterMatchesScalarOracle() throws {
        let runtime = try makeRuntime()
        let input = fixture()
        let expected = try GLM52RouterReference.route(
            logits: input.logits,
            bias: input.bias
        )
        let actual = try runtime.glm52Route(logits: input.logits, bias: input.bias)

        XCTAssertEqual(actual.selected, expected.selected)
        for index in 0..<256 {
            XCTAssertEqual(
                actual.probabilities[index],
                expected.probabilities[index],
                accuracy: 2e-6,
                "probability expert \(index)"
            )
        }
        for index in 0..<8 {
            XCTAssertEqual(
                actual.weights[index],
                expected.weights[index],
                accuracy: 2e-6,
                "route weight \(index)"
            )
        }
    }

    func testExactSelectionTiesPreferLowerExpertID() throws {
        let runtime = try makeRuntime()
        let logits = [Float](repeating: 0, count: 256)
        let bias = [Float](repeating: 0, count: 256)

        let expected = Array(Int32(0)..<Int32(8))
        XCTAssertEqual(
            try GLM52RouterReference.route(logits: logits, bias: bias).selected,
            expected
        )
        XCTAssertEqual(try runtime.glm52Route(logits: logits, bias: bias).selected,
                       expected)
    }

    func testRejectsWrongGeometryBeforeDispatch() throws {
        let runtime = try makeRuntime()
        XCTAssertThrowsError(try runtime.glm52Route(
            logits: [Float](repeating: 0, count: 255),
            bias: [Float](repeating: 0, count: 256)
        ))
        XCTAssertThrowsError(try GLM52RouterReference.route(
            logits: [Float](repeating: 0, count: 256),
            bias: [Float](repeating: 0, count: 255)
        ))
    }
}
