import XCTest
import DS4Core
@testable import DS4Metal

final class LagunaTensorSchemaTests: XCTestCase {
    private let f32 = LagunaTensorSchema.f32
    private let f16 = LagunaTensorSchema.f16
    private let q8 = LagunaTensorSchema.q8_0
    private let q2K = LagunaTensorSchema.q2_K
    private let q3K = LagunaTensorSchema.q3_K
    private let q4K = LagunaTensorSchema.q4_K
    private let q6K = LagunaTensorSchema.q6_K

    private enum Recipe {
        case signalQ8
        case legacy
    }

    /// Build the complete tensor directory of one published recipe.
    /// `routedType(layer)` lets tests model the mixed Q2_K/Q3_K file.
    private func records(_ recipe: Recipe,
                         shape: LagunaShape = .s2_1,
                         routedType: ((Int) -> UInt32)? = nil,
                         routedDownType: ((Int) -> UInt32)? = nil)
        -> [LagunaTensorRecord] {
        let signal = recipe == .signalQ8
        let attention = signal ? q8 : f16
        let dense = signal ? q8 : q4K
        let denseDown = signal ? q8 : q6K
        let embd = UInt64(shape.nEmbd)
        let vocab = UInt64(shape.nVocab)
        let ffDense = UInt64(shape.nFFDense)
        let ffExp = UInt64(shape.nFFExpert)
        let ffShared = UInt64(shape.nFFShared)
        let experts = UInt64(shape.nExpert)
        let kvWidth = UInt64(shape.keyValueProjectionWidth)

        var out: [LagunaTensorRecord] = [
            .init(name: "token_embd.weight", type: signal ? q8 : q4K,
                  dimensions: [embd, vocab]),
            .init(name: "output_norm.weight", type: f32, dimensions: [embd]),
            .init(name: "output.weight", type: signal ? q8 : q6K,
                  dimensions: [embd, vocab]),
        ]
        for layer in 0..<Int(shape.nLayer) {
            let p = "blk.\(layer)."
            let heads = UInt64(shape.layerHeadCount(layer))
            let qWidth = heads * UInt64(shape.nHeadDim)
            out += [
                .init(name: p + "attn_norm.weight", type: f32, dimensions: [embd]),
                .init(name: p + "attn_q.weight", type: attention,
                      dimensions: [embd, qWidth]),
                .init(name: p + "attn_k.weight", type: attention,
                      dimensions: [embd, kvWidth]),
                .init(name: p + "attn_v.weight", type: attention,
                      dimensions: [embd, kvWidth]),
                .init(name: p + "attn_gate.weight", type: attention,
                      dimensions: [embd, heads]),
                .init(name: p + "attn_q_norm.weight", type: f32,
                      dimensions: [UInt64(shape.nHeadDim)]),
                .init(name: p + "attn_k_norm.weight", type: f32,
                      dimensions: [UInt64(shape.nHeadDim)]),
                .init(name: p + "attn_output.weight", type: attention,
                      dimensions: [qWidth, embd]),
                .init(name: p + "ffn_norm.weight", type: f32, dimensions: [embd]),
            ]
            if layer < Int(shape.nLeadingDense) {
                out += [
                    .init(name: p + "ffn_gate.weight", type: dense,
                          dimensions: [embd, ffDense]),
                    .init(name: p + "ffn_up.weight", type: dense,
                          dimensions: [embd, ffDense]),
                    .init(name: p + "ffn_down.weight", type: denseDown,
                          dimensions: [ffDense, embd]),
                ]
                continue
            }
            let routed = routedType?(layer) ?? q4K
            let routedDown = routedDownType?(layer) ?? routed
            out += [
                .init(name: p + "ffn_gate_inp.weight", type: f32,
                      dimensions: [embd, experts]),
                .init(name: p + "exp_probs_b.bias", type: f32, dimensions: [experts]),
                .init(name: p + "ffn_gate_exps.weight", type: routed,
                      dimensions: [embd, ffExp, experts]),
                .init(name: p + "ffn_up_exps.weight", type: routed,
                      dimensions: [embd, ffExp, experts]),
                .init(name: p + "ffn_down_exps.weight", type: routedDown,
                      dimensions: [ffExp, embd, experts]),
                .init(name: p + "ffn_gate_shexp.weight", type: dense,
                      dimensions: [embd, ffShared]),
                .init(name: p + "ffn_up_shexp.weight", type: dense,
                      dimensions: [embd, ffShared]),
                .init(name: p + "ffn_down_shexp.weight",
                      type: signal ? q8 : q4K,
                      dimensions: [ffShared, embd]),
            ]
        }
        return out
    }

    func testOfficialQ8SignalRecipeValidates() throws {
        let directory = records(.signalQ8)
        XCTAssertEqual(try LagunaTensorSchema.quantizationLayout(records: directory),
                       .signalQ8)
        XCTAssertNoThrow(try LagunaTensorSchema.validate(records: directory))
    }

    func testLegacyF16RecipeWithQ6KDownsValidates() throws {
        let directory = records(.legacy, routedDownType: { _ in self.q6K })
        XCTAssertEqual(try LagunaTensorSchema.quantizationLayout(records: directory),
                       .legacy)
        XCTAssertNoThrow(try LagunaTensorSchema.validate(records: directory))
    }

    func testMixedQ2Q3RoutedFileValidates() throws {
        // laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf: official dense weights,
        // Q2_K routed experts in layers 1…20, Q3_K in layers 21…47.
        let directory = records(.signalQ8, routedType: { $0 <= 20 ? self.q2K : self.q3K })
        XCTAssertNoThrow(try LagunaTensorSchema.validate(records: directory))
    }

    func testUnsupportedRoutedTypeIsRejectedPerLayer() {
        let directory = records(.signalQ8, routedType: { $0 == 7 ? self.q6K : self.q4K })
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: directory)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError,
                           .unsupportedRoutedType(layer: 7, type: q6K))
        }
    }

    func testRoutedDownMustBeCoherentWithGateUp() {
        // Q6_K downs are a legacy-recipe allowance; the Q8_0 signal recipe
        // requires the down projection to match the gate/up type.
        let directory = records(.signalQ8, routedDownType: { $0 == 3 ? self.q6K : self.q4K })
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: directory)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError,
                           .routedDownMismatch(layer: 3, downType: q6K, routedType: q4K))
        }
    }

    func testUnsupportedLayoutMarkerIsRejected() {
        var directory = records(.signalQ8)
        directory[0] = .init(name: "token_embd.weight", type: f16,
                             dimensions: directory[0].dimensions)
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: directory)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError,
                           .unsupportedLayoutMarker(type: f16))
        }
    }

    func testLayerOnlyViewIdentifiesTheLayoutFromAttentionQ() throws {
        let directory = records(.legacy)
            .filter { $0.name.hasPrefix("blk.") }
        XCTAssertEqual(try LagunaTensorSchema.quantizationLayout(records: directory),
                       .legacy)
        XCTAssertNoThrow(try LagunaTensorSchema.validate(
            records: directory,
            requireTokenEmbedding: false,
            requireOutput: false
        ))
    }

    func testSwappedHeadCountWidthIsRejected() {
        // Layer 0 is a full-attention block with 48 heads; a 72-head width
        // there means the per-layer alternation was lost.
        var directory = records(.signalQ8)
        let index = directory.firstIndex { $0.name == "blk.0.attn_q.weight" }!
        directory[index] = .init(name: "blk.0.attn_q.weight", type: q8,
                                 dimensions: [3_072, 72 * 128])
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: directory)) { error in
            guard case LagunaTensorSchemaError.layout(let name, _, let got, _, let expected)?
                    = error as? LagunaTensorSchemaError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(name, "blk.0.attn_q.weight")
            XCTAssertEqual(got, [3_072, 9_216])
            XCTAssertEqual(expected, [3_072, 6_144])
        }
    }

    func testMissingTensorAndPartialOutputHeadAreDistinctErrors() {
        var missing = records(.signalQ8)
        missing.removeAll { $0.name == "blk.11.exp_probs_b.bias" }
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: missing)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError,
                           .missing("blk.11.exp_probs_b.bias"))
        }

        var partial = records(.signalQ8)
        partial.removeAll { $0.name == "output.weight" }
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: partial)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError, .partialOutputHead)
        }
    }

    func testDuplicateTensorIsRejected() {
        var directory = records(.signalQ8)
        directory.append(directory[0])
        XCTAssertThrowsError(try LagunaTensorSchema.validate(records: directory)) { error in
            XCTAssertEqual(error as? LagunaTensorSchemaError,
                           .duplicateTensor("token_embd.weight"))
        }
    }
}
