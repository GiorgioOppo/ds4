import XCTest
import Foundation
@testable import DS4Core

final class LagunaConfigurationTests: XCTestCase {
    private enum MetadataValue {
        case string(String)
        case u32(UInt32)
        case u64(UInt64)
        case f32(Float)
        case f64(Double)
        case bool(Bool)
        case u32Array([UInt32])
    }

    private func headCountPattern() -> [UInt32] {
        (0..<48).map { $0 % 4 == 0 ? 48 : 72 }
    }

    private func validMetadata() -> [(String, MetadataValue)] {
        let p = "laguna."
        return [
            ("general.architecture", .string("laguna")),
            (p + "block_count", .u32(48)),
            (p + "context_length", .u64(262_144)),
            (p + "embedding_length", .u32(3_072)),
            (p + "vocab_size", .u32(100_352)),
            (p + "feed_forward_length", .u32(12_288)),
            (p + "attention.head_count", .u32Array(headCountPattern())),
            (p + "attention.head_count_kv", .u32(8)),
            (p + "attention.key_length", .u32(128)),
            (p + "attention.value_length", .u32(128)),
            (p + "rope.dimension_count", .u32(64)),
            (p + "rope.dimension_count_swa", .u32(128)),
            (p + "attention.sliding_window", .u32(512)),
            (p + "expert_count", .u32(256)),
            (p + "expert_used_count", .u32(10)),
            (p + "expert_feed_forward_length", .u32(1_024)),
            (p + "expert_shared_feed_forward_length", .u32(1_024)),
            (p + "expert_gating_func", .u32(2)),
            (p + "leading_dense_block_count", .u32(1)),
            (p + "rope.scaling.type", .string("yarn")),
            (p + "rope.scaling.original_context_length", .u64(8_192)),
            (p + "rope.freq_base", .f32(500_000)),
            (p + "rope.freq_base_swa", .f32(10_000)),
            (p + "rope.scaling.factor", .f32(32)),
            (p + "rope.scaling.yarn_attn_factor", .f32(1)),
            (p + "rope.scaling.yarn_beta_fast", .f32(32)),
            (p + "rope.scaling.yarn_beta_slow", .f32(1)),
            (p + "attention.layer_norm_rms_epsilon", .f64(1.0e-6)),
            (p + "expert_weights_scale", .f32(2.5)),
            (p + "expert_weights_norm", .bool(true)),
        ]
    }

    private func temporaryGGUF(_ metadata: [(String, MetadataValue)]) throws -> URL {
        var data = Data()
        func appendU32(_ raw: UInt32) {
            var value = raw.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendU64(_ raw: UInt64) {
            var value = raw.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendValue<T>(_ raw: T) {
            var value = raw
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendString(_ string: String) {
            let bytes = Data(string.utf8)
            appendU64(UInt64(bytes.count))
            data.append(bytes)
        }

        appendU32(GGUF.magic)
        appendU32(3)
        appendU64(0)
        appendU64(UInt64(metadata.count))
        for (key, value) in metadata {
            appendString(key)
            switch value {
            case .string(let string):
                appendU32(GGUFValueType.string.rawValue)
                appendString(string)
            case .u32(let number):
                appendU32(GGUFValueType.uint32.rawValue)
                appendU32(number)
            case .u64(let number):
                appendU32(GGUFValueType.uint64.rawValue)
                appendU64(number)
            case .f32(let number):
                appendU32(GGUFValueType.float32.rawValue)
                appendValue(number)
            case .f64(let number):
                appendU32(GGUFValueType.float64.rawValue)
                appendValue(number)
            case .bool(let value):
                appendU32(GGUFValueType.bool.rawValue)
                data.append(value ? 1 : 0)
            case .u32Array(let values):
                appendU32(GGUFValueType.array.rawValue)
                appendU32(GGUFValueType.uint32.rawValue)
                appendU64(UInt64(values.count))
                for value in values { appendU32(value) }
            }
        }
        while data.count < 32 || data.count % 32 != 0 { data.append(0) }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("laguna-config-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    private func load(_ metadata: [(String, MetadataValue)]) throws -> LagunaConfiguration {
        let url = try temporaryGGUF(metadata)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)
        return try LagunaConfiguration(model: model)
    }

    private func replacing(_ key: String, with value: MetadataValue)
        -> [(String, MetadataValue)] {
        var metadata = validMetadata()
        metadata[metadata.firstIndex(where: { $0.0 == key })!] = (key, value)
        return metadata
    }

    func testValidatesExactLagunaS21Geometry() throws {
        let configuration = try load(validMetadata())
        let shape = configuration.shape

        XCTAssertEqual(shape, .s2_1)
        XCTAssertEqual(shape.metadataNamespace, "laguna")
        XCTAssertEqual(shape.nLayer, 48)
        XCTAssertEqual(shape.nExpertUsed, 10)
        XCTAssertEqual(shape.keyValueProjectionWidth, 1_024)
        XCTAssertEqual(shape.contextLength, 262_144)
        XCTAssertEqual(shape.ropeOriginalContext, 8_192)
        XCTAssertEqual(configuration.headCounts, headCountPattern())
        XCTAssertTrue(configuration.expertWeightsNormalized)
        XCTAssertEqual(configuration.descriptor.architecture.id, .laguna)
        XCTAssertEqual(configuration.descriptor.architecture.family, .laguna)
        XCTAssertEqual(configuration.descriptor.architecture.backendAvailability,
                       .recognizedButNotImplemented)
        XCTAssertEqual(configuration.descriptor.layerCount, 48)
        XCTAssertTrue(configuration.descriptor.capabilities.contains(.mixtureOfExperts))
        XCTAssertFalse(configuration.descriptor.capabilities.contains(.compressedAttention))
    }

    func testHeadCountAlternationAndSlidingWindowRule() throws {
        let shape = try load(validMetadata()).shape
        for layer in 0..<48 {
            let expected: UInt32 = layer % 4 == 0 ? 48 : 72
            XCTAssertEqual(shape.layerHeadCount(layer), expected, "layer \(layer)")
            XCTAssertEqual(shape.isSlidingWindowLayer(layer), expected == 72,
                           "full attention runs exactly on the 48-head blocks")
            XCTAssertEqual(shape.queryProjectionWidth(layer), expected * 128)
        }
    }

    func testRejectsAnUnexpectedHeadCountEntry() throws {
        var heads = headCountPattern()
        heads[5] = 48
        XCTAssertThrowsError(
            try load(replacing("laguna.attention.head_count", with: .u32Array(heads)))
        ) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .unexpectedHeadCount(layer: 5, expected: 72, got: 48)
            )
        }
    }

    func testRejectsAWrongLengthHeadCountArray() throws {
        let heads = Array(headCountPattern().dropLast())
        XCTAssertThrowsError(
            try load(replacing("laguna.attention.head_count", with: .u32Array(heads)))
        ) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .invalidHeadCountArray("laguna.attention.head_count")
            )
        }
    }

    func testRequiresYarnRopeScaling() throws {
        XCTAssertThrowsError(
            try load(replacing("laguna.rope.scaling.type", with: .string("linear")))
        ) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .mismatch("laguna.rope.scaling.type", expected: "yarn", got: "linear")
            )
        }
    }

    func testRejectsACompatibleLookingButDifferentShape() throws {
        XCTAssertThrowsError(
            try load(replacing("laguna.expert_used_count", with: .u32(8)))
        ) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .mismatch("laguna.expert_used_count", expected: "10", got: "8")
            )
        }
    }

    func testRequiresExactLagunaArchitecture() throws {
        var metadata = validMetadata()
        metadata[0] = ("general.architecture", .string("glm-dsa"))

        XCTAssertThrowsError(try load(metadata)) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .wrongArchitecture(expected: "laguna", got: "glm-dsa")
            )
        }
    }

    func testRejectsFalseExpertWeightNormalization() throws {
        XCTAssertThrowsError(
            try load(replacing("laguna.expert_weights_norm", with: .bool(false)))
        ) { error in
            XCTAssertEqual(
                error as? LagunaConfigurationError,
                .mismatch("laguna.expert_weights_norm", expected: "true", got: "false")
            )
        }
    }
}
