import XCTest
import Foundation
@testable import DS4Core

final class GLM52ConfigurationTests: XCTestCase {
    private enum MetadataValue {
        case string(String)
        case u32(UInt32)
        case u64(UInt64)
        case f32(Float)
        case f64(Double)
        case bool(Bool)
    }

    private func validMetadata() -> [(String, MetadataValue)] {
        let p = "glm-dsa."
        return [
            ("general.architecture", .string("glm-dsa")),
            (p + "block_count", .u32(79)),
            (p + "context_length", .u64(1_048_576)),
            (p + "embedding_length", .u32(6_144)),
            (p + "vocab_size", .u32(154_880)),
            (p + "feed_forward_length", .u32(12_288)),
            (p + "attention.head_count", .u32(64)),
            (p + "attention.head_count_kv", .u32(1)),
            (p + "attention.key_length", .u32(576)),
            (p + "attention.value_length", .u32(512)),
            (p + "rope.dimension_count", .u32(64)),
            (p + "attention.q_lora_rank", .u32(2_048)),
            (p + "attention.kv_lora_rank", .u32(512)),
            (p + "attention.key_length_mla", .u32(256)),
            (p + "attention.value_length_mla", .u32(256)),
            (p + "expert_count", .u32(256)),
            (p + "expert_used_count", .u32(8)),
            (p + "expert_feed_forward_length", .u32(2_048)),
            (p + "expert_shared_count", .u32(1)),
            (p + "expert_group_count", .u32(1)),
            (p + "expert_group_used_count", .u32(1)),
            (p + "expert_gating_func", .u32(2)),
            (p + "leading_dense_block_count", .u32(3)),
            (p + "nextn_predict_layers", .u32(1)),
            (p + "attention.indexer.head_count", .u32(32)),
            (p + "attention.indexer.key_length", .u32(128)),
            (p + "attention.indexer.top_k", .u32(2_048)),
            (p + "rope.freq_base", .f32(8_000_000)),
            (p + "attention.layer_norm_rms_epsilon", .f64(1.0e-5)),
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
            }
        }
        while data.count < 32 || data.count % 32 != 0 { data.append(0) }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm52-config-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    private func load(_ metadata: [(String, MetadataValue)]) throws -> GLM52Configuration {
        let url = try temporaryGGUF(metadata)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)
        return try GLM52Configuration(model: model)
    }

    func testValidatesExactGLM52Geometry() throws {
        let configuration = try load(validMetadata())
        let shape = configuration.shape

        XCTAssertEqual(shape, .v5_2)
        XCTAssertEqual(shape.metadataNamespace, "glm-dsa")
        XCTAssertEqual(shape.nLayer, 79)
        XCTAssertEqual(shape.inferenceLayerCount, 78)
        XCTAssertEqual(shape.queryProjectionWidth, 16_384)
        XCTAssertEqual(shape.queryNonRoPEWidth, 192)
        XCTAssertEqual(shape.indexerQueryWidth, 4_096)
        XCTAssertEqual(shape.nExpertUsed, 8)
        XCTAssertEqual(shape.nIndexerTopK, 2_048)
        XCTAssertEqual(shape.originalContextLength, 1_048_576)
        XCTAssertTrue(configuration.expertWeightsNormalized)
        XCTAssertEqual(configuration.descriptor.architecture.id, .glmDSA)
        XCTAssertEqual(configuration.descriptor.architecture.family, .glm)
        XCTAssertEqual(configuration.descriptor.architecture.backendAvailability,
                       .recognizedButNotImplemented)
        XCTAssertEqual(configuration.descriptor.layerCount, 78)
        XCTAssertTrue(configuration.descriptor.capabilities.contains(.mixtureOfExperts))
    }

    func testRejectsACompatibleLookingButDifferentShape() throws {
        var metadata = validMetadata()
        let key = "glm-dsa.attention.indexer.top_k"
        metadata[metadata.firstIndex(where: { $0.0 == key })!] = (key, .u32(1_024))

        XCTAssertThrowsError(try load(metadata)) { error in
            guard case GLM52ConfigurationError.mismatch(let gotKey, let expected, let got) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(gotKey, key)
            XCTAssertEqual(expected, "2048")
            XCTAssertEqual(got, "1024")
        }
    }

    func testRequiresExactGLMDSAArchitectureNamespace() throws {
        var metadata = validMetadata()
        metadata[0] = ("general.architecture", .string("deepseek4"))

        XCTAssertThrowsError(try load(metadata)) { error in
            XCTAssertEqual(
                error as? GLM52ConfigurationError,
                .wrongArchitecture(expected: "glm-dsa", got: "deepseek4")
            )
        }
    }

    func testRejectsFalseExpertWeightNormalization() throws {
        var metadata = validMetadata()
        let key = "glm-dsa.expert_weights_norm"
        metadata[metadata.firstIndex(where: { $0.0 == key })!] = (key, .bool(false))

        XCTAssertThrowsError(try load(metadata)) { error in
            XCTAssertEqual(
                error as? GLM52ConfigurationError,
                .mismatch(key, expected: "true", got: "false")
            )
        }
    }
}
