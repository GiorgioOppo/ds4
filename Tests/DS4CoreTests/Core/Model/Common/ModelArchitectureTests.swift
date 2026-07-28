import XCTest
import Foundation
@testable import DS4Core

final class ModelArchitectureTests: XCTestCase {
    private func temporaryGGUF(metadata: [(key: String, string: String?, u32: UInt32?)]) throws -> URL {
        var data = Data()
        func appendU32(_ value: UInt32) {
            var value = value.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendU64(_ value: UInt64) {
            var value = value.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendString(_ value: String) {
            let bytes = Data(value.utf8)
            appendU64(UInt64(bytes.count))
            data.append(bytes)
        }

        appendU32(GGUF.magic)
        appendU32(3)
        appendU64(0) // tensors
        appendU64(UInt64(metadata.count))
        for item in metadata {
            appendString(item.key)
            if let string = item.string {
                appendU32(GGUFValueType.string.rawValue)
                appendString(string)
            } else if let u32 = item.u32 {
                appendU32(GGUFValueType.uint32.rawValue)
                appendU32(u32)
            }
        }
        while data.count < 32 || data.count % 32 != 0 { data.append(0) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("architecture-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    func testArchitectureIDNormalization() {
        XCTAssertEqual(ModelArchitectureID("  DeepSeek-V4\n"), .deepSeekV4)
        XCTAssertEqual(ModelArchitectureID("GLM_DSA"), .glmDSA)
        XCTAssertEqual(ModelArchitectureID.glmDSA.ggufMetadataNamespace, "glm-dsa")
        XCTAssertEqual(ModelArchitectureID("QWEN_2.MOE").rawValue, "qwen2moe")
        XCTAssertEqual(ModelArchitectureID.normalize(" Qwen 3 "), "qwen3")
    }

    func testDetectsExplicitDeepSeekV4() throws {
        let detected = try ModelArchitectureDetector.detect(
            generalArchitecture: "deepseek4",
            hasDeepSeekV4Metadata: false
        )
        XCTAssertEqual(detected.id, .deepSeekV4)
        XCTAssertEqual(detected.family, .deepSeek)
        XCTAssertEqual(detected.backendAvailability, .implemented)
        XCTAssertNoThrow(try ModelArchitectureDetector.requireImplemented(detected))
    }

    func testLegacyDeepSeekMetadataFallback() throws {
        let detected = try ModelArchitectureDetector.detect(
            generalArchitecture: nil,
            hasDeepSeekV4Metadata: true
        )
        XCTAssertEqual(detected.id, .deepSeekV4)
        XCTAssertEqual(detected.backendAvailability, .implemented)

        let blank = try ModelArchitectureDetector.detect(
            generalArchitecture: "  \n",
            hasDeepSeekV4Metadata: true
        )
        XCTAssertEqual(blank, detected)
    }

    func testExplicitArchitectureIsAuthoritativeOverFallbackMetadata() throws {
        let detected = try ModelArchitectureDetector.detect(
            generalArchitecture: "qwen2",
            hasDeepSeekV4Metadata: true
        )
        XCTAssertEqual(detected.id, ModelArchitectureID("qwen2"))
        XCTAssertEqual(detected.family, .qwen)
        XCTAssertEqual(detected.backendAvailability, .recognizedButNotImplemented)
    }

    func testRecognizesQwenFamilyWithoutClaimingBackendSupport() throws {
        for value in ["qwen", "qwen2", "Qwen2-MoE", "qwen3", "qwen3_moe"] {
            let detected = try ModelArchitectureDetector.detect(generalArchitecture: value)
            XCTAssertEqual(detected.family, .qwen, value)
            XCTAssertEqual(detected.backendAvailability, .recognizedButNotImplemented, value)

            XCTAssertThrowsError(try ModelArchitectureDetector.requireImplemented(detected)) { error in
                guard case ModelArchitectureError.backendNotImplemented(let id, let family) = error else {
                    return XCTFail("unexpected error for \(value): \(error)")
                }
                XCTAssertEqual(id, ModelArchitectureID(value))
                XCTAssertEqual(family, .qwen)
            }
        }
    }

    func testRecognizesGLMDSAWithoutClaimingRuntimeSupport() throws {
        for value in ["glm-dsa", "GLM_DSA", "glm.dsa"] {
            let detected = try ModelArchitectureDetector.detect(generalArchitecture: value)
            XCTAssertEqual(detected.id, .glmDSA, value)
            XCTAssertEqual(detected.family, .glm, value)
            XCTAssertEqual(detected.backendAvailability, .recognizedButNotImplemented, value)

            XCTAssertThrowsError(try ModelArchitectureDetector.requireImplemented(detected)) { error in
                guard case ModelArchitectureError.backendNotImplemented(let id, let family) = error else {
                    return XCTFail("unexpected error for \(value): \(error)")
                }
                XCTAssertEqual(id, .glmDSA)
                XCTAssertEqual(family, .glm)
            }
        }
    }

    func testRecognizesLagunaWithoutClaimingRuntimeSupport() throws {
        for value in ["laguna", "Laguna", " LAGUNA "] {
            let detected = try ModelArchitectureDetector.detect(generalArchitecture: value)
            XCTAssertEqual(detected.id, .laguna, value)
            XCTAssertEqual(detected.id.ggufMetadataNamespace, "laguna", value)
            XCTAssertEqual(detected.family, .laguna, value)
            XCTAssertEqual(detected.backendAvailability, .recognizedButNotImplemented, value)

            XCTAssertThrowsError(try ModelArchitectureDetector.requireImplemented(detected)) { error in
                guard case ModelArchitectureError.backendNotImplemented(let id, let family) = error else {
                    return XCTFail("unexpected error for \(value): \(error)")
                }
                XCTAssertEqual(id, .laguna)
                XCTAssertEqual(family, .laguna)
            }
        }
    }

    func testUnknownArchitectureIsDetectedButRejectedByRuntimeBoundary() throws {
        let detected = try ModelArchitectureDetector.detect(generalArchitecture: "future-llm")
        XCTAssertEqual(detected.family, .unknown)
        XCTAssertEqual(detected.backendAvailability, .unknown)
        XCTAssertThrowsError(try ModelArchitectureDetector.requireImplemented(detected)) { error in
            XCTAssertEqual(error as? ModelArchitectureError,
                           .unsupportedArchitecture(ModelArchitectureID("future-llm")))
        }
    }

    func testMissingArchitectureWithoutFallbackThrows() {
        XCTAssertThrowsError(try ModelArchitectureDetector.detect(
            generalArchitecture: nil,
            hasDeepSeekV4Metadata: false
        )) { error in
            XCTAssertEqual(error as? ModelArchitectureError, .missingArchitecture)
        }
    }

    func testGGUFDetectorUsesLegacyDeepSeekFallback() throws {
        let url = try temporaryGGUF(metadata: [
            (key: "deepseek4.block_count", string: nil, u32: 43),
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)
        XCTAssertEqual(try ModelArchitectureDetector.detect(in: model).id, .deepSeekV4)
    }

    func testDeepSeekConfigurationRejectsQwenBeforeReadingDeepSeekShape() throws {
        let url = try temporaryGGUF(metadata: [
            (key: "general.architecture", string: "qwen3", u32: nil),
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)

        XCTAssertThrowsError(try DeepSeekV4Configuration(model: model)) { error in
            guard case ModelArchitectureError.backendNotImplemented(let id, let family) = error else {
                return XCTFail("expected backendNotImplemented before DeepSeek metadata validation, got \(error)")
            }
            XCTAssertEqual(id, ModelArchitectureID("qwen3"))
            XCTAssertEqual(family, .qwen)
        }
    }

    func testPortableDescriptorDoesNotRequireBackendSpecificFields() throws {
        let architecture = try ModelArchitectureDetector.detect(generalArchitecture: "qwen3")
        let descriptor = ModelDescriptor(
            architecture: architecture,
            name: "Qwen fixture",
            layerCount: 32,
            embeddingLength: 4096,
            vocabularySize: 151_936,
            capabilities: [.chat, .reasoning]
        )
        XCTAssertEqual(descriptor.architecture.family, .qwen)
        XCTAssertTrue(descriptor.capabilities.contains(.chat))
        XCTAssertFalse(descriptor.capabilities.contains(.mixtureOfExperts))
    }
}
