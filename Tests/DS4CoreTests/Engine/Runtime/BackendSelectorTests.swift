import XCTest
import DS4Core
@testable import DS4Engine

final class BackendSelectorTests: XCTestCase {
    func testDeepSeekV4SelectsExistingConcreteBackend() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "deepseek4",
            displayName: "DeepSeek V4 Flash"
        )

        XCTAssertEqual(descriptor.architecture, .deepSeekV4)
        XCTAssertEqual(descriptor.family, .deepSeek)
        XCTAssertEqual(descriptor.backendAvailability, .implemented)
        XCTAssertTrue(descriptor.capabilities.contains(.generation))
        XCTAssertTrue(descriptor.capabilities.contains(.deepSeekPerformanceTuning))
        XCTAssertEqual(try BackendSelector.select(descriptor), .deepSeekV4)
    }

    func testQwenIsRecognizedButRejectedBeforeDeepSeekConfiguration() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "QWEN-2",
            displayName: "Qwen 2"
        )

        XCTAssertEqual(descriptor.architecture, ModelArchitectureID("qwen2"))
        XCTAssertEqual(descriptor.family, .qwen)
        XCTAssertEqual(descriptor.backendAvailability, .recognizedButNotImplemented)
        XCTAssertTrue(descriptor.model.capabilities.contains(.chat))
        XCTAssertFalse(descriptor.model.capabilities.contains(.tools))
        XCTAssertTrue(descriptor.capabilities.isEmpty)

        XCTAssertThrowsError(try BackendSelector.select(descriptor)) { error in
            guard case BackendSelectionError.backendNotImplemented(let id) = error else {
                return XCTFail("errore inatteso: \(error)")
            }
            XCTAssertEqual(id, ModelArchitectureID("qwen2"))
            XCTAssertEqual(String(describing: error),
                           "backend qwen2 non ancora implementato")
        }
    }

    func testUnknownArchitectureHasDistinctUnsupportedError() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "future-transformer",
            displayName: "Future Transformer"
        )

        XCTAssertEqual(descriptor.family, .unknown)
        XCTAssertEqual(descriptor.backendAvailability, .unknown)
        XCTAssertThrowsError(try BackendSelector.select(descriptor)) { error in
            guard case BackendSelectionError.unsupportedArchitecture(let id) = error else {
                return XCTFail("errore inatteso: \(error)")
            }
            XCTAssertEqual(id, ModelArchitectureID("future-transformer"))
            XCTAssertNotEqual(String(describing: error),
                              "backend futuretransformer non ancora implementato")
            XCTAssertEqual(String(describing: error),
                           "architettura GGUF non supportata: futuretransformer")
        }
    }

    func testLegacyDeepSeekMetadataFallbackRemainsCompatible() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: nil,
            displayName: "Legacy DeepSeek",
            hasDeepSeekV4Metadata: true
        )
        XCTAssertEqual(try BackendSelector.select(descriptor), .deepSeekV4)
    }

    func testDeepSeekProProfileSelectsTheSharedDeepSeekBackend() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "deepseek4",
            displayName: "DeepSeek V4 Pro",
            layerCount: 61
        )

        XCTAssertTrue(DeepSeekV4BackendDefinition.supportsLocalRuntime(.pro))
        XCTAssertEqual(DeepSeekV4BackendDefinition.supportedLayerCounts, [43, 61])
        XCTAssertEqual(try BackendSelector.select(descriptor), .deepSeekV4)
    }

    func testUnknownDeepSeekProfileRemainsRejected() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "deepseek4",
            displayName: "DeepSeek V4 future profile",
            layerCount: 62
        )

        XCTAssertThrowsError(try BackendSelector.select(descriptor)) { error in
            guard case BackendSelectionError.unsupportedProfile(let id, let layers) = error else {
                return XCTFail("errore inatteso: \(error)")
            }
            XCTAssertEqual(id, .deepSeekV4)
            XCTAssertEqual(layers, 62)
        }
    }

    func testDeepSeekAccessoryIsNotSelectableAsAFullModel() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "deepseek4",
            displayName: "DeepSeek V4 MTP",
            layerCount: 1
        )
        XCTAssertThrowsError(try BackendSelector.select(descriptor))
    }

    func testLegacyModelInfoInitializerRemainsSourceCompatible() {
        let info = ModelInfo(name: "legacy.gguf", layers: 43, nEmbd: 4096,
                             nVocab: 129_280, contextSize: 4096,
                             routedQuantBits: 4, kvCacheBytes: 123)
        XCTAssertEqual(info.name, "legacy.gguf")
        XCTAssertEqual(info.displayName, "legacy.gguf")
        XCTAssertEqual(info.quantizationSummary, "routed 4-bit")
        XCTAssertTrue(info.capabilities.isEmpty)
    }
}
