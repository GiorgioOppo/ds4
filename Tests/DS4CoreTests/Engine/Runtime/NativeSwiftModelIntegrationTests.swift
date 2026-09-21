import DS4Core
import Foundation
import XCTest
@testable import DS4Engine

final class NativeSwiftModelIntegrationTests: XCTestCase {
    private func inspect(_ values: [(String, GGUFMetadataValue)]) throws -> RuntimeModelDescriptor {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("native-inspect-\(UUID().uuidString).gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        try GGUFWriter(metadata: values).write(to: url.path)
        return try ModelInspector.inspect(GGUFModel(path: url.path, metalMapping: false, prefetchCPU: false))
    }

    func testExactArchitecturesSelectDedicatedTextBackendWithoutDeepSeekControls() throws {
        let cases: [(String, RuntimeBackendKind, String)] = [
            ("glm5-next", .glm53, "GLM 5.3 Flash"),
            ("qwen4exp", .qwen38, "Qwen 3.8 Flash Next"),
            ("deepseek41", .deepSeek41, "DeepSeek V4.1 Flash")
        ]
        for (architecture, backend, name) in cases {
            let descriptor = try ModelInspector.inspect(generalArchitecture: architecture)
            XCTAssertEqual(try BackendSelector.select(descriptor), backend)
            XCTAssertEqual(descriptor.displayName, name)
            XCTAssertEqual(descriptor.capabilities, .nativeText)
            XCTAssertTrue(descriptor.usesSwiftModelDecoder)
            XCTAssertFalse(descriptor.capabilities.contains(.deepSeekPerformanceTuning))
            XCTAssertFalse(descriptor.capabilities.contains(.diskKV))
            XCTAssertFalse(descriptor.capabilities.contains(.distributedPipeline))
        }
    }

    func testOrdinaryQwen35DoesNotBecomeBonsaiWithoutValidatedPrismCheckpoint() throws {
        let descriptors = [
            try ModelInspector.inspect(generalArchitecture: "qwen35"),
            try inspect([("general.architecture", .text("qwen35")),
                         ("prism.hadamard.version", .uint32(1)),
                         ("prism.hadamard.block_size", .uint32(1024))])
        ]
        for descriptor in descriptors {
            XCTAssertEqual(descriptor.backendAvailability, .recognizedButNotImplemented)
            XCTAssertFalse(descriptor.usesSwiftModelDecoder)
            XCTAssertFalse(descriptor.displayName.contains("Bonsai"))
            XCTAssertTrue(descriptor.capabilities.isEmpty)
            XCTAssertThrowsError(try BackendSelector.select(descriptor))
        }
    }

    func testDeepSeek41MetadataAndGLM53HyphenatedNamespace() throws {
        let ds = try inspect([
            ("general.architecture", .text("deepseek41")),
            ("deepseek41.num_hidden_layers", .uint64(43)),
            ("deepseek41.hidden_size", .uint32(4096)),
            ("deepseek41.vocab_size", .uint32(129280))
        ])
        XCTAssertEqual(ds.layerCount, 43); XCTAssertEqual(ds.embeddingLength, 4096)
        XCTAssertEqual(ds.vocabularySize, 129280)
        let glm = try inspect([
            ("general.architecture", .text("glm5-next")),
            ("glm5-next.block_count", .uint32(46)),
            ("glm5-next.trunk_block_count", .uint32(45)),
            ("glm5-next.embedding_length", .uint32(4096)),
            ("glm5next.block_count", .uint32(999)),
            ("tokenizer.ggml.tokens", .array(elementType: .string, elements: [.text("a"), .text("b")]))
        ])
        XCTAssertEqual(glm.layerCount, 45); XCTAssertEqual(glm.embeddingLength, 4096)
        XCTAssertEqual(glm.vocabularySize, 2)
    }

    func testSevenPinnedArtifactsAreTextModelsAndExcludeEncodersAndByteFragments() throws {
        let expected: [(ModelCatalogID, String, String, String, Int64, String)] = [
            (.bonsai2PQ2, "prism-ml/Ternary-Bonsai-2-27B-gguf", "6ed5e12bf84b7a63069882c91dd9e9218647d17b", "Ternary-Bonsai-2-27B-PQ2_0.gguf", 7206168928, "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"),
            (.bonsai2PTQ1, "prism-ml/Ternary-Bonsai-2-27B-gguf", "6ed5e12bf84b7a63069882c91dd9e9218647d17b", "Ternary-Bonsai-2-27B-PTQ1_0.gguf", 5946648928, "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"),
            (.qwen38Q2, "antirez/qwen3.8-flash-next-gguf", "d600fe1a43d2e1cdcadb85144ce3142f66f9eefe", "Qwen3.8-Flash-Next-Q2.gguf", 147207127040, "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9"),
            (.qwen38Q4, "antirez/qwen3.8-flash-next-gguf", "d600fe1a43d2e1cdcadb85144ce3142f66f9eefe", "Qwen3.8-Flash-Next-Q4.gguf", 177280286720, "680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a"),
            (.deepSeek41Q2, "antirez/deepseek-v4.1-flash-gguf", "dd8a266f7145edc19e2334b46e19b6821f221dc7", "DeepSeek-V4.1-Flash-Q2.gguf", 365713686528, "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"),
            (.glm53Q2, "antirez/glm-5.3-flash-gguf", "b2fa29d7a6b410db11221c904973967b80b760f5", "GLM-5.3-Flash-Q2.gguf", 96505816384, "e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32"),
            (.glm53Q4, "antirez/glm-5.3-flash-gguf", "b2fa29d7a6b410db11221c904973967b80b760f5", "GLM-5.3-Flash-Q4_K.gguf", 190875526464, "c7a0d950363238dd7804782c88340d737775aba53a15f8d4fdcc34e984f25221")
        ]
        for (id, repository, revision, filename, bytes, digest) in expected {
            let entry = try XCTUnwrap(ModelCatalogRegistry.entry(id))
            let target = try XCTUnwrap(entry.primaryArtifact)
            XCTAssertTrue(entry.isSelectable); XCTAssertFalse(entry.requiresVisionEncoder)
            XCTAssertEqual(target.role, .mainModel)
            XCTAssertEqual(target.file, filename); XCTAssertEqual(target.expectedSizeBytes, bytes)
            XCTAssertEqual(target.sha256, digest)
            XCTAssertEqual(ModelDownloader.resolveURL(target).absoluteString,
                "https://huggingface.co/\(repository)/resolve/\(revision)/\(filename)")
        }
        let newEntries = BonsaiModelCatalog.entries + Qwen38ModelCatalog.entries
            + DeepSeek41ModelCatalog.entries.filter { $0.assemblyOutput == nil } + GLM53ModelCatalog.entries
        XCTAssertEqual(newEntries.count, 7)
        XCTAssertFalse(newEntries.flatMap(\.artifacts).contains { $0.file.contains("Vision") || $0.file.contains("mmproj") || $0.file.contains(".part") })
    }
}
