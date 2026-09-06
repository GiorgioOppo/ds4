import XCTest
@testable import DS4Engine

final class DeepSeekV4VisionCatalogTests: XCTestCase {
    func testVisionCompanionsNeverBecomeLanguageModelChoices() throws {
        XCTAssertEqual(DeepSeekV4VisionCatalog.entries.count, 3)
        XCTAssertTrue(DeepSeekV4VisionCatalog.entries.allSatisfy(\.requiresVisionEncoder))
        for id in [ModelCatalogID.visionEncoder, .visionDSparkSupport] {
            XCTAssertNil(ModelCatalogRegistry.entry(id))
            let entry = try XCTUnwrap(ModelCatalogRegistry.downloadEntry(id))
            XCTAssertFalse(entry.isSelectable)
            XCTAssertNil(entry.primaryArtifact)
            XCTAssertEqual(entry.artifacts.first?.role, .optionalComponent)
            XCTAssertEqual(ModelDownloader.target(id.rawValue), entry.artifacts.first)
        }
        XCTAssertEqual(DeepSeekV4VisionCatalog.encoder.expectedSizeBytes, 932_857_760)
        XCTAssertEqual(DeepSeekV4VisionCatalog.encoder.sha256,
                       "00cd4d81a435364967400a95c42703343e11da6b6f18c5143fe76e1d94d5035f")
        XCTAssertFalse(try XCTUnwrap(ModelCatalogRegistry.entry(.flashVisionMXFP4)).isSelectable)
        XCTAssertTrue(try XCTUnwrap(ModelCatalogRegistry.entry(.flashVisionQ2)).isSelectable)
        XCTAssertTrue(try XCTUnwrap(ModelCatalogRegistry.entry(.flashVisionQ2Q4)).isSelectable)
    }

    func testVisionArtifactsResolveAtOneImmutableRevision() {
        let artifacts = (DeepSeekV4VisionCatalog.entries
            + DeepSeekV4VisionCatalog.accessoryEntries).flatMap(\.artifacts)
        XCTAssertEqual(artifacts.count, 5)
        XCTAssertEqual(Set(artifacts.map(\.id)).count, 5)
        for target in artifacts {
            XCTAssertEqual(target.source.repository, "antirez/deepseek-v4-gguf")
            XCTAssertEqual(target.source.revision, "f71f23d552d664e523b422157b2befbf74040380")
            XCTAssertEqual(target.sha256?.count, 64)
            XCTAssertGreaterThan(target.expectedSizeBytes ?? 0, 0)
            XCTAssertEqual(ModelDownloader.resolveURL(target).absoluteString,
                "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/\(target.source.revision)/\(target.file)")
        }
    }
}
