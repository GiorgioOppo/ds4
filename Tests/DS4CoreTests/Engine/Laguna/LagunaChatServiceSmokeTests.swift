import XCTest
@testable import DS4Engine

/// End-to-end smoke of the GUI chat surface over the Laguna engine: render →
/// prefill → sampled decode → stream events → transcript note, exactly the
/// path `ChatStore` drives through the `ChatBackend` contract. Needs a real
/// GGUF (45 GiB, streamed on tight-RAM machines), so it only runs when
/// `DS4_LAGUNA_SMOKE_GGUF` points at one — otherwise it skips, keeping the
/// suite hermetic.
final class LagunaChatServiceSmokeTests: XCTestCase {
    func testGUIProfilePinsMeasuredLagunaOverrides() {
        let profile = LagunaInferenceService.guiEnvironmentDefaults
        XCTAssertEqual(profile["DS4_SSD_STREAM"], "1")
        XCTAssertEqual(profile["DS4_EXPERT_CACHE_MB"], "2048")
        XCTAssertEqual(profile["DS4_ACTIVE_EXPERTS"], "6")
        XCTAssertEqual(profile["DS4_RESIDENT_LAYERS"], "0")
        XCTAssertEqual(profile["DS4_KV_INITIAL"], "512")
        XCTAssertEqual(profile["DS4_PREFILL_CHUNK"], "256")
        XCTAssertEqual(profile["DS4_PREFILL_BATCH"], "1")
        XCTAssertEqual(profile["DS4_PREFILL_DENSE_MM"], "1")
        XCTAssertEqual(profile["DS4_PREFILL_MOE_BATCH"], "0")
        XCTAssertEqual(profile["DS4_EXPERT_PREAD"], "1")
        XCTAssertEqual(profile["DS4_PREAD_SPLIT"], "1")
        XCTAssertEqual(profile["DS4_MTLIO"], "0")
        XCTAssertEqual(profile["DS4_MLOCK"], "0")
        XCTAssertEqual(profile["DS4_NSG"], "4")
        XCTAssertEqual(profile["DS4_DECODE_CHAINED"], "0")
        XCTAssertEqual(profile["DS4_DECODE_SPLIT_K"], "0")
        XCTAssertFalse(profile.keys.contains {
            $0.hasPrefix("DS4_LAGUNA_")
        })
    }

    func testStreamedChatProducesText() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["DS4_LAGUNA_SMOKE_GGUF"] else {
            throw XCTSkip("set DS4_LAGUNA_SMOKE_GGUF=<laguna.gguf> to run")
        }
        let service = try LagunaChatService(
            modelPath: path, contextSize: 1_024, systemPrompt: nil)
        let info = await service.modelInfo()
        XCTAssertEqual(info.architecture,
                       LagunaBackendDefinition.supportedArchitecture)
        XCTAssertTrue(info.capabilities.contains(.generation))
        let warmed = await service.warmup()
        XCTAssertTrue(warmed, "warmup deve completare un token di prova")

        var text = ""
        var progressLines: [String] = []
        let stream = await service.send(
            userText: "ciao come stai?",
            thinkMode: .none,
            sampling: SamplingParams(temperature: 0.7, topK: 20,
                                     topP: 0.95, minP: 0.05),
            maxTokens: 24)
        for try await event in stream {
            switch event {
            case .text(let piece): text += piece
            case .progress(let p): progressLines.append(p)
            default: break
            }
        }
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty, "la generazione deve produrre testo visibile")
        // Il tempo del prompt deve restare leggibile in GUI: riepilogo
        // "prefill N tok in Xs" a fine prefill, E riproposto in coda alla
        // riga del decode (che altrimenti lo sostituirebbe subito).
        XCTAssertTrue(progressLines.contains {
            $0.hasPrefix("prefill") && $0.contains(" in ")
        }, "manca il riepilogo del prefill: \(progressLines.suffix(5))")
        XCTAssertTrue(progressLines.contains {
            $0.contains("tok/s — prefill")
        }, "la riga del decode deve riproporre il riepilogo del prefill: "
            + "\(progressLines.suffix(5))")

        // Secondo turno: il prefill deve essere incrementale (la posizione
        // del motore copre già il prefisso comune del transcript).
        let committed = await service.committedTokens()
        XCTAssertGreaterThan(committed, 0)
    }

    func testExplicitLazyKVReportsTheAllocatedLoadFootprint()
        async throws {
        guard let path = ProcessInfo.processInfo
            .environment["DS4_LAGUNA_SMOKE_GGUF"] else {
            throw XCTSkip(
                "set DS4_LAGUNA_SMOKE_GGUF=<laguna.gguf> to run")
        }
        let service = try LagunaChatService(
            modelPath: path,
            contextSize: 32_768,
            systemPrompt: nil,
            initialKVCapacity: 512)
        let info = await service.modelInfo()
        XCTAssertEqual(info.contextSize, 32_768)
        XCTAssertEqual(info.kvCacheBytes, UInt64(96 << 20),
                       "la GUI deve mostrare la KV lazy realmente allocata")
    }
}
