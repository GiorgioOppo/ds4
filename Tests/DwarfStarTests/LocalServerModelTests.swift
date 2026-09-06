import Foundation
import XCTest
import DS4Core
import DS4Engine
@testable import DwarfStar

final class LocalServerModelTests: XCTestCase, @unchecked Sendable {
    func testDiscoveryReportsLoadedContextSeparatelyFromCompletionDefault() async throws {
        let server = makeServer(contextSize: 32_768, completionDefault: 512)
        let list = try object(await server.modelsJSON())
        let models = try XCTUnwrap(list["data"] as? [[String: Any]])
        XCTAssertEqual(list["object"] as? String, "list")
        XCTAssertEqual(models.count, 1)
        let advertised = try XCTUnwrap(models.first)
        XCTAssertEqual(advertised["context_length"] as? Int, 32_768)
        XCTAssertEqual(advertised["max_completion_tokens"] as? Int, 512)
        XCTAssertEqual(advertised["id"] as? String, "loaded-\"model")
        XCTAssertEqual(advertised["owned_by"] as? String, "dwarfstar")

        let detail = try object(await server.modelJSON(server.resolveModel("other-model")))
        XCTAssertTrue(NSDictionary(dictionary: advertised).isEqual(to: detail))
    }

    func testUnknownContextIsNotReplacedByAnInventedCapacity() async throws {
        for unknown in [0, -1] {
            let server = makeServer(contextSize: unknown, completionDefault: 8_192)
            let model = try object(await server.modelJSON(server.modelId))
            XCTAssertNil(model["context_length"])
            XCTAssertEqual(model["max_completion_tokens"] as? Int, 8_192)
        }
    }

    func testContextOverflowUsesRecoverableOpenAIErrorCode() throws {
        let data = LocalServer.inferenceErrorResponse(
            .contextExceeded(prompt: 8_193, context: 8_192), cors: true)
        let http = try XCTUnwrap(String(data: data, encoding: .utf8))
        let separator = try XCTUnwrap(http.range(of: "\r\n\r\n"))
        let headers = String(http[..<separator.lowerBound])
        let body = String(http[separator.upperBound...])
        let error = try XCTUnwrap(try object(body)["error"] as? [String: Any])

        XCTAssertTrue(headers.hasPrefix("HTTP/1.1 400 Bad Request\r\n"))
        XCTAssertTrue(headers.contains("Content-Type: application/json\r\n"))
        XCTAssertTrue(headers.contains("Content-Length: \(body.utf8.count)\r\n"))
        XCTAssertTrue(headers.contains("Access-Control-Allow-Origin: *\r\n"))
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertEqual(error["code"] as? String, "context_length_exceeded")
        let message = try XCTUnwrap(error["message"] as? String)
        XCTAssertTrue(message.lowercased().contains("context length exceeded"))
        XCTAssertTrue(message.contains("8193"))
        XCTAssertTrue(message.contains("8192"))
    }

    func testOrdinaryBadRequestIsNotLabeledAsContextOverflow() throws {
        let message = "invalid JSON body: \"quoted\"\nnext line"
        let data = LocalServer.httpError(400, message, cors: false)
        let http = try XCTUnwrap(String(data: data, encoding: .utf8))
        let separator = try XCTUnwrap(http.range(of: "\r\n\r\n"))
        let body = String(http[separator.upperBound...])
        let error = try XCTUnwrap(try object(body)["error"] as? [String: Any])

        XCTAssertEqual(error["message"] as? String, message)
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertNil(error["code"])
        XCTAssertFalse(http[..<separator.lowerBound].contains("Access-Control-Allow-Origin"))
    }

    private func makeServer(contextSize: Int, completionDefault: Int) -> LocalServer {
        LocalServer(backend: ModelMetadataBackend(contextSize: contextSize),
                    modelName: "loaded-\"model.gguf",
                    config: .init(host: "127.0.0.1", port: 8080, cors: false,
                                  maxTokens: completionDefault), onLog: { _ in })
    }

    private func object(_ json: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }
}

private actor ModelMetadataBackend: ChatBackend {
    let contextSize: Int
    init(contextSize: Int) { self.contextSize = contextSize }

    func modelInfo() -> ModelInfo {
        ModelInfo(name: "test", layers: 1, nEmbd: 1, nVocab: 1,
                  contextSize: contextSize, routedQuantBits: 4, kvCacheBytes: 0)
    }

    func warmup() async -> Bool { true }
    func quiesceForTeardown() async {}
    func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) {}
    func setTools(_ tools: [ToolSpec]) {}
    func setCompactTools(_ on: Bool) {}
    func committedTokens() -> Int { 0 }
    func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
              maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        fatalError("Model discovery must not invoke inference")
    }
    func sendWithHistory(_ history: [ChatTurn], userText: String, systemPrompt: String?,
                         thinkMode: DS4ThinkMode, sampling: SamplingParams,
                         maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        fatalError("Model discovery must not invoke inference")
    }
    func provideToolResults(_ outputs: [ToolOutput], thinkMode: DS4ThinkMode,
                            sampling: SamplingParams,
                            maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        fatalError("Model discovery must not invoke inference")
    }
    func complete(turns: [ChatTurn], tools: [ToolSpec], thinkMode: DS4ThinkMode,
                  sampling: SamplingParams,
                  maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        fatalError("Model discovery must not invoke inference")
    }
}
