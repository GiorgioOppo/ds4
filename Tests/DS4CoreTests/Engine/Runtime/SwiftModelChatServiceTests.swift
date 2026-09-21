import Foundation
import XCTest
@testable import DS4Core
@testable import DS4Metal
@testable import DS4Engine

private final class NativeDecoderLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [[Int]] = []
    private var resets = 0
    func append(_ tokens: [Int]) { lock.lock(); defer { lock.unlock() }; entries.append(tokens) }
    func reset() { lock.lock(); defer { lock.unlock() }; resets += 1 }
    func snapshot() -> (calls: [[Int]], resets: Int) { lock.lock(); defer { lock.unlock() }; return (entries, resets) }
}
private final class NativeTestDecoder: SwiftModelDecoder {
    let contextCapacity: Int
    let vocabularySize = 261
    var position = 0
    let log: NativeDecoderLog
    var failFirst: Bool
    init(capacity: Int, log: NativeDecoderLog, failFirst: Bool = false) { contextCapacity = capacity; self.log = log; self.failFirst = failFirst }
    func reset() throws { position = 0; log.reset() }
    func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float] {
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        log.append(tokens)
        position += tokens.count
        if failFirst { failFirst = false; throw SwiftModelDecoderError.invalidState }
        let next = tokens.count == 1 && tokens[0] == 79 ? 75 : 79
        var logits = [Float](repeating: -100, count: vocabularySize); logits[next] = 100
        return logits
    }
}

@MainActor
final class SwiftModelChatServiceTests: XCTestCase {
    private func service(capacity: Int = 2048, log: NativeDecoderLog, failFirst: Bool = false) throws -> SwiftModelChatService {
        var tokens = (0..<256).map { ByteLevel.byteEncode([UInt8($0)][...]) }
        tokens += ["<|im_start|>","<|im_end|>","<|endoftext|>","<think>","</think>"].map { Array($0.utf8) }
        let tokenizer = try QwenTokenizer(architecture: .bonsai2, tokens: tokens, merges: [])
        let info = ModelInfo(name: "test", layers: 1, nEmbd: 1, nVocab: tokens.count, contextSize: capacity,
                             routedQuantBits: 0, kvCacheBytes: 0, architecture: .bonsai2,
                             capabilities: [.generation, .reasoning, .tools])
        return SwiftModelChatService(decoder: NativeTestDecoder(capacity: capacity, log: log, failFirst: failFirst), tokenizer: tokenizer, info: info)
    }
    private func answer(_ stream: AsyncThrowingStream<GenEvent, Error>) async throws -> String {
        var text = ""
        for try await event in stream { if case .text(let value) = event { text += value } }
        return text
    }
    func testExactPrefixReuseAndHistoryReplacement() async throws {
        let log = NativeDecoderLog(), model = try service(log: log)
        let first = try await answer(model.send(userText: "uno", thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 2))
        XCTAssertEqual(first, "OK")
        let initial = log.snapshot()
        XCTAssertEqual(initial.resets, 1)
        _ = try await answer(model.send(userText: "due", thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 2))
        XCTAssertEqual(log.snapshot().resets, 1, "The exact cached prefix should survive the next turn")
        _ = try await answer(model.sendWithHistory([], userText: "nuova chat", systemPrompt: nil, thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 2))
        XCTAssertEqual(log.snapshot().resets, 2, "Different history resets the recurrent and KV state")
        let committed = await model.committedTokens()
        XCTAssertGreaterThan(committed, 0)
        await model.quiesceForTeardown()
        let empty = await model.committedTokens()
        XCTAssertEqual(empty, 0)
    }
    func testEvaluationFailureResetsBeforeRetry() async throws {
        let log = NativeDecoderLog(), model = try service(log: log, failFirst: true)
        do {
            _ = try await answer(model.send(userText: "one", thinkMode: .none, sampling: .init(), maxTokens: 2))
            XCTFail("Expected failed decoder")
        } catch {}
        XCTAssertEqual(log.snapshot().resets, 2)
        let committed = await model.committedTokens(); XCTAssertEqual(committed, 0)
        let retry = try await answer(model.send(userText: "one", thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 2))
        XCTAssertEqual(retry, "OK")
        XCTAssertEqual(log.snapshot().resets, 3)
    }
    func testOverflowDoesNotDispatchAndToolStreamDelimitersDoNotLeak() async throws {
        let log = NativeDecoderLog(), model = try service(capacity: 8, log: log)
        do {
            _ = try await answer(model.send(userText: String(repeating: "a", count: 30), thinkMode: .none, sampling: .init(), maxTokens: 2))
            XCTFail("Expected context overflow")
        } catch let error as InferenceError {
            guard case .contextExceeded = error else { return XCTFail("Wrong inference error") }
        }
        XCTAssertTrue(log.snapshot().calls.isEmpty)
        var stream = NativeToolTextStream(marker: "<tool_call>")
        var plain = "", tool = ""
        for chunk in ["Ready ", "<to", "ol_", "call>", "<function=x>"] {
            for event in stream.feed(chunk) {
                if case .text(let text) = event { plain += text }
                if case .toolStream(let text) = event { tool += text }
            }
        }
        XCTAssertEqual(plain, "Ready "); XCTAssertEqual(tool, "<tool_call><function=x>")
    }
}
