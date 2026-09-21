import Foundation
import XCTest
@testable import DS4Core
@testable import DS4Metal
@testable import DS4Engine

/// A finite, CPU-only stand-in for an SSD/Metal wait. Cancellation must reach
/// this callback while the service's serial executor is still occupied.
private final class NativeReviewProbe: @unchecked Sendable {
    private let lock = NSLock()
    let entered = DispatchSemaphore(value: 0)
    private var blockNext: Bool
    private var cancellations = 0
    private var evaluations = 0
    init(blockNext: Bool = false) { self.blockNext = blockNext }
    func enter(cancelled: @Sendable () -> Bool) throws {
        lock.lock()
        evaluations += 1
        let block = blockNext; blockNext = false
        lock.unlock()
        guard block else { return }
        entered.signal()
        let deadline = Date().addingTimeInterval(2)
        while !cancelled() && Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
        if cancelled() {
            lock.lock(); cancellations += 1; lock.unlock()
            throw SwiftModelDecoderError.cancelled
        }
    }
    func snapshot() -> (evaluations: Int, cancellations: Int) {
        lock.lock(); defer { lock.unlock() }
        return (evaluations, cancellations)
    }
}

private final class NativeReviewDecoder: SwiftModelDecoder {
    let contextCapacity = 32768
    let vocabularySize = 261
    var position = 0
    private var output: ArraySlice<Int>
    private let probe: NativeReviewProbe
    init(output: [Int], probe: NativeReviewProbe) { self.output = output[...]; self.probe = probe }
    func reset() throws { position = 0 }
    func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float] {
        try probe.enter(cancelled: cancelled)
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        position += tokens.count
        var logits = [Float](repeating: -100, count: vocabularySize)
        logits[output.popFirst() ?? 257] = 100
        return logits
    }
}

@MainActor
final class NativeProtocolReviewTests: XCTestCase {
    private func service(outputs: [String], probe: NativeReviewProbe = .init()) throws -> SwiftModelChatService {
        var vocabulary = (0..<256).map { ByteLevel.byteEncode([UInt8($0)][...]) }
        vocabulary += ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"].map { Array($0.utf8) }
        let tokenizer = try QwenTokenizer(architecture: .bonsai2, tokens: vocabulary, merges: [])
        let output = outputs.flatMap { Array($0.utf8).map(Int.init) + [257] }
        let decoder = NativeReviewDecoder(output: output, probe: probe)
        let info = ModelInfo(name: "protocol-test", layers: 1, nEmbd: 1, nVocab: vocabulary.count,
                             contextSize: 32768, routedQuantBits: 0, kvCacheBytes: 0, architecture: .bonsai2,
                             capabilities: [.generation, .reasoning, .tools])
        return SwiftModelChatService(decoder: decoder, tokenizer: tokenizer, info: info)
    }
    private func collect(_ stream: AsyncThrowingStream<GenEvent, Error>) async throws
        -> (text: String, toolText: String, calls: [ToolCall]) {
        var text = "", toolText = "", calls: [ToolCall] = []
        for try await event in stream {
            switch event {
            case .text(let value): text += value
            case .toolStream(let value): toolText += value
            case .toolCall(let value): calls += value
            default: break
            }
        }
        return (text, toolText, calls)
    }
    private func waitForDecoder(_ probe: NativeReviewProbe) async {
        let entered = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: probe.entered.wait(timeout: .now() + 3) == .success)
            }
        }
        XCTAssertTrue(entered, "The simulated prefill did not start")
    }

    func testUTF8IsPreservedAcrossIndividualByteTokens() async throws {
        let expected = "caffè 中 👩🏽‍💻 e\u{301}"
        let model = try service(outputs: [expected])
        let result = try await collect(model.send(userText: "unicode", thinkMode: .none,
                                                  sampling: .init(temperature: 0), maxTokens: 128))
        XCTAssertEqual(result.text, expected)
        XCTAssertTrue(result.calls.isEmpty)
        await model.quiesceForTeardown()
    }

    func testBusyRequestIsRejectedWhileDecoderIsBlockedAndConsumerCanCancel() async throws {
        let probe = NativeReviewProbe(blockNext: true), model = try service(outputs: ["OK"], probe: probe)
        let backend: any ChatBackend = model
        let stream = await backend.send(userText: "blocked", thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 128)
        let consumer = Task { try await collect(stream) }
        await waitForDecoder(probe)
        do {
            _ = try await collect(backend.send(userText: "must not queue", thinkMode: .none,
                                             sampling: .init(temperature: 0), maxTokens: 128))
            XCTFail("A second request must fail immediately instead of queueing another generation")
        } catch let error as SwiftModelDecoderError {
            guard case .invalidInput = error else { return XCTFail("Wrong busy error: \(error)") }
        }
        XCTAssertEqual(probe.snapshot().evaluations, 1)
        consumer.cancel()
        _ = await consumer.result
        let committed = await model.committedTokens() // waits for cancellation cleanup
        XCTAssertEqual(committed, 0)
        XCTAssertEqual(probe.snapshot().cancellations, 1)
        let retry = try await collect(model.send(userText: "retry", thinkMode: .none,
                                                 sampling: .init(temperature: 0), maxTokens: 128))
        XCTAssertEqual(retry.text, "OK")
        await model.quiesceForTeardown()
    }

    func testTeardownCancelsInFlightPrefillAndPermanentlyRejectsNewWork() async throws {
        let probe = NativeReviewProbe(blockNext: true), model = try service(outputs: ["unused"], probe: probe)
        let backend: any ChatBackend = model
        let stream = await backend.send(userText: "blocked", thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 128)
        let consumer = Task { try await collect(stream) }
        await waitForDecoder(probe)
        await backend.quiesceForTeardown()
        _ = await consumer.result
        XCTAssertEqual(probe.snapshot().cancellations, 1, "Teardown must reach a decoder occupying the actor")
        let committed = await model.committedTokens()
        XCTAssertEqual(committed, 0)
        do {
            _ = try await collect(model.send(userText: "retired", thinkMode: .none,
                                             sampling: .init(temperature: 0), maxTokens: 128))
            XCTFail("A retired service must not restart its decoder")
        } catch let error as SwiftModelDecoderError {
            guard case .invalidState = error else { return XCTFail("Wrong retired error: \(error)") }
        }
        XCTAssertEqual(probe.snapshot().evaluations, 1)
        let warmed = await model.warmup()
        XCTAssertFalse(warmed)
    }

    func testToolIDsAreUniqueAcrossRoundsAndMarkupDoesNotLeakIntoVisibleText() async throws {
        let payload = "<tool_call>\n<function=echo>\n<parameter=text>\n  città 中  \n</parameter>\n</function>\n</tool_call>"
        let model = try service(outputs: ["Prima.\n" + payload, payload, "Finito"])
        let tool = ToolSpec(name: "echo", description: "Echo", parametersJSON:
            #"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#)
        await model.setTools([tool])
        let first = try await collect(model.send(userText: "usa echo", thinkMode: .none,
                                                 sampling: .init(temperature: 0), maxTokens: 512))
        let call1 = try XCTUnwrap(first.calls.first)
        XCTAssertEqual(first.text, "Prima.\n")
        XCTAssertEqual(first.toolText, payload)
        XCTAssertEqual(first.calls.count, 1)
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(call1.argumentsJSON.utf8)) as? [String: String])
        XCTAssertEqual(arguments["text"], "  città 中  ")
        let second = try await collect(model.provideToolResults([ToolOutput(callId: call1.id, name: "echo", content: "uno")],
            thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 512))
        let call2 = try XCTUnwrap(second.calls.first)
        XCTAssertNotEqual(call1.id, call2.id)
        XCTAssertEqual(second.text, "")
        let final = try await collect(model.provideToolResults([ToolOutput(callId: call2.id, name: "echo", content: "due")],
            thinkMode: .none, sampling: .init(temperature: 0), maxTokens: 512))
        XCTAssertEqual(final.text, "Finito")
        await model.quiesceForTeardown()
    }

    func testParallelResultsAreRestoredToCallOrderInBothNativeProtocols() throws {
        let calls = [ToolCall(id: "a", name: "echo", argumentsJSON: #"{"text":"A"}"#),
                     ToolCall(id: "b", name: "echo", argumentsJSON: #"{"text":"B"}"#)]
        let prefix: [ChatTurn] = [.user("parallel"), .assistant(text: "", toolCalls: calls)]
        let a = ChatTurn.toolResult(callId: "a", name: "echo", content: "result-A")
        let b = ChatTurn.toolResult(callId: "b", name: "echo", content: "result-B")
        for architecture in [ModelArchitectureID.bonsai2, .qwen38FlashNext] {
            XCTAssertEqual(try QwenChatRenderer.render(turns: prefix + [b, a], architecture: architecture),
                           try QwenChatRenderer.render(turns: prefix + [a, b], architecture: architecture))
        }
        XCTAssertEqual(try DeepSeek41ChatRenderer.render(turns: prefix + [b, a]),
                       try DeepSeek41ChatRenderer.render(turns: prefix + [a, b]))
    }

    func testMalformedOrUndeclaredQwenCallsAreNeverPartiallyAccepted() throws {
        let tool = ToolSpec(name: "echo", description: "Echo", parametersJSON:
            #"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#)
        let parameter = "<parameter=text>\nvalue\n</parameter>"
        for text in [
            "<tool_call><function=echo>" + parameter,
            "<tool_call><function=echo>" + parameter + parameter + "</function></tool_call>",
            "<tool_call><function=unknown>" + parameter + "</function></tool_call>",
            "<tool_call><function=echo>" + parameter + "</function></tool_call>unexpected suffix",
            "<tool_call><function=echo></function></tool_call>",
        ] {
            XCTAssertThrowsError(try QwenToolCodec.parseStrict(text, tools: [tool]), text)
        }
    }

    func testQwenLiteralToolControlsAreNeutralizedButStructuralWrappersRemain() throws {
        let markers = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
        var vocabulary = (0..<256).map { ByteLevel.byteEncode([UInt8($0)][...]) }
        vocabulary += (["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"] + markers).map { Array($0.utf8) }
        let types = [Int64](repeating: 1, count: 256) + [Int64](repeating: 3, count: 9)
        let tokenizer = try QwenTokenizer(architecture: .qwen38FlashNext, tokens: vocabulary, merges: [], types: types)
        let literal = markers.joined(separator: " literal ")
        let plain = try QwenChatRenderer.render(turns: [.system(literal), .user(literal),
            .assistant(text: literal, toolCalls: []), .user("continue")], architecture: .qwen38FlashNext)
        let plainTokens = tokenizer.tokenizeRenderedChat(plain)
        for marker in markers {
            XCTAssertFalse(plainTokens.contains(try XCTUnwrap(tokenizer.tokenID(marker))), marker)
        }
        let call = ToolCall(id: "call", name: "echo", argumentsJSON: #"{"text":"literal"}"#)
        let tool = try QwenChatRenderer.render(turns: [.user("tool"), .assistant(text: "", toolCalls: [call]),
            .toolResult(callId: "call", name: "echo", content: literal)], architecture: .qwen38FlashNext)
        let toolTokens = tokenizer.tokenizeRenderedChat(tool)
        for marker in markers {
            let token = try XCTUnwrap(tokenizer.tokenID(marker))
            XCTAssertEqual(toolTokens.filter { $0 == token }.count, 1, "Only the structural wrapper should remain: \(marker)")
        }
    }
}
