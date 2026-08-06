import XCTest
@testable import DS4Engine

final class GLM52ChatStreamSplitterTests: XCTestCase {
    func testDecodeProgressKeepsPrefillTimingVisible() {
        let prefill = "prefill 512 tok in 20.0s · 25.60 tok/s"

        XCTAssertEqual(
            GLM52ChatService.decodeProgressSummary(
                produced: 4, elapsed: 2, prefillSummary: prefill),
            "4 tok · 2.00 tok/s — \(prefill)")
    }

    func testLivePrefixRewindReevaluatesFinalPromptToken() {
        XCTAssertEqual(
            GLM52ChatService.livePrefixRewindTarget(
                oldPosition: 17, promptLength: 8,
                commonPrefix: 8, liveTokenCount: 17),
            7)
        XCTAssertEqual(
            GLM52ChatService.livePrefixRewindTarget(
                oldPosition: 49_826, promptLength: 48_379,
                commonPrefix: 48_379, liveTokenCount: 49_826),
            48_378)
    }

    func testLivePrefixRewindRejectsUnsafeStates() {
        XCTAssertNil(GLM52ChatService.livePrefixRewindTarget(
            oldPosition: 17, promptLength: 8,
            commonPrefix: 7, liveTokenCount: 17))
        XCTAssertNil(GLM52ChatService.livePrefixRewindTarget(
            oldPosition: 8, promptLength: 8,
            commonPrefix: 8, liveTokenCount: 8))
        XCTAssertNil(GLM52ChatService.livePrefixRewindTarget(
            oldPosition: 17, promptLength: 1,
            commonPrefix: 1, liveTokenCount: 17))
        XCTAssertNil(GLM52ChatService.livePrefixRewindTarget(
            oldPosition: 17, promptLength: 8,
            commonPrefix: 8, liveTokenCount: 16))
    }

    func testOrdinaryTextIsEmittedWithoutFixedTailDelay() {
        var splitter = GLM52ChatService.StreamSplitter(startsInThink: false)

        XCTAssertEqual(text(from: splitter.feed(Array("ciao".utf8))), "ciao")
        XCTAssertEqual(text(from: splitter.feed(Array("!".utf8))), "!")
        XCTAssertTrue(splitter.flush().isEmpty)
    }

    func testOnlyPossibleMarkerPrefixIsHeld() {
        var splitter = GLM52ChatService.StreamSplitter(startsInThink: false)

        XCTAssertEqual(text(from: splitter.feed(Array("risposta<".utf8))),
                       "risposta")
        XCTAssertTrue(splitter.feed(Array("think>".utf8)).isEmpty)
        XCTAssertEqual(reasoning(from: splitter.feed(Array("passo".utf8))),
                       "passo")
    }

    func testUTF8ScalarSplitAcrossTokensIsReassembled() {
        var splitter = GLM52ChatService.StreamSplitter(startsInThink: false)
        let bytes = Array("è".utf8)

        XCTAssertTrue(splitter.feed([bytes[0]]).isEmpty)
        XCTAssertEqual(text(from: splitter.feed([bytes[1]])), "è")
        XCTAssertTrue(splitter.flush().isEmpty)
    }

    func testFlushPublishesHeldMarkerPrefix() {
        var splitter = GLM52ChatService.StreamSplitter(startsInThink: false)

        XCTAssertTrue(splitter.feed(Array("<".utf8)).isEmpty)
        XCTAssertEqual(text(from: splitter.flush()), "<")
    }

    private func text(from events: [GenEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .text(let value) = event { result += value }
        }
    }

    private func reasoning(from events: [GenEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .reasoning(let value) = event { result += value }
        }
    }
}
