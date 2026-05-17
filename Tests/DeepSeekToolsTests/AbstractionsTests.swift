import XCTest
@testable import DeepSeekTools

/// Test della fondazione OOP introdotta in
/// `Sources/DeepSeekTools/Abstractions/` + `Sources/DeepSeekTools/Demo/`.
/// Copre Plugin, Agent, Chat, ToolProvider, ModelBackend,
/// MCPTransport.
final class AbstractionsTests: XCTestCase {

    // MARK: - Plugin

    func testPluginRegistryBootstrapAndBroadcast() async throws {
        let registry = PluginRegistry()
        let logger = DemoLoggerPlugin()
        let host = DemoPluginHost(registry: registry)

        try await registry.register(logger, host: host)
        XCTAssertEqual(logger.bootstrapCount, 1)

        let env = Question(content: "ping")
        await registry.broadcast(env)
        XCTAssertEqual(logger.observedCount, 1)
        XCTAssertEqual(logger.observedKinds, ["chat.question"])

        await registry.shutdownAll()
        XCTAssertEqual(await registry.names(), [])
    }

    // MARK: - Agent

    func testDemoEchoAgentEmitsEnvelopes() async throws {
        let agent = DemoEchoAgent()
        let input = AgentInput(messages: [
            AgentChatMessage(role: .user, content: "ciao"),
        ])

        var tokens: [String] = []
        var sawDone = false
        for try await env in agent.step(input: input) {
            if let t = env as? AgentTokenEnvelope { tokens.append(t.text) }
            if env is AgentDoneEnvelope { sawDone = true }
        }

        XCTAssertEqual(tokens.joined(), "echo: ciao")
        XCTAssertTrue(sawDone)
    }

    // MARK: - Chat

    func testChatAskRoundTrip() async throws {
        let chat = DemoEchoChat()
        let answer = try await chat.ask("hello")

        XCTAssertEqual(chat.chat.turns.count, 1)
        XCTAssertEqual(chat.chat.turns[0].question.content, "hello")
        XCTAssertEqual(chat.chat.turns[0].question.role, .user)
        XCTAssertNotNil(chat.chat.turns[0].answer)
        XCTAssertTrue(answer.content.contains("echo: hello"))
        XCTAssertEqual(answer.questionID,
                       chat.chat.turns[0].question.envelopeID)
    }

    func testChatPublishesEnvelopesToPlugin() async throws {
        let registry = PluginRegistry()
        let logger = DemoLoggerPlugin()
        let host = DemoPluginHost(registry: registry)
        try await registry.register(logger, host: host)

        let chat = DemoEchoChat(plugins: registry)
        _ = try await chat.ask("ping")

        XCTAssertGreaterThanOrEqual(logger.observedCount, 2)
        XCTAssertTrue(logger.observedKinds.contains("chat.question"))
        XCTAssertTrue(logger.observedKinds.contains("chat.answer"))
    }

    func testChatBuilderFromTurnSource() throws {
        struct Src: ChatTurnSource {
            let sourceRole: ChatRole
            let sourceContent: String
            let sourceReasoning: String? = nil
            let sourceToolCalls: [ChatToolCall] = []
        }
        let turns: [any ChatTurnSource] = [
            Src(sourceRole: .user, sourceContent: "Q1"),
            Src(sourceRole: .assistant, sourceContent: "A1"),
            Src(sourceRole: .user, sourceContent: "Q2"),
        ]
        let chat = Chat.from(turns)
        XCTAssertEqual(chat.turns.count, 2)
        XCTAssertEqual(chat.turns[0].question.content, "Q1")
        XCTAssertEqual(chat.turns[0].answer?.content, "A1")
        XCTAssertEqual(chat.turns[1].question.content, "Q2")
        XCTAssertNil(chat.turns[1].answer)
    }

    // MARK: - ToolProvider

    func testToolProviderDiscoversTools() async throws {
        let provider = DemoBuiltinToolProvider()
        let tools = try await provider.discover()

        XCTAssertEqual(tools.count, 3)
        let names = Set(tools.map { $0.schema.name }).sorted()
        XCTAssertEqual(names, ["edit", "read", "write"])

        // Smoke integration: i tool del provider devono lavorare
        // attraverso un ToolRegistry esistente.
        let registry = ToolRegistry()
        for t in tools { await registry.register(t) }
        let registered = await registry.names().sorted()
        XCTAssertEqual(registered, ["edit", "read", "write"])
    }

    // MARK: - ModelBackend

    func testModelBackendStreamsDeltas() async throws {
        let backend = DemoEchoBackend()
        let req = GenerationRequest(messages: [
            AgentChatMessage(role: .user, content: "ciao"),
        ])

        var assembled = ""
        var lastWasFinal = false
        for try await delta in backend.generate(req) {
            assembled.append(delta.token)
            lastWasFinal = delta.isFinal
        }

        XCTAssertEqual(assembled, "echo: ciao")
        XCTAssertTrue(lastWasFinal)
    }

    // MARK: - MCPTransport

    func testMCPTransportLoopback() async throws {
        let transport = DemoStubMCPTransport()
        try await transport.connect()

        let payload = "hello mcp".data(using: .utf8)!

        let receiver = Task<Data?, Error> {
            for try await msg in transport.receive() {
                return msg
            }
            return nil
        }

        try await transport.send(payload)
        let echoed = try await receiver.value
        XCTAssertEqual(echoed, payload)

        await transport.disconnect()
    }

    // MARK: - MessageEnvelope contract

    func testQuestionAnswerEnvelopeContract() {
        let q = Question(content: "x")
        let a = Answer(content: "y", questionID: q.envelopeID)

        XCTAssertEqual(q.kind, "chat.question")
        XCTAssertEqual(a.kind, "chat.answer")
        XCTAssertGreaterThanOrEqual(q.schemaVersion, 1)
        XCTAssertGreaterThanOrEqual(a.schemaVersion, 1)
        XCTAssertNotEqual(q.envelopeID, a.envelopeID)
        XCTAssertEqual(a.questionID, q.envelopeID)
    }
}
