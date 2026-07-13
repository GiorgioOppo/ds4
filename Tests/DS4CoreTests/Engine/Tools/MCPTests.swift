import XCTest
@testable import DS4Engine

/// MCP client support: config interchange, JSON-RPC frame handling, result
/// translation, and tool-name namespacing. Everything here is pure (no
/// processes, no network) — the protocol layer is deliberately transport-free.
final class MCPTests: XCTestCase {

    // MARK: Config + `mcpServers` interchange

    func testSanitizeID() {
        XCTAssertEqual(MCPServerConfig.sanitizeID("GitHub Tools"), "github_tools")
        XCTAssertEqual(MCPServerConfig.sanitizeID("fs"), "fs")
        XCTAssertEqual(MCPServerConfig.sanitizeID("  weird--name!! "), "weird_name")
        XCTAssertEqual(MCPServerConfig.sanitizeID("///"), "server")
    }

    func testImportClaudeDesktopFormat() throws {
        let json = """
        {"mcpServers": {
           "filesystem": {"command": "npx",
                          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                          "env": {"DEBUG": "1"}},
           "remote": {"url": "https://example.com/mcp",
                      "headers": {"Authorization": "Bearer abc"}}
        }}
        """
        let configs = try MCPServerConfig.importJSON(json)
        XCTAssertEqual(configs.count, 2)

        let fs = try XCTUnwrap(configs.first { $0.name == "filesystem" })
        guard case .stdio(let command, let args, let env) = fs.transport else {
            return XCTFail("expected stdio transport")
        }
        XCTAssertEqual(command, "npx")
        XCTAssertEqual(args, ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        XCTAssertEqual(env, ["DEBUG": "1"])

        let remote = try XCTUnwrap(configs.first { $0.name == "remote" })
        guard case .http(let url, let headers) = remote.transport else {
            return XCTFail("expected http transport")
        }
        XCTAssertEqual(url, "https://example.com/mcp")
        XCTAssertEqual(headers, ["Authorization": "Bearer abc"])
    }

    func testImportRejectsGarbage() {
        XCTAssertThrowsError(try MCPServerConfig.importJSON("not json"))
        XCTAssertThrowsError(try MCPServerConfig.importJSON(#"{"mcpServers":{}}"#))
        XCTAssertThrowsError(try MCPServerConfig.importJSON(#"{"mcpServers":{"x":{"nope":1}}}"#))
    }

    func testExportImportRoundTrip() throws {
        let original = [
            MCPServerConfig(name: "fs", transport: .stdio(command: "uvx", arguments: ["server-fs"],
                                                          environment: [:])),
            MCPServerConfig(name: "api", transport: .http(url: "http://localhost:8808/mcp",
                                                          headers: ["X-Key": "k"])),
        ]
        let reimported = try MCPServerConfig.importJSON(MCPServerConfig.exportJSON(original))
        XCTAssertEqual(reimported.map(\.name).sorted(), ["api", "fs"])
        XCTAssertEqual(reimported.first { $0.name == "fs" }?.transport, original[0].transport)
        XCTAssertEqual(reimported.first { $0.name == "api" }?.transport, original[1].transport)
    }

    func testConfigCodableRoundTrip() throws {
        let config = MCPServerConfig(name: "My Server",
                                     transport: .stdio(command: "npx", arguments: ["-y", "x"],
                                                       environment: ["A": "b"]),
                                     enabled: false)
        let decoded = try JSONDecoder().decode(MCPServerConfig.self,
                                               from: try JSONEncoder().encode(config))
        XCTAssertEqual(decoded, config)
        XCTAssertEqual(decoded.id, "my_server")
    }

    // MARK: JSON-RPC frames

    func testRequestFrameShape() throws {
        let frame = MCPProtocol.request(id: 7, method: "tools/call",
                                        params: ["name": "t", "arguments": ["a": 1]])
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: frame) as? [String: Any])
        XCTAssertEqual(obj["jsonrpc"] as? String, "2.0")
        XCTAssertEqual(obj["id"] as? Int, 7)
        XCTAssertEqual(obj["method"] as? String, "tools/call")
        XCTAssertEqual((obj["params"] as? [String: Any])?["name"] as? String, "t")
    }

    func testClassifyResponseSuccessAndError() {
        let ok = Data(#"{"jsonrpc":"2.0","id":3,"result":{"x":1}}"#.utf8)
        guard case .response(let id, let result, let err) = MCPProtocol.classify(ok) else {
            return XCTFail("expected response")
        }
        XCTAssertEqual(id, 3)
        XCTAssertNil(err)
        XCTAssertEqual(result?["x"] as? Int, 1)

        let failed = Data(#"{"jsonrpc":"2.0","id":4,"error":{"code":-32000,"message":"boom"}}"#.utf8)
        guard case .response(_, _, let msg) = MCPProtocol.classify(failed) else {
            return XCTFail("expected response")
        }
        XCTAssertEqual(msg, "boom (code -32000)")
    }

    func testClassifyNotificationServerRequestAndInvalid() {
        let note = Data(#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}"#.utf8)
        guard case .notification(let m) = MCPProtocol.classify(note) else {
            return XCTFail("expected notification")
        }
        XCTAssertEqual(m, "notifications/tools/list_changed")

        let ping = Data(#"{"jsonrpc":"2.0","id":"srv-1","method":"ping"}"#.utf8)
        guard case .serverRequest(_, let method) = MCPProtocol.classify(ping) else {
            return XCTFail("expected server request")
        }
        XCTAssertEqual(method, "ping")

        guard case .invalid = MCPProtocol.classify(Data("nope".utf8)) else {
            return XCTFail("expected invalid")
        }
    }

    // MARK: Result translation

    func testParseToolsList() {
        let result: [String: Any] = [
            "tools": [
                ["name": "read_file",
                 "description": "Read a file.",
                 "inputSchema": ["type": "object",
                                 "properties": ["path": ["type": "string"]],
                                 "required": ["path"]]],
                ["name": "no_schema"],
                ["description": "nameless — skipped"],
            ],
            "nextCursor": "page2",
        ]
        let (tools, cursor) = MCPProtocol.parseToolsList(result)
        XCTAssertEqual(tools.map(\.name), ["read_file", "no_schema"])
        XCTAssertEqual(cursor, "page2")
        XCTAssertTrue(tools[0].inputSchemaJSON.contains(#""required":["path"]"#))
        XCTAssertEqual(tools[1].inputSchemaJSON, #"{"type":"object","properties":{}}"#)
    }

    func testFlattenCallResult() {
        let text: [String: Any] = ["content": [["type": "text", "text": "hello"],
                                               ["type": "text", "text": "world"]]]
        XCTAssertEqual(MCPProtocol.flattenCallResult(text), "hello\nworld")

        let error: [String: Any] = ["content": [["type": "text", "text": "no such file"]],
                                    "isError": true]
        XCTAssertEqual(MCPProtocol.flattenCallResult(error), "Tool error: no such file")

        let structured: [String: Any] = ["content": [],
                                         "structuredContent": ["count": 2]]
        XCTAssertEqual(MCPProtocol.flattenCallResult(structured), #"{"count":2}"#)

        let image: [String: Any] = ["content": [["type": "image", "mimeType": "image/png",
                                                 "data": "…"]]]
        XCTAssertEqual(MCPProtocol.flattenCallResult(image), "[image content: image/png]")

        XCTAssertEqual(MCPProtocol.flattenCallResult([:]), "(empty tool result)")
    }

    // MARK: SSE parsing (Streamable HTTP)

    func testSSEFrames() {
        let body = """
        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{}}

        data: line1
        data: line2

        """
        XCTAssertEqual(MCPHTTPTransport.sseFrames(body),
                       [#"{"jsonrpc":"2.0","id":1,"result":{}}"#, "line1\nline2"])
        XCTAssertEqual(MCPHTTPTransport.sseFrames("data:no-space\n"), ["no-space"])
        XCTAssertEqual(MCPHTTPTransport.sseFrames(""), [])
    }

    // MARK: Namespacing

    func testSpecName() {
        XCTAssertEqual(MCPManager.specName(server: "fs", tool: "read_file"), "mcp_fs_read_file")
        XCTAssertEqual(MCPManager.specName(server: "api", tool: "search.web"), "mcp_api_search_web")
    }
}
