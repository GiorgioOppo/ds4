import XCTest
@testable import DeepSeekTools

/// Smoke tests for the tool runtime. Cover the registry's plan/build
/// filtering, the permission gate, and a couple of representative
/// tools (read / edit / glob / plan). Anything that touches the
/// network (`webfetch`, `websearch`) is out of scope here — those
/// live in integration tests against fixtures.
final class ToolRegistryTests: XCTestCase {

    func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("DeepSeekToolsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url,
                                                withIntermediateDirectories: true)
        return url
    }

    func ctx(root: URL,
             mode: AgentMode = .build,
             permission: PermissionDelegate = AutoPermissionDelegate(allowDangerous: true))
        -> ToolContext
    {
        ToolContext(rootDirectory: root, mode: mode, permission: permission)
    }

    // MARK: - registry

    func testPlanModeFiltersMutatingAndDangerousTools() async throws {
        let registry = ToolRegistry()
        let store = PlanStore()
        await registry.registerAll(
            DefaultTools.standard(planStore: store,
                                  includeShell: true,
                                  includeNetwork: false,
                                  includeRepoClone: false))

        let buildSchemas = await registry.availableSchemas(mode: .build)
        XCTAssertTrue(buildSchemas.contains { $0.name == "write" })
        XCTAssertTrue(buildSchemas.contains { $0.name == "edit" })
        XCTAssertTrue(buildSchemas.contains { $0.name == "shell" })

        let planSchemas = await registry.availableSchemas(mode: .plan)
        XCTAssertFalse(planSchemas.contains { $0.name == "write" })
        XCTAssertFalse(planSchemas.contains { $0.name == "edit" })
        XCTAssertFalse(planSchemas.contains { $0.name == "shell" })
        XCTAssertTrue(planSchemas.contains { $0.name == "read" })
        XCTAssertTrue(planSchemas.contains { $0.name == "plan" })
    }

    func testRegistryRejectsPlanModeMutation() async throws {
        let registry = ToolRegistry()
        await registry.register(WriteTool())
        let root = try makeTempRoot()
        let result = await registry.dispatch(
            name: "write",
            input: ["path": "foo.txt", "content": "hi"],
            context: ctx(root: root, mode: .plan))
        XCTAssertTrue(result.isError, "plan mode must reject 'write'")
        XCTAssertTrue(result.output.contains("denied"))
    }

    // MARK: - file tools

    func testReadWriteEditRoundTrip() async throws {
        let registry = ToolRegistry()
        let store = PlanStore()
        await registry.registerAll(
            DefaultTools.standard(planStore: store,
                                  includeShell: false,
                                  includeNetwork: false,
                                  includeRepoClone: false))
        let root = try makeTempRoot()
        let c = ctx(root: root)

        // write
        let w = await registry.dispatch(
            name: "write",
            input: ["path": "a.txt", "content": "hello\nworld"],
            context: c)
        XCTAssertFalse(w.isError)
        // read
        let r = await registry.dispatch(
            name: "read", input: ["path": "a.txt"], context: c)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.output.contains("hello"))
        XCTAssertTrue(r.output.contains("world"))
        // edit
        let e = await registry.dispatch(
            name: "edit",
            input: ["path": "a.txt", "oldString": "world", "newString": "universe"],
            context: c)
        XCTAssertFalse(e.isError, e.output)
        let updated = try String(contentsOf: root.appendingPathComponent("a.txt"))
        XCTAssertTrue(updated.contains("universe"))
    }

    func testEditRefusesAmbiguousMatch() async throws {
        let root = try makeTempRoot()
        try "abc\nabc\n".write(to: root.appendingPathComponent("dup.txt"),
                                atomically: true, encoding: .utf8)
        let r = ToolRegistry()
        await r.register(EditTool())
        let result = await r.dispatch(
            name: "edit",
            input: ["path": "dup.txt", "oldString": "abc", "newString": "xyz"],
            context: ctx(root: root))
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("matches"))
    }

    func testPathEscapeDenied() async throws {
        let root = try makeTempRoot()
        let r = ToolRegistry()
        await r.register(ReadTool())
        let result = await r.dispatch(
            name: "read",
            input: ["path": "../../etc/passwd"],
            context: ctx(root: root))
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("permission_denied"))
    }

    // MARK: - glob

    func testGlobFindsFiles() async throws {
        let root = try makeTempRoot()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("Sources/A"),
            withIntermediateDirectories: true)
        try "// a".write(to: root.appendingPathComponent("Sources/A/a.swift"),
                          atomically: true, encoding: .utf8)
        try "// b".write(to: root.appendingPathComponent("Sources/A/b.swift"),
                          atomically: true, encoding: .utf8)
        let r = ToolRegistry()
        await r.register(GlobTool())
        let result = await r.dispatch(
            name: "glob",
            input: ["pattern": "Sources/**/*.swift"],
            context: ctx(root: root))
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("a.swift"))
        XCTAssertTrue(result.output.contains("b.swift"))
    }

    // MARK: - planning

    func testPlanToolReadAndWrite() async throws {
        let store = PlanStore()
        let r = ToolRegistry()
        await r.register(PlanTool(store: store))
        let root = try makeTempRoot()
        let write = await r.dispatch(
            name: "plan",
            input: ["operation": "write", "text": "1. read\n2. write"],
            context: ctx(root: root))
        XCTAssertFalse(write.isError)
        let read = await r.dispatch(
            name: "plan",
            input: ["operation": "read"],
            context: ctx(root: root))
        XCTAssertEqual(read.output, "1. read\n2. write")
    }
}
