import Foundation

/// `swift test` — run the package's tests. Supports XCTest and the
/// newer Swift Testing (`@Test`) DSL transparently.
public struct SwiftTestTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "swift_test",
            description:
                "Run a Swift package's tests. 'filter' is forwarded to --filter (regex). " +
                "'parallel' enables --parallel.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "packagePath": SchemaBuilder.string(description: "Directory containing Package.swift. Default agent root."),
                    "filter": SchemaBuilder.string(description: "Regex passed to --filter."),
                    "parallel": SchemaBuilder.boolean(description: "Run tests in parallel. Default false.", defaultValue: false),
                    "configuration": SchemaBuilder.string(
                        description: "Build configuration. Default debug.",
                        enumValues: ["debug", "release"]),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 900.", minimum: 1),
                ]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "swift test\(input["filter"].map { " --filter \($0)" } ?? "")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pkg = try resolveInsideRoot(input.optionalString("packagePath") ?? ".",
                                        context: context)
        var args: [String] = ["test"]
        if let cfg = input.optionalString("configuration") {
            args.append("-c"); args.append(cfg)
        }
        if let f = input.optionalString("filter") {
            args.append("--filter"); args.append(f)
        }
        if input.optionalBool("parallel") ?? false {
            args.append("--parallel")
        }
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 900)
        return try await Xcrun.run(tool: "swift",
                                   arguments: args,
                                   context: context,
                                   cwd: pkg,
                                   timeout: timeout)
    }
}
