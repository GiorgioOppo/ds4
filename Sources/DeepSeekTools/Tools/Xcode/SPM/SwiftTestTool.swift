import Foundation

/// `swift test` — run the package's tests. Supports XCTest and the
/// newer Swift Testing (`@Test`) DSL transparently.
public struct SwiftTestTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "swift_test",
            description:
                "Esegue i test di un pacchetto Swift Package Manager (funzionano sia XCTest sia Swift Testing). " +
                "Usalo per un pacchetto con Package.swift — non serve scheme né destination. " +
                "Per test dentro un progetto Xcode usa 'xcodebuild_test' (che richiede scheme + destination). " +
                "'filter' viene passato a --filter (regex). 'parallel' abilita --parallel.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "packagePath": SchemaBuilder.string(description: "Directory contenente Package.swift. Default: root dell'agente."),
                    "filter": SchemaBuilder.string(description: "Regex passata a --filter."),
                    "parallel": SchemaBuilder.boolean(description: "Esegue i test in parallelo. Default false.", defaultValue: false),
                    "configuration": SchemaBuilder.string(
                        description: "Configurazione di build. Default debug.",
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
