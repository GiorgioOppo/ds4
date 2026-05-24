import Foundation

/// `xcodebuild -showsdks` — list every SDK in the active toolchain.
/// Lets the model pick the right `-sdk` for visionOS / iOS / macOS
/// without guessing version suffixes.
public struct XcodebuildShowSdksTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_showsdks",
            description:
                "Elenca ogni SDK installato nell'Xcode attivo (macOS, iOS, iOS Simulator, visionOS, " +
                "visionOS Simulator, watchOS, tvOS, ecc).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "json": SchemaBuilder.boolean(description: "Emette JSON. Default true.", defaultValue: true),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = ["-showsdks"]
        if input.optionalBool("json") ?? true { args.append("-json") }
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory)
    }
}
