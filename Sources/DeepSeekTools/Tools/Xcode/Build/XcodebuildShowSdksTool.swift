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
                "List every SDK installed in the active Xcode (macOS, iOS, iOS Simulator, visionOS, " +
                "visionOS Simulator, watchOS, tvOS, etc).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "json": SchemaBuilder.boolean(description: "Emit JSON. Default true.", defaultValue: true),
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
