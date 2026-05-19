import Foundation

/// `xcodebuild build` — compile a scheme for a destination. Default
/// timeout 600 s because a cold build on a real project routinely
/// takes minutes.
public struct XcodebuildBuildTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_build",
            description:
                "Build an Xcode scheme. Provide 'scheme' (required), 'workspace' OR 'project', " +
                "and optionally 'destination' (e.g. 'platform=iOS Simulator,name=iPhone 15'), " +
                "'configuration' (Debug/Release), 'sdk', 'derivedDataPath'. " +
                "Set 'noCodesign=true' to disable code signing (useful for simulator/CI builds).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Scheme name (from xcodebuild_list)."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relative to agent root."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relative to agent root."),
                    "destination": SchemaBuilder.string(description: "xcodebuild destination string."),
                    "configuration": SchemaBuilder.string(description: "Build configuration (e.g. Debug, Release)."),
                    "sdk": SchemaBuilder.string(description: "SDK name (e.g. iphonesimulator, macosx, xrsimulator)."),
                    "derivedDataPath": SchemaBuilder.string(description: "DerivedData dir, relative to agent root. Default 'build/'."),
                    "noCodesign": SchemaBuilder.boolean(description: "Disable code signing. Default false.", defaultValue: false),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 600.", minimum: 1),
                ],
                required: ["scheme"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcodebuild build \(input["scheme"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let scheme = try input.string("scheme")
        var args: [String] = ["-scheme", scheme]
        if let ws = input.optionalString("workspace") {
            let url = try resolveInsideRoot(ws, context: context)
            args.append("-workspace"); args.append(url.path)
        } else if let proj = input.optionalString("project") {
            let url = try resolveInsideRoot(proj, context: context)
            args.append("-project"); args.append(url.path)
        }
        if let dest = input.optionalString("destination") {
            args.append("-destination"); args.append(dest)
        }
        if let cfg = input.optionalString("configuration") {
            args.append("-configuration"); args.append(cfg)
        }
        if let sdk = input.optionalString("sdk") {
            args.append("-sdk"); args.append(sdk)
        }
        let ddpRel = input.optionalString("derivedDataPath") ?? "build"
        let ddp = try resolveInsideRoot(ddpRel, context: context)
        args.append("-derivedDataPath"); args.append(ddp.path)
        if input.optionalBool("noCodesign") ?? false {
            args.append("CODE_SIGNING_ALLOWED=NO")
            args.append("CODE_SIGNING_REQUIRED=NO")
            args.append("CODE_SIGN_IDENTITY=")
        }
        args.append("build")
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 600)
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: timeout)
    }
}
