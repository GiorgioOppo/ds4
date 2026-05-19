import Foundation

/// `xcodebuild test` — build and run a test plan for a scheme.
/// Generates a `.xcresult` bundle that `xcresulttool_get` can parse.
public struct XcodebuildTestTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_test",
            description:
                "Run XCTest / Swift Testing of a scheme inside an Xcode project on a simulator or device. " +
                "Use this for app targets where the tests need a destination (UI tests on iOS sim, " +
                "visionOS tests on the Vision Pro simulator, macOS app unit tests, etc.). " +
                "For a Swift Package Manager package (no scheme/destination) use 'swift_test' — simpler and faster. " +
                "Always pass 'destination' explicitly; the xcodebuild default frequently picks a generic iOS " +
                "destination that fails to build. Pair with 'xcresulttool_get' on the produced 'resultBundlePath' " +
                "to parse failures.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Scheme name."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relative to agent root."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relative to agent root."),
                    "destination": SchemaBuilder.string(description: "xcodebuild destination string."),
                    "configuration": SchemaBuilder.string(description: "Build configuration."),
                    "testPlan": SchemaBuilder.string(description: "Test plan name."),
                    "onlyTesting": SchemaBuilder.array(itemsType: "string", description: "Test identifiers to run."),
                    "skipTesting": SchemaBuilder.array(itemsType: "string", description: "Test identifiers to skip."),
                    "resultBundlePath": SchemaBuilder.string(description: ".xcresult output path, relative to agent root."),
                    "derivedDataPath": SchemaBuilder.string(description: "DerivedData dir, relative to agent root. Default 'build/'."),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 900.", minimum: 1),
                ],
                required: ["scheme"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcodebuild test \(input["scheme"] as? String ?? "?")"
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
        if let plan = input.optionalString("testPlan") {
            args.append("-testPlan"); args.append(plan)
        }
        for t in input.optionalStringArray("onlyTesting") ?? [] {
            args.append("-only-testing:\(t)")
        }
        for t in input.optionalStringArray("skipTesting") ?? [] {
            args.append("-skip-testing:\(t)")
        }
        if let rbpRel = input.optionalString("resultBundlePath") {
            let rbp = try resolveInsideRoot(rbpRel, context: context)
            args.append("-resultBundlePath"); args.append(rbp.path)
        }
        let ddpRel = input.optionalString("derivedDataPath") ?? "build"
        let ddp = try resolveInsideRoot(ddpRel, context: context)
        args.append("-derivedDataPath"); args.append(ddp.path)
        args.append("test")
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 900)
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: timeout)
    }
}
