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
                "Compila uno scheme da un progetto Xcode (.xcodeproj) o workspace (.xcworkspace). " +
                "Usalo per target di app macOS / iOS / iPadOS / visionOS / watchOS / tvOS e per ogni " +
                "build che richieda una specifica destinazione simulator/device. " +
                "Per un pacchetto puro Swift Package Manager (Package.swift, senza .xcodeproj) usa invece 'swift_build'. " +
                "Prima di chiamarlo, usa 'xcodebuild_list' per scoprire gli scheme e 'xcodebuild_showdestinations' " +
                "per scegliere una stringa di destinazione valida. " +
                "'destination' = 'platform=iOS Simulator,name=iPhone 15' ecc.; " +
                "'noCodesign=true' disabilita la firma (tipico per build simulator/CI).",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Nome dello scheme (da xcodebuild_list)."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relativo alla root dell'agente."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relativo alla root dell'agente."),
                    "destination": SchemaBuilder.string(description: "Stringa di destinazione xcodebuild."),
                    "configuration": SchemaBuilder.string(description: "Configurazione di build (es. Debug, Release)."),
                    "sdk": SchemaBuilder.string(description: "Nome dell'SDK (es. iphonesimulator, macosx, xrsimulator)."),
                    "derivedDataPath": SchemaBuilder.string(description: "Directory DerivedData, relativa alla root dell'agente. Default 'build/'."),
                    "noCodesign": SchemaBuilder.boolean(description: "Disabilita la firma del codice. Default false.", defaultValue: false),
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
