import Foundation

/// `xcodebuild test` — build and run a test plan for a scheme.
/// Generates a `.xcresult` bundle that `xcresulttool_get` can parse.
public struct XcodebuildTestTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_test",
            description:
                "Esegue XCTest / Swift Testing di uno scheme dentro un progetto Xcode su un simulator o device. " +
                "Usalo per target di app in cui i test richiedono una destinazione (UI test su simulator iOS, " +
                "test visionOS sul simulator Vision Pro, unit test di app macOS, ecc.). " +
                "Per un pacchetto Swift Package Manager (senza scheme/destination) usa 'swift_test' — più semplice e veloce. " +
                "Passa sempre 'destination' esplicitamente; il default di xcodebuild spesso sceglie una destinazione iOS " +
                "generica che non riesce a buildare. Combinalo con 'xcresulttool_get' sul 'resultBundlePath' prodotto " +
                "per analizzare i fallimenti.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Nome dello scheme."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relativo alla root dell'agente."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relativo alla root dell'agente."),
                    "destination": SchemaBuilder.string(description: "Stringa di destinazione xcodebuild."),
                    "configuration": SchemaBuilder.string(description: "Configurazione di build."),
                    "testPlan": SchemaBuilder.string(description: "Nome del test plan."),
                    "onlyTesting": SchemaBuilder.array(itemsType: "string", description: "Identificatori dei test da eseguire."),
                    "skipTesting": SchemaBuilder.array(itemsType: "string", description: "Identificatori dei test da saltare."),
                    "resultBundlePath": SchemaBuilder.string(description: "Path di output del .xcresult, relativo alla root dell'agente."),
                    "derivedDataPath": SchemaBuilder.string(description: "Directory DerivedData, relativa alla root dell'agente. Default 'build/'."),
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
