import Foundation

/// DEMO: `ToolProvider` che restituisce un sottoinsieme
/// hardcoded dei tool built-in di DeepSeekTools.
///
/// Volutamente non usa `DefaultTools.standard(_:)` per evitare
/// la dipendenza da `PlanStore`. Dimostra il flow
/// `provider → discover → ToolRegistry.register`.
public final class DemoBuiltinToolProvider: ToolProviderBase,
                                             @unchecked Sendable {
    public init() {
        super.init(providerName: "demo.builtin")
    }

    public override func discover() async throws -> [any Tool] {
        [ReadTool(), WriteTool(), EditTool()]
    }
}
