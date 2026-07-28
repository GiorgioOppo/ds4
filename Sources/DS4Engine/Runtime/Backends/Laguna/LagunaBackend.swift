import DS4Core
import DS4Metal

/// Registration record for the Laguna S 2.1 (`laguna`) family.
public enum LagunaBackendDefinition {
    public static let supportedArchitecture: ModelArchitectureID = .laguna
    public static let expectedBlockCount = 48

    /// The complete frontend exists (native chat, interleaved reasoning,
    /// tagged tool calls), so the portable capabilities can be advertised
    /// even while the decoder is missing.
    public static let modelCapabilities: ModelCapabilities = [
        .chat, .tools, .reasoning, .mixtureOfExperts,
    ]

    /// No runtime generation capability is advertised until the Laguna Metal
    /// decoder from the reference `laguna-s2.1` branch has been ported and
    /// validated (see `docs/PORTING-GAPS.md`).
    public static let runtimeCapabilities: BackendCapabilities = []

    /// The enablement switch, forwarded from the target-visible gate in
    /// DS4Metal (`LagunaRuntimeGate.enabled`): selector routing, catalog
    /// availability and demo dispatch all key off that single constant.
    public static var runtimeEnabled: Bool { LagunaRuntimeGate.enabled }
}
