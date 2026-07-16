import Foundation

/// Runtime features that callers may safely expose for the selected backend.
///
/// These flags describe implemented engine paths, not only properties declared
/// by a GGUF. Keeping that distinction prevents a recognized Qwen file from
/// accidentally enabling DeepSeek-only cache, MoE or distributed controls.
public struct BackendCapabilities: OptionSet, Sendable, Hashable, Codable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) { self.rawValue = rawValue }

    public static let generation = BackendCapabilities(rawValue: 1 << 0)
    public static let reasoning = BackendCapabilities(rawValue: 1 << 1)
    public static let tools = BackendCapabilities(rawValue: 1 << 2)
    public static let diskKV = BackendCapabilities(rawValue: 1 << 3)
    public static let expertRouting = BackendCapabilities(rawValue: 1 << 4)
    public static let expertBundle = BackendCapabilities(rawValue: 1 << 5)
    public static let deepSeekPerformanceTuning = BackendCapabilities(rawValue: 1 << 6)
    public static let distributedPipeline = BackendCapabilities(rawValue: 1 << 7)
    public static let expertParallel = BackendCapabilities(rawValue: 1 << 8)
}
