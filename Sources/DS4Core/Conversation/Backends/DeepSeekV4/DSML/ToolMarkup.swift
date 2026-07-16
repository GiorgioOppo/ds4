import Foundation

// MARK: - DSML markup

/// The DeepSeek-V4 tool markup, built on the single special token `｜DSML｜`.
public struct ToolMarkup: Sendable, Equatable {
    /// The DSML special token (vocab token, with fullwidth bars), e.g. "｜DSML｜".
    public var dsml: String
    public init(dsml: String) { self.dsml = dsml }

    public static let dsv4 = ToolMarkup(dsml: "｜DSML｜")

    public var callsOpen: String  { "<\(dsml)tool_calls>" }
    public var callsClose: String { "</\(dsml)tool_calls>" }
    public func invokeOpen(_ name: String) -> String { "<\(dsml)invoke name=\"\(name)\">" }
    public var invokeClose: String { "</\(dsml)invoke>" }
    public func paramOpen(_ name: String, string: Bool) -> String {
        "<\(dsml)parameter name=\"\(name)\" string=\"\(string)\">"
    }
    public var paramClose: String { "</\(dsml)parameter>" }

    /// Confirm the DSML token's exact spelling against the model vocab.
    public static func discover(in tokenizer: Tokenizer) -> ToolMarkup {
        for c in ["｜DSML｜", "|DSML|"] where tokenizer.tokenId(c) != nil { return ToolMarkup(dsml: c) }
        return dsv4
    }
}
