import Foundation

/// Shared lookup policy for runtime environment knobs.
///
/// Backend-specific names are treated as overrides so existing launch
/// configurations keep their exact behavior. A backend-agnostic canonical
/// name is used only when none of the more specific names is present.
public enum DS4RuntimeEnvironment {
    public static func value(
        _ canonical: String,
        overrides: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        for name in overrides {
            if let value = environment[name] { return value }
        }
        return environment[canonical]
    }

    public static func integer(
        _ canonical: String,
        overrides: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        value(canonical, overrides: overrides, environment: environment)
            .flatMap(Int.init)
    }

    /// Boolean convention used by the engines: `0` is false and `1` is
    /// true. Other present values preserve the historical "not zero"
    /// behavior used by default-on knobs.
    public static func flag(
        _ canonical: String,
        overrides: [String] = [],
        default defaultValue: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = value(
            canonical, overrides: overrides, environment: environment
        ) else {
            return defaultValue
        }
        return raw != "0"
    }
}
