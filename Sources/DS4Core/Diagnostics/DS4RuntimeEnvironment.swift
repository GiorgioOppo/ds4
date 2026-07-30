import Foundation

/// Shared lookup policy for runtime environment knobs.
///
/// The backend-agnostic name is authoritative. Backend-specific historical
/// names are accepted only as compatibility fallbacks when the canonical
/// name is absent.
public enum DS4RuntimeEnvironment {
    public static let schemaVersion = 1

    public static func value(
        _ canonical: String,
        overrides: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let value = environment[canonical] { return value }
        for name in overrides {
            if let value = environment[name] { return value }
        }
        return nil
    }

    public static func integer(
        _ canonical: String,
        overrides: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        value(canonical, overrides: overrides, environment: environment)
            .flatMap(Int.init)
    }

    public static func value(
        _ knob: DS4RuntimeKnob,
        backend: DS4RuntimeBackend,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        value(
            knob.rawValue,
            overrides: knob.deprecatedAliases(for: backend),
            environment: environment)
    }

    public static func integer(
        _ knob: DS4RuntimeKnob,
        backend: DS4RuntimeBackend,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        value(knob, backend: backend, environment: environment)
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

    public static func flag(
        _ knob: DS4RuntimeKnob,
        backend: DS4RuntimeBackend,
        default defaultValue: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = value(
            knob, backend: backend, environment: environment
        ) else {
            return defaultValue
        }
        return raw != "0"
    }
}
