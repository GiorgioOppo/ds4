import Foundation

/// Theme descriptors. Lives in `DeepSeekTools` (not the UI target)
/// because future headless clients also need to know "what tint
/// does the user prefer for the model's brain disclosure" without
/// importing SwiftUI. The actual `Color` mapping happens in the UI
/// layer (`AgentTint`-style switch).
public struct Theme: Codable, Sendable, Identifiable, Hashable {
    public var id: String { name }
    public var name: String
    public var summary: String
    /// Hex strings, `#RRGGBB`. Empty when the system default applies.
    public var accent: String
    public var background: String
    public var foreground: String
    public var assistantBubble: String
    public var userBubble: String
    public var appearance: Appearance

    public enum Appearance: String, Codable, Sendable {
        case system, light, dark
    }

    public init(name: String,
                summary: String = "",
                accent: String = "",
                background: String = "",
                foreground: String = "",
                assistantBubble: String = "",
                userBubble: String = "",
                appearance: Appearance = .system) {
        self.name = name
        self.summary = summary
        self.accent = accent
        self.background = background
        self.foreground = foreground
        self.assistantBubble = assistantBubble
        self.userBubble = userBubble
        self.appearance = appearance
    }
}

public enum BuiltInThemes {
    public static let all: [Theme] = [system, dimmed, sunset, terminal]

    public static let system = Theme(
        name: "System",
        summary: "Follow macOS appearance.",
        appearance: .system
    )

    public static let dimmed = Theme(
        name: "Dimmed",
        summary: "Low-contrast dark with a violet accent.",
        accent: "#9B8CFF",
        background: "#181821",
        foreground: "#E6E6F0",
        assistantBubble: "#23232E",
        userBubble: "#2F2F3F",
        appearance: .dark
    )

    public static let sunset = Theme(
        name: "Sunset",
        summary: "Warm light theme with orange accents.",
        accent: "#E66F2D",
        background: "#FFF8F1",
        foreground: "#2A1F1A",
        assistantBubble: "#FFEDD8",
        userBubble: "#FFE0BF",
        appearance: .light
    )

    public static let terminal = Theme(
        name: "Terminal",
        summary: "Monochrome green-on-black, fixed-pitch vibe.",
        accent: "#00FF80",
        background: "#0A0F0A",
        foreground: "#A8FFC8",
        assistantBubble: "#0F1A0F",
        userBubble: "#1A2A1A",
        appearance: .dark
    )
}
