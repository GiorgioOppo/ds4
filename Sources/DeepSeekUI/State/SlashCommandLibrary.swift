import Foundation
import SwiftUI
import DeepSeekTools

/// Registry of slash commands the composer can dispatch. Bundles
/// `BuiltInSlashCommands` with any user-defined commands loaded
/// from `~/Library/Application Support/.../commands/*.json`.
///
/// The host (`ChatView` or `InferenceService`) calls
/// `parse(line:)` on the composer text before any model dispatch; if
/// a command matches, the host invokes `dispatch` which returns a
/// `SlashCommandEffect` describing what the UI should do. The effect
/// switch happens in the view layer so commands stay declarative
/// here.
@MainActor
final class SlashCommandLibrary: ObservableObject {
    @Published private(set) var commands: [SlashCommand] = []

    init() { load() }

    /// Try to interpret `text` as a slash command. Returns `nil` if
    /// it isn't one, or if the parsed verb doesn't match any
    /// registered command. The caller should fall through to normal
    /// chat dispatch on `nil`.
    func parse(line text: String) -> Dispatch? {
        guard let parsed = ParsedSlashCommand.parse(text) else { return nil }
        guard let command = commands.first(where: { $0.name == parsed.name }) else {
            return Dispatch(command: nil, raw: parsed)
        }
        return Dispatch(command: command, raw: parsed)
    }

    struct Dispatch {
        let command: SlashCommand?
        let raw: ParsedSlashCommand
    }

    private func load() {
        var merged: [String: SlashCommand] = Dictionary(
            uniqueKeysWithValues: BuiltInSlashCommands.all.map { ($0.name, $0) })
        if let dir = try? PersistencePaths.slashCommandsDir(),
           let entries = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) {
            for entry in entries where entry.pathExtension == "json" {
                guard let data = try? Data(contentsOf: entry),
                      let cmd = try? JSONDecoder().decode(SlashCommand.self, from: data) else {
                    continue
                }
                merged[cmd.name] = cmd
            }
        }
        commands = merged.values.sorted { $0.name < $1.name }
    }
}

/// What the UI does in response to a matched slash command. Mapped
/// from `SlashCommand.action` by the host. Keeping this enum here
/// (not in `DeepSeekTools`) so it can refer to UI-only intentions
/// like "open Settings".
enum SlashCommandEffect {
    case showHelp
    case clearConversation
    case newConversation
    case switchMode(AgentMode)
    case showModelPicker
    case showAgentPicker
    case activateSkill(name: String)
    case listTools
    case showPermissions
    case switchTheme(name: String)
    case compactContext
    case echo(String)
}
