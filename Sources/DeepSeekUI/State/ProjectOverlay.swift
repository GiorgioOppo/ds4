import Foundation
import DeepSeekTools

/// Per-project overlay loader (TODO §11). When the active chat has
/// a `Project` attached, callers can pass the project's primary
/// source path here to read agent / skill / slash-command overrides
/// from `<root>/.deepseek/{agents,skills,slash}.json`. The on-disk
/// shapes match what `AgentLibrary` / `SkillLibrary` /
/// `SlashCommandLibrary` already persist globally, so a user can
/// drop their global config into a project directory verbatim.
///
/// The overlay is read-only: edits go through the normal
/// (global) library APIs. Project-local items are merged on read
/// by `effective…(in:)` helpers that union the overlay over the
/// global set with project-local taking precedence on name /
/// trigger collisions.
///
/// Pattern matches `.opencode/` and `CLAUDE.md` overlays — anything
/// in the repo travels with the repo without requiring the user to
/// sync `~/Library/Application Support` between machines.
struct ProjectOverlay: Sendable {
    let agents: [AgentConfig]
    let skills: [Skill]
    let slashCommands: [SlashCommand]
    let rootDirectory: URL

    init(agents: [AgentConfig] = [],
         skills: [Skill] = [],
         slashCommands: [SlashCommand] = [],
         rootDirectory: URL)
    {
        self.agents = agents
        self.skills = skills
        self.slashCommands = slashCommands
        self.rootDirectory = rootDirectory
    }

    /// Empty overlay rooted at the given URL. Used when the
    /// project doesn't carry a `.deepseek/` directory — callers
    /// still get a stable handle they can ask for overrides
    /// from (yielding the global library each time).
    static func empty(rootDirectory: URL) -> ProjectOverlay {
        ProjectOverlay(rootDirectory: rootDirectory)
    }
}

enum ProjectOverlayLoader {
    /// Read a project's overlay (if present). Missing files are
    /// silently ignored — projects opt in by creating the
    /// matching JSON. Malformed files emit a stderr note and
    /// fall back to empty.
    static func load(rootDirectory: URL) -> ProjectOverlay {
        let dotDir = rootDirectory.appendingPathComponent(".deepseek")
        let fm = FileManager.default
        guard fm.fileExists(atPath: dotDir.path) else {
            return .empty(rootDirectory: rootDirectory)
        }
        let agents = decodeArray(
            AgentConfig.self,
            at: dotDir.appendingPathComponent("agents.json"))
        let skills = decodeArray(
            Skill.self,
            at: dotDir.appendingPathComponent("skills.json"))
        let slash = decodeArray(
            SlashCommand.self,
            at: dotDir.appendingPathComponent("slash.json"))
        return ProjectOverlay(
            agents: agents,
            skills: skills,
            slashCommands: slash,
            rootDirectory: rootDirectory)
    }

    private static func decodeArray<T: Codable>(
        _ type: T.Type, at url: URL) -> [T]
    {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode([T].self, from: data)
        } catch {
            let message = "[ProjectOverlay] failed to decode "
                + "\(url.lastPathComponent): \(error)\n"
            FileHandle.standardError.write(Data(message.utf8))
            return []
        }
    }

    /// Merge an overlay's agent list with the global library's
    /// snapshot. Project-local entries with the SAME NAME as a
    /// global agent take precedence (override semantics); entries
    /// with unique names are appended. Returns the merged array
    /// in stable order: global first, then any new project-local
    /// additions.
    static func mergeAgents(
        global: [AgentConfig],
        overlay: [AgentConfig]) -> [AgentConfig]
    {
        var byName: [String: AgentConfig] = [:]
        for a in global { byName[a.name] = a }
        for a in overlay { byName[a.name] = a }  // override
        // Preserve original order for unchanged entries; append
        // anything new the overlay introduced.
        let originalOrder = global.map(\.name)
        var seen = Set(originalOrder)
        var out: [AgentConfig] = originalOrder.compactMap { byName[$0] }
        for a in overlay where !seen.contains(a.name) {
            out.append(a)
            seen.insert(a.name)
        }
        return out
    }

    /// Merge for skills. Same precedence as agents — overlay
    /// wins on name collision.
    static func mergeSkills(
        global: [Skill],
        overlay: [Skill]) -> [Skill]
    {
        var byName: [String: Skill] = [:]
        for s in global { byName[s.name] = s }
        for s in overlay { byName[s.name] = s }
        let originalOrder = global.map(\.name)
        var seen = Set(originalOrder)
        var out: [Skill] = originalOrder.compactMap { byName[$0] }
        for s in overlay where !seen.contains(s.name) {
            out.append(s)
            seen.insert(s.name)
        }
        return out
    }

    /// Merge for slash commands. Match key is `name` — that's the
    /// palette label and also (today) the `/foo` trigger.
    static func mergeSlashCommands(
        global: [SlashCommand],
        overlay: [SlashCommand]) -> [SlashCommand]
    {
        var byName: [String: SlashCommand] = [:]
        for c in global { byName[c.name] = c }
        for c in overlay { byName[c.name] = c }
        let originalOrder = global.map(\.name)
        var seen = Set(originalOrder)
        var out: [SlashCommand] = originalOrder.compactMap { byName[$0] }
        for c in overlay where !seen.contains(c.name) {
            out.append(c)
            seen.insert(c.name)
        }
        return out
    }
}
