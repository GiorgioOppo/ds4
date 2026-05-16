import Foundation
import SwiftUI
import DeepSeekTools

/// In-memory registry of skills: the bundled `BuiltInSkills` plus any
/// user-defined ones from `skills.json`. Behaves like
/// `AgentLibrary`: published list of `Skill`, CRUD, file-backed.
@MainActor
final class SkillLibrary: ObservableObject {
    @Published private(set) var skills: [Skill] = []

    init() { load() }

    func add(_ skill: Skill) {
        skills.append(skill); save()
    }

    func update(_ skill: Skill) {
        guard let idx = skills.firstIndex(where: { $0.id == skill.id }) else { return }
        skills[idx] = skill; save()
    }

    func delete(_ id: UUID) {
        if BuiltInSkills.all.contains(where: { $0.id == id }) { return }
        skills.removeAll { $0.id == id }; save()
    }

    /// Find skill by registered name (case-insensitive). Used by
    /// `/skill <name>`.
    func skill(matching name: String) -> Skill? {
        skills.first { $0.name.lowercased() == name.lowercased() }
    }

    // MARK: - persistence

    private func load() {
        // Always seed with built-ins; user-defined entries are merged
        // on top, last-write-wins by id.
        var merged: [UUID: Skill] = [:]
        for s in BuiltInSkills.all { merged[s.id] = s }
        if let url = try? PersistencePaths.skillsConfigURL(),
           let data = try? Data(contentsOf: url),
           let user = try? JSONDecoder().decode([Skill].self, from: data) {
            for s in user { merged[s.id] = s }
        }
        skills = merged.values.sorted { $0.createdAt < $1.createdAt }
    }

    private func save() {
        // Persist only entries that differ from the built-in defaults.
        let builtinIDs = Set(BuiltInSkills.all.map(\.id))
        let user = skills.filter { !builtinIDs.contains($0.id) }
        guard let url = try? PersistencePaths.skillsConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(user) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
