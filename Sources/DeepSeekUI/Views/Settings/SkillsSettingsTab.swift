import SwiftUI
import DeepSeekTools

/// Skill registry editor. Built-in skills are read-only (greyed
/// pencil); user skills can be added, renamed, and deleted. Future:
/// inline edit sheet for per-skill prompt + tool allowlist. For now
/// the editor is purely list-level so the SkillLibrary surface is
/// usable from `/skill`.
struct SkillsSettingsTab: View {
    @ObservedObject var library: SkillLibrary
    @State private var selection: UUID?

    private var builtinIDs: Set<UUID> { Set(BuiltInSkills.all.map(\.id)) }

    var body: some View {
        HStack(spacing: 0) {
            List(selection: $selection) {
                ForEach(library.skills) { skill in
                    SkillRowView(skill: skill, isBuiltIn: builtinIDs.contains(skill.id))
                        .tag(skill.id as UUID?)
                }
            }
            .frame(minWidth: 200)

            Divider()

            if let id = selection, let skill = library.skills.first(where: { $0.id == id }) {
                SkillDetailView(skill: skill, isBuiltIn: builtinIDs.contains(skill.id))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Text("Pick a skill on the left.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

private struct SkillRowView: View {
    let skill: Skill
    let isBuiltIn: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(skill.name).font(.headline)
                if isBuiltIn {
                    Image(systemName: "lock.fill")
                        .foregroundStyle(.secondary)
                        .help("Built-in skill")
                }
            }
            Text(skill.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }
}

private struct SkillDetailView: View {
    let skill: Skill
    let isBuiltIn: Bool

    var body: some View {
        Form {
            Section("Identity") {
                LabeledContent("Name", value: skill.name)
                if !skill.summary.isEmpty {
                    LabeledContent("Summary", value: skill.summary)
                }
            }
            Section("System prompt addendum") {
                if skill.systemPromptAddendum.isEmpty {
                    Text("(none)").foregroundStyle(.secondary)
                } else {
                    Text(skill.systemPromptAddendum)
                        .font(.callout)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            Section("Allowed tools") {
                if let names = skill.allowedToolNames {
                    if names.isEmpty {
                        Text("No tools (chat only).")
                    } else {
                        ForEach(Array(names).sorted(), id: \.self) { Text($0).font(.body.monospaced()) }
                    }
                } else {
                    Text("All registered tools.")
                }
            }
            Section("Sampling override") {
                if let t = skill.temperature {
                    LabeledContent("Temperature", value: String(format: "%.2f", t))
                }
                if let m = skill.maxTokens {
                    LabeledContent("Max tokens", value: "\(m)")
                }
                if skill.temperature == nil && skill.maxTokens == nil {
                    Text("Inherits from the active agent.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }
}
