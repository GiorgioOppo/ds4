import SwiftUI

/// "Create new project" modal. Captures the name, the import
/// strategy, and — when the strategy is `.gitClone` — the upstream
/// URL and optional branch. The actual `sourcePaths` for symlink /
/// copy modes are still picked from `ProjectDetailView` after
/// creation, so this sheet stays under a screen worth of fields
/// regardless of strategy.
struct CreateProjectSheet: View {
    @ObservedObject var library: ProjectLibrary
    let onCreated: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name: String = ""
    @State private var strategyID: String = "copy"
    @State private var gitURL: String = ""
    @State private var gitBranch: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("New project")
                .font(.title3.bold())

            TextField("Project name", text: $name)
                .textFieldStyle(.roundedBorder)

            strategyPicker

            if strategyID == "git" {
                gitFields
            }

            Text(strategyHint)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: 4)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Create") { create() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(!canCreate)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 420, height: strategyID == "git" ? 360 : 280)
    }

    private var strategyPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Import strategy").font(.subheadline.bold())
            Picker("Import strategy", selection: $strategyID) {
                Text("Copy files").tag("copy")
                Text("Symlink farm (advanced)").tag("symlink")
                Text("Clone from Git").tag("git")
            }
            .labelsHidden()
            .pickerStyle(.segmented)
        }
    }

    @ViewBuilder
    private var gitFields: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField("https://github.com/user/repo.git", text: $gitURL)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
            TextField("Branch (optional)", text: $gitBranch)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
        }
    }

    private var strategyHint: String {
        switch strategyID {
        case "copy":
            return "Copies bytes from your source folders into the "
                + "project. The agent's tools read and write the "
                + "copy, never your original files. The farm is "
                + "initialised as a Git repo so you can diff and "
                + "roll back changes."
        case "symlink":
            return "Symlinks each file in your source folders into "
                + "the project. Live updates from your real files; "
                + "the sandbox needs a security-scoped bookmark for "
                + "every read, and agent edits land on your original "
                + "files."
        case "git":
            return "Shallow-clones the remote into the project. "
                + "Self-contained; \"Pull\" updates from upstream. "
                + "Requires the app to be allowed to spawn `git` — "
                + "App Store builds run with sandbox restrictions "
                + "that block this."
        default:
            return ""
        }
    }

    private var canCreate: Bool {
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty
        else { return false }
        if strategyID == "git" {
            return !gitURL.trimmingCharacters(in: .whitespaces).isEmpty
        }
        return true
    }

    private func create() {
        let trimmedName = name.trimmingCharacters(in: .whitespaces)
        guard !trimmedName.isEmpty else { return }
        let strategy: ProjectImportStrategy
        switch strategyID {
        case "symlink":
            strategy = .symlinkFarm
        case "git":
            let url = gitURL.trimmingCharacters(in: .whitespaces)
            let branch = gitBranch.trimmingCharacters(in: .whitespaces)
            guard !url.isEmpty else { return }
            strategy = .gitClone(
                repoURL: url, branch: branch.isEmpty ? nil : branch)
        default:
            strategy = .copy
        }
        let p = library.create(name: trimmedName, strategy: strategy)
        onCreated(p.id)
        dismiss()
    }
}
