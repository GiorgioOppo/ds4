import SwiftUI

/// Minimal "create new project" modal. Sources and indexing happen in
/// `ProjectDetailView` once the project has been created — this sheet
/// just captures the name.
struct CreateProjectSheet: View {
    @ObservedObject var library: ProjectLibrary
    let onCreated: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("New project")
                .font(.title3.bold())
            TextField("Project name", text: $name)
                .textFieldStyle(.roundedBorder)
            Spacer()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Create") {
                    let trimmed = name.trimmingCharacters(in: .whitespaces)
                    guard !trimmed.isEmpty else { return }
                    let p = library.create(name: trimmed)
                    onCreated(p.id)
                    dismiss()
                }
                .keyboardShortcut(.return, modifiers: [])
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 360, height: 160)
    }
}
