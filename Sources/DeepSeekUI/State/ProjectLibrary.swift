import Foundation
import SwiftUI

/// A user-defined project: a name plus a list of source paths (files
/// or directories). Indexing walks those paths and produces one
/// `VectorizedDocument` per discovered text file, tagged with the
/// project's id so the Projects tab can group them.
///
/// `lastIndexedAt` is nil until the user runs an index; re-indexing
/// nukes every document tagged with this project and re-runs the
/// scan. `modelFingerprint` is recorded at index time so the chat
/// loader (later step) can refuse to use a project whose tokens were
/// produced against a different model.
struct Project: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    /// Absolute filesystem paths. May point at single files or whole
    /// directories — `ProjectIndexer` walks directories recursively.
    var sourcePaths: [String]
    var createdAt: Date
    var lastIndexedAt: Date?
    var modelFingerprint: String?

    init(id: UUID = UUID(),
         name: String,
         sourcePaths: [String] = [],
         createdAt: Date = .now,
         lastIndexedAt: Date? = nil,
         modelFingerprint: String? = nil) {
        self.id = id
        self.name = name
        self.sourcePaths = sourcePaths
        self.createdAt = createdAt
        self.lastIndexedAt = lastIndexedAt
        self.modelFingerprint = modelFingerprint
    }
}

/// Global list of projects + persistence. Documents themselves live in
/// `DocumentLibrary` (tagged with `projectID`); `ProjectLibrary` only
/// owns the project metadata and orchestrates indexing.
@MainActor
final class ProjectLibrary: ObservableObject {
    @Published private(set) var projects: [Project] = []

    init() {
        load()
    }

    func create(name: String) -> Project {
        let p = Project(name: name)
        projects.insert(p, at: 0)
        saveIndex()
        return p
    }

    func update(_ project: Project) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id }) else { return }
        projects[idx] = project
        saveIndex()
    }

    func delete(_ id: UUID, documents: DocumentLibrary) {
        documents.purge(projectID: id)
        projects.removeAll { $0.id == id }
        saveIndex()
    }

    func project(id: UUID) -> Project? {
        projects.first(where: { $0.id == id })
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.projectsIndexURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let entries = try? decoder.decode([Project].self, from: data) {
            projects = entries.sorted { $0.createdAt > $1.createdAt }
        }
    }

    func saveIndex() {
        guard let url = try? PersistencePaths.projectsIndexURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(projects) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
