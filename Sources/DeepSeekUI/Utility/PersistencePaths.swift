import Foundation

/// Where the app writes its on-disk state. One JSON file per
/// conversation under
/// `~/Library/Application Support/DeepSeek-V4-Pro-MacOS/conversations/`.
enum PersistencePaths {
    static let appName = "DeepSeek-V4-Pro-MacOS"

    /// Creates the directory tree on first use; failures bubble up.
    static func conversationsDir() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent(appName, isDirectory: true)
            .appendingPathComponent("conversations", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func conversationURL(id: UUID) throws -> URL {
        try conversationsDir()
            .appendingPathComponent("\(id.uuidString).json")
    }
}
