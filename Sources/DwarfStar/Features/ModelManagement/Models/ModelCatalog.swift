import Foundation

/// A GGUF file discovered on disk.
struct DiscoveredModel: Identifiable, Hashable {
    let path: String
    let name: String
    let sizeBytes: Int64

    var id: String { path }
    var displaySize: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

/// A `download_model.sh` target the user can fetch.
struct DownloadTarget: Identifiable, Hashable {
    let id: String      // the argument passed to download_model.sh
    let title: String
    let detail: String
}

enum ModelCatalog {
    /// Scan the given directories for *.gguf files (skips partial *.part).
    static func scan(directories: [String]) -> [DiscoveredModel] {
        let fm = FileManager.default
        var seen = Set<String>()
        var out: [DiscoveredModel] = []
        for dir in directories {
            guard let items = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for item in items where item.hasSuffix(".gguf") {
                let full = (dir as NSString).appendingPathComponent(item)
                let resolved = (try? fm.destinationOfSymbolicLink(atPath: full)).map {
                    ($0 as NSString).isAbsolutePath ? $0 : (dir as NSString).appendingPathComponent($0)
                } ?? full
                guard seen.insert(resolved).inserted else { continue }
                let size = ((try? fm.attributesOfItem(atPath: full))?[.size] as? NSNumber)?.int64Value ?? 0
                out.append(DiscoveredModel(path: full, name: item, sizeBytes: size))
            }
        }
        return out.sorted { $0.name < $1.name }
    }

    /// The download targets exposed by download_model.sh (see README).
    static let downloadTargets: [DownloadTarget] = [
        .init(id: "q2-imatrix", title: "Flash q2 (imatrix)", detail: "96/128 GB RAM"),
        .init(id: "q2-q4-imatrix", title: "Flash q2 + last 6 layers q4", detail: "96/128 GB RAM"),
        .init(id: "q4-imatrix", title: "Flash q4 (imatrix)", detail: ">= 256 GB RAM"),
        .init(id: "pro-q2-imatrix", title: "PRO q2 (imatrix)", detail: "512 GB / 128 GB streaming"),
        .init(id: "mtp", title: "MTP speculative (Flash)", detail: "optional, use with --mtp"),
    ]
}
