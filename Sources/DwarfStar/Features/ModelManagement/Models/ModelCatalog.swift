import Foundation
import DS4Engine

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

enum ModelCatalog {
    /// Scan the given directories for runnable catalog models. Custom GGUFs
    /// remain available through Browse, but optional components, split packages
    /// and download-only architectures never appear as a one-click load choice.
    static func scan(directories: [String]) -> [DiscoveredModel] {
        let fm = FileManager.default
        let supportedFiles = Set(
            ModelCatalogRegistry.selectableEntries
                .compactMap { $0.primaryArtifact?.file }
        )
        var seen = Set<String>()
        var out: [DiscoveredModel] = []
        for dir in directories {
            guard let items = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for item in items where item.hasSuffix(".gguf") {
                let full = (dir as NSString).appendingPathComponent(item)
                let resolved = (try? fm.destinationOfSymbolicLink(atPath: full)).map {
                    ($0 as NSString).isAbsolutePath ? $0 : (dir as NSString).appendingPathComponent($0)
                } ?? full
                let resolvedName = (resolved as NSString).lastPathComponent
                guard supportedFiles.contains(item) || supportedFiles.contains(resolvedName) else {
                    continue
                }
                let values = try? URL(fileURLWithPath: resolved).resourceValues(
                    forKeys: [.isRegularFileKey, .fileSizeKey]
                )
                guard values?.isRegularFile == true, let bytes = values?.fileSize, bytes > 0 else {
                    continue
                }
                guard seen.insert(resolved).inserted else { continue }
                let displayName = supportedFiles.contains(item) ? item : resolvedName
                out.append(DiscoveredModel(path: full, name: displayName, sizeBytes: Int64(bytes)))
            }
        }
        return out.sorted { $0.name < $1.name }
    }
}
