import Foundation

/// One-shot, model-readable summary of a repository: top-level
/// directory tree (depth-limited), language breakdown by extension,
/// and the contents of conventional metadata files (README, package
/// manifests). Useful as a first turn when the model has just been
/// pointed at an unfamiliar repo.
public struct RepoOverviewTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "repo_overview",
            description:
                "Riepilogo testuale compatto di un repository: albero (limitato in profondità), " +
                "conteggio dei file per estensione, contenuti del README e dei principali " +
                "manifest di package. Sola lettura.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Root del repository, relativa alla root dell'agente. Default: '.'."),
                    "maxDepth": SchemaBuilder.integer(description: "Profondità massima dell'albero. Default 3.", minimum: 1),
                ],
                required: []
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "repo_overview \(input["path"] as? String ?? ".")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let relPath = input.optionalString("path") ?? "."
        let maxDepth = max(1, input.optionalInteger("maxDepth") ?? 3)
        let root = try resolveInsideRoot(relPath, context: context)
        var sections: [String] = []

        // 1. Tree.
        var tree: [String] = []
        try buildTree(at: root, prefix: "", depth: 0, maxDepth: maxDepth, lines: &tree)
        sections.append("# Tree (depth ≤ \(maxDepth))\n" + tree.joined(separator: "\n"))

        // 2. Extension histogram.
        var extCounts: [String: Int] = [:]
        let walker = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        while let next = walker?.nextObject() as? URL {
            if context.isCancelled() { break }
            let isFile = (try? next.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            guard isFile else { continue }
            let ext = next.pathExtension.lowercased()
            let key = ext.isEmpty ? "(no extension)" : ext
            extCounts[key, default: 0] += 1
        }
        let topExt = extCounts
            .sorted { $0.value > $1.value }
            .prefix(15)
            .map { "  \($0.key): \($0.value)" }
            .joined(separator: "\n")
        sections.append("# File counts by extension\n\(topExt)")

        // 3. Manifests & README. Best-effort by filename match.
        let probeNames: [String] = [
            "README.md", "README", "README.txt", "Package.swift",
            "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
            "Gemfile", "build.gradle", "pom.xml", "Makefile",
        ]
        for name in probeNames {
            let candidate = root.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: candidate.path),
               let data = try? Data(contentsOf: candidate),
               let text = String(data: data, encoding: .utf8) {
                let preview = text.count > 4000 ? String(text.prefix(4000)) + "\n…[truncated]" : text
                sections.append("# \(name)\n\(preview)")
            }
        }
        return ToolOutput(
            output: sections.joined(separator: "\n\n"),
            metadata: ["files-by-ext": "\(extCounts.values.reduce(0, +))"]
        )
    }

    private func buildTree(at url: URL,
                           prefix: String,
                           depth: Int,
                           maxDepth: Int,
                           lines: inout [String]) throws {
        if depth == 0 {
            lines.append(url.lastPathComponent + "/")
        }
        guard depth < maxDepth else {
            lines.append(prefix + "  …")
            return
        }
        let children = (try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]))?
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) ?? []
        for child in children.prefix(40) {
            let isDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            lines.append("\(prefix)  \(child.lastPathComponent)\(isDir ? "/" : "")")
            if isDir {
                try buildTree(at: child, prefix: prefix + "  ",
                              depth: depth + 1, maxDepth: maxDepth, lines: &lines)
            }
        }
        if children.count > 40 {
            lines.append("\(prefix)  …\(children.count - 40) more")
        }
    }
}
