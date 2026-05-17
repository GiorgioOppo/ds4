import Foundation

/// Modalità di contesto progetto: come il progetto viene
/// presentato al modello sulla prima turn di chat.
///
/// - `pathsOnly`: solo l'albero gerarchico dei path. Il modello
///   usa i tool `read`/`glob`/`grep`/`repo_overview` per esplorare
///   on-demand. Default per i nuovi progetti.
/// - `indexedContent`: il vecchio comportamento — i token
///   pre-calcolati di ogni documento indicizzato vengono iniettati
///   direttamente nello stream del prompt (con marker
///   `<｜begin▁of▁file▁…｜>`). Mantenuto come opt-in per chi vuole
///   il time-to-first-token ridotto.
enum ProjectContextMode: String, Codable, Sendable, CaseIterable {
    case pathsOnly
    case indexedContent

    var displayName: String {
        switch self {
        case .pathsOnly:      return "Paths only (lazy tools)"
        case .indexedContent: return "Indexed content (eager)"
        }
    }

    var summary: String {
        switch self {
        case .pathsOnly:
            return "Inject only file paths as a tree. The model uses " +
                   "`read`/`glob`/`grep` to explore on demand."
        case .indexedContent:
            return "Inject pre-tokenized file contents on the first turn. " +
                   "Faster time-to-first-answer for small projects but uses " +
                   "context window proportional to the indexed size."
        }
    }
}

/// Singola entry dell'inventario: path relativo (con prefisso
/// root display name), dimensione, mtime opzionale.
struct ProjectFileEntry: Sendable, Equatable {
    let relativePath: String
    let byteCount: Int
    let lastModified: Date?

    init(relativePath: String,
         byteCount: Int,
         lastModified: Date? = nil) {
        self.relativePath = relativePath
        self.byteCount = byteCount
        self.lastModified = lastModified
    }
}

/// Snapshot strutturato di un progetto pronto per essere
/// renderizzato in markdown nel system prompt. Costruito al
/// volo da `ProjectInventoryBuilder`; non persistito.
struct ProjectInventory: Sendable {
    let projectName: String
    let entries: [ProjectFileEntry]
    /// True se `entries.count < totalDiscovered` perché la cap
    /// per-progetto ha troncato la lista.
    let truncated: Bool
    let totalDiscovered: Int
    /// Le directory radice di cui è composto il progetto, per la
    /// nota finale al modello su dove guardare con `glob`.
    let rootDisplayNames: [String]

    init(projectName: String,
         entries: [ProjectFileEntry],
         truncated: Bool,
         totalDiscovered: Int,
         rootDisplayNames: [String]) {
        self.projectName = projectName
        self.entries = entries
        self.truncated = truncated
        self.totalDiscovered = totalDiscovered
        self.rootDisplayNames = rootDisplayNames
    }
}

/// Trie interno usato dal renderer dell'albero. File-private per
/// non inquinare l'API pubblica.
fileprivate final class _TreeNode {
    var children: [String: _TreeNode] = [:]
    var isFile: Bool = false
    /// Ordine di prima apparizione, per stabilità di output.
    var childOrder: [String] = []
}

extension ProjectInventory {

    /// Renderizza l'inventario come blocco markdown da prependere
    /// al system prompt. Pattern:
    ///
    ///     ## Project: <name>
    ///
    ///     <N> files (<+M truncated>). Use `read`, `glob`, `grep`,
    ///     `repo_overview` to explore content on demand.
    ///
    ///     ```
    ///     RepoName/
    ///     ├── Sources/
    ///     │   └── …
    ///     └── README.md
    ///     ```
    ///
    /// `maxDepth` taglia i livelli più profondi e li sostituisce
    /// con `…/` sotto il parent. Default 3 (allineato con
    /// `repo_overview` tool).
    func renderTree(maxDepth: Int = 3) -> String {
        var out = "## Project: \(projectName)\n\n"

        let countLine: String
        if truncated {
            countLine = "\(entries.count) files listed " +
                "(\(totalDiscovered - entries.count) more truncated)."
        } else {
            countLine = "\(entries.count) file" +
                (entries.count == 1 ? "" : "s") + "."
        }
        out += countLine
        out += " Use `read`, `glob`, `grep`, `repo_overview` " +
               "to fetch content on demand.\n\n"

        if entries.isEmpty {
            return out
        }

        out += "```\n"
        out += Self.renderBoxTree(
            entries: entries.map { $0.relativePath },
            maxDepth: maxDepth)
        out += "\n```\n"

        if truncated {
            out += "\n_Tree truncated. " +
                   "Run `glob` with a specific pattern to enumerate " +
                   "more files._\n"
        }

        return out
    }

    /// Costruisce un trie da una lista di path relativi con
    /// separatore `/`, lo rende con box-drawing characters
    /// (`├── ` / `└── ` / `│   ` / quattro spazi).
    /// Path che superano `maxDepth` vengono collassati: il parent
    /// mostra `…/` come unico figlio.
    static func renderBoxTree(entries: [String], maxDepth: Int) -> String {
        let root = _TreeNode()
        for path in entries {
            let parts = path.split(separator: "/").map(String.init)
            var current = root
            for (idx, part) in parts.enumerated() {
                if current.children[part] == nil {
                    let n = _TreeNode()
                    current.children[part] = n
                    current.childOrder.append(part)
                }
                let next = current.children[part]!
                if idx == parts.count - 1 {
                    next.isFile = true
                }
                current = next
            }
        }

        var lines: [String] = []
        // Render ogni radice di primo livello (un progetto può
        // avere più sourcePaths con root distinte).
        for (i, name) in root.childOrder.enumerated() {
            let isLast = (i == root.childOrder.count - 1)
            Self.renderNode(name,
                            node: root.children[name]!,
                            prefix: "",
                            isLast: isLast,
                            depth: 0,
                            maxDepth: maxDepth,
                            into: &lines)
        }
        return lines.joined(separator: "\n")
    }

    fileprivate static func renderNode(_ name: String,
                                        node: _TreeNode,
                                        prefix: String,
                                        isLast: Bool,
                                        depth: Int,
                                        maxDepth: Int,
                                        into lines: inout [String]) {
        let connector = isLast ? "└── " : "├── "
        let isDir = !node.children.isEmpty
        let displayName = isDir ? "\(name)/" : name
        lines.append(prefix + connector + displayName)

        if depth + 1 >= maxDepth, isDir {
            let childPrefix = prefix + (isLast ? "    " : "│   ")
            lines.append(childPrefix + "└── …/")
            return
        }

        let childPrefix = prefix + (isLast ? "    " : "│   ")
        for (i, childName) in node.childOrder.enumerated() {
            let lastChild = (i == node.childOrder.count - 1)
            Self.renderNode(childName,
                            node: node.children[childName]!,
                            prefix: childPrefix,
                            isLast: lastChild,
                            depth: depth + 1,
                            maxDepth: maxDepth,
                            into: &lines)
        }
    }
}

/// Builder che cammina le `sourcePaths` del progetto via
/// `ProjectIndexer.scan(_:)` e applica la cap configurata
/// (per-progetto o globale). Non tokenizza nulla — è il path
/// che salta il tokenizer della modalità `indexedContent`.
enum ProjectInventoryBuilder {

    /// Cap di default sul numero di file mostrati nell'albero.
    static let defaultMaxFiles = 500

    /// Default `maxDepth` allineato con il pattern di
    /// `repo_overview` tool.
    static let defaultMaxDepth = 3

    static func build(_ project: Project,
                      maxFiles: Int,
                      maxDepth: Int = defaultMaxDepth) -> ProjectInventory {
        let candidates = ProjectIndexer.scan(project)
        let total = candidates.count

        // Sort by displayPath per stabilità: il modello vede
        // sempre lo stesso ordine, il troncamento è prevedibile
        // (top-N alfabetico).
        let sorted = candidates.sorted { $0.displayPath < $1.displayPath }
        let trimmed = Array(sorted.prefix(maxFiles))
        let fm = FileManager.default
        let entries = trimmed.map { c -> ProjectFileEntry in
            let mtime = (try? fm.attributesOfItem(atPath: c.url.path))?
                [.modificationDate] as? Date
            return ProjectFileEntry(
                relativePath: c.displayPath,
                byteCount: c.byteCount,
                lastModified: mtime)
        }

        let roots = Array(Set(project.sourcePaths.compactMap {
            URL(fileURLWithPath: $0).lastPathComponent
        })).sorted()

        return ProjectInventory(
            projectName: project.name,
            entries: entries,
            truncated: trimmed.count < total,
            totalDiscovered: total,
            rootDisplayNames: roots)
    }
}
