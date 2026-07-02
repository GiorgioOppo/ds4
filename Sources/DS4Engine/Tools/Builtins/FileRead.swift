import Foundation
import DS4Core

extension ToolRegistry {
    /// Read any file inside the project root (raw, not limited to the index),
    /// optionally a line range [from_line, to_line].
    static let fileRead = BuiltinTool(
        spec: ToolSpec(name: "file_read",
                       description: "Leggi un file QUALSIASI dentro la radice del progetto importato (anche non indicizzato). Senza from_line/to_line restituisce l'intero file (cap 96 KB); con from_line/to_line (1-based, inclusi) restituisce SOLO quelle righe, numerate.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"from_line":{"type":"number","description":"prima riga, 1-based (opzionale)"},"to_line":{"type":"number","description":"ultima riga inclusa, 1-based (opzionale)"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            return ProjectCache.shared.readFileTool(path: p,
                                                    fromLine: intArg(argsJSON, "from_line"),
                                                    toLine: intArg(argsJSON, "to_line"))
        })
}
