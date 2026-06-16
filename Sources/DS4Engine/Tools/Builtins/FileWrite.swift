import Foundation
import DS4Core

extension ToolRegistry {
    /// Create/overwrite the WHOLE file inside the project root.
    static let fileWrite = BuiltinTool(
        spec: ToolSpec(name: "file_write",
                       description: "Crea o sovrascrivi l'INTERO file dentro la radice del progetto importato (qualunque estensione; crea le cartelle). Per AGGIUNGERE righe usa file_add, per MODIFICARE righe esistenti usa file_modify.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"contenuto completo del file"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.writeFileTool(path: p, content: c)
        })
}
