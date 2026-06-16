import Foundation
import DS4Core

extension ToolRegistry {
    /// ADD lines (insert) without overwriting.
    static let fileAdd = BuiltinTool(
        spec: ToolSpec(name: "file_add",
                       description: "AGGIUNGI righe a un file (senza sovrascrivere): inserisce 'content' PRIMA della riga 'at_line' (1-based); senza 'at_line' accoda in fondo. Crea il file se non esiste.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"righe da inserire"},"at_line":{"type":"number","description":"inserisci prima di questa riga, 1-based (opzionale: in coda)"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.addLinesTool(path: p, content: c, atLine: intArg(argsJSON, "at_line"))
        })
}
