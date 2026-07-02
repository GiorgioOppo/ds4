import Foundation
import DS4Core

extension ToolRegistry {
    /// Count the lines of a file (for choosing line ranges).
    static let fileLines = BuiltinTool(
        spec: ToolSpec(name: "file_lines",
                       description: "Conta le righe (e i byte) di un file dentro la radice del progetto. Utile PRIMA di file_read/file_modify/file_add per scegliere correttamente gli intervalli di riga.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            return ProjectCache.shared.lineCountTool(path: p)
        })
}
