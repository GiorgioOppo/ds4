import Foundation
import DS4Core

extension ToolRegistry {
    /// MODIFY (replace) a line range.
    static let fileModify = BuiltinTool(
        spec: ToolSpec(name: "file_modify",
                       description: "MODIFICA un file sostituendo le righe [from_line, to_line] (1-based, incluse) con 'content' (to_line omesso = una sola riga; 'content' vuoto = cancella quelle righe). Il file deve esistere. Per sostituzioni su testo esatto preferisci project_edit.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"righe sostitutive (vuoto = cancella)"},"from_line":{"type":"number","description":"prima riga da sostituire, 1-based"},"to_line":{"type":"number","description":"ultima riga inclusa, 1-based (opzionale = from_line)"}},"required":["path","content","from_line"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            guard let f = intArg(argsJSON, "from_line") else { return "Argomento 'from_line' mancante." }
            return ProjectCache.shared.modifyLinesTool(path: p, content: c, fromLine: f, toLine: intArg(argsJSON, "to_line"))
        })
}
