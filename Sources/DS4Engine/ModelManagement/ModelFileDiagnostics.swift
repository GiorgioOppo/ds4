import Foundation

/// Pre-flight explanation of why a model path cannot be opened, so the user
/// sees the actual cause and the fix instead of a bare "cannot open model".
///
/// The classic failure this decodes: a sandboxed build resolves the managed
/// models directory INSIDE the app container, while a previous non-sandboxed
/// build downloaded the same file into the plain `~/Library/Application
/// Support`. The sandbox denies reading the legacy location, so the app can
/// only point the user at the one-line `mv` that migrates the file.
public enum ModelFileDiagnostics {
    /// nil when `path` looks openable; otherwise a message that states what
    /// is wrong and what to do about it.
    public static func openabilityIssue(path: String) -> String? {
        guard !path.isEmpty else { return "nessun modello selezionato" }
        let manager = FileManager.default
        if manager.fileExists(atPath: path) {
            if manager.isReadableFile(atPath: path) { return nil }
            return "il file esiste ma non è leggibile (permessi o sandbox): \(path)"
        }

        var hints = ["file non trovato: \(path)"]
        let partPath = path + ".part"
        if manager.fileExists(atPath: partPath) {
            let bytes = (try? manager.attributesOfItem(atPath: partPath)[.size])
                .flatMap { $0 as? Int64 }
            let size = bytes.map { " (\($0) byte scaricati)" } ?? ""
            hints.append(
                "esiste un download incompleto \((partPath as NSString).lastPathComponent)\(size): riprendi il download dall'app")
        }
        if let legacy = legacyLocation(forContainerPath: path) {
            if manager.fileExists(atPath: legacy) {
                hints.append(
                    "trovato nella posizione legacy non-sandbox: spostalo con  mv \"\(legacy)\" \"\(path)\"")
            } else {
                hints.append(
                    "se il file era stato scaricato da una build non-sandbox, sta in \"\(legacy)\" (invisibile alla sandbox): spostalo da Terminale con  mv \"\(legacy)\" \"\(path)\"  oppure ri-scarica/ri-seleziona il modello")
            }
        }
        return hints.joined(separator: "; ")
    }

    /// For a path inside an app container
    /// (`<home>/Library/Containers/<bundle>/Data/<rest>`), the equivalent
    /// non-sandboxed location `<home>/<rest>`; nil for non-container paths.
    static func legacyLocation(forContainerPath path: String) -> String? {
        guard let markerRange = path.range(of: "/Library/Containers/") else {
            return nil
        }
        let home = String(path[..<markerRange.lowerBound])
        let afterMarker = path[markerRange.upperBound...]
        guard let dataRange = afterMarker.range(of: "/Data/") else { return nil }
        let rest = String(afterMarker[dataRange.upperBound...])
        guard !home.isEmpty, !rest.isEmpty else { return nil }
        return home + "/" + rest
    }
}
