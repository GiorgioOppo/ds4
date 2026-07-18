import Foundation

/// Logger comune dei motori: UNA convenzione per le righe diagnostiche su
/// stderr (che la GUI raccoglie nel Log motore). Formato storico
/// "DS4 <tag>: <messaggio>" — già usato a mano ovunque; centralizzarlo rende
/// i prefissi uniformi e cercabili. Tag in uso: "glm", "expbundle", "bench",
/// "server", "q4cache", "dist".
public enum DS4Log {
    /// Una write POSIX singola su stderr, senza buffering: visibile in
    /// tempo reale e sicura da qualunque thread.
    public static func info(_ tag: String, _ message: String) {
        FileHandle.standardError.write(
            Data("DS4 \(tag): \(message)\n".utf8))
    }
}
